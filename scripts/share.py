#!/usr/bin/env python3
"""一条命令把本地界面发布成公网链接（无需账号、无需备案）。

做法：先在本机起 Gradio（若 7860 已在监听则直接复用），再用 Cloudflare
Quick Tunnel 把本地端口映射成一个 https://xxx.trycloudflare.com 地址。

用法：
    python scripts/share.py                # 默认 7860 端口
    python scripts/share.py --port 7861    # 换端口
    python scripts/share.py --no-app       # 只用隧道连已有的服务

产物：终端打印公网地址，同时写入 PUBLIC_URL.txt（已 gitignore，避免过期链接进版本库）。
说明：这是**临时**隧道，进程结束即失效，域名每次重新生成。
需要长期稳定的地址，请用 GitHub Codespaces（见 .devcontainer）或 Docker 部署。
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "person-counter"
URL_FILE = ROOT / "PUBLIC_URL.txt"

CF_RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
CF_ASSETS = {
    ("Windows", "AMD64"): "cloudflared-windows-amd64.exe",
    ("Windows", "x86"): "cloudflared-windows-386.exe",
    ("Linux", "x86_64"): "cloudflared-linux-amd64",
    ("Linux", "aarch64"): "cloudflared-linux-arm64",
    ("Darwin", "arm64"): "cloudflared-darwin-arm64.tgz",
    ("Darwin", "x86_64"): "cloudflared-darwin-amd64.tgz",
}


def log(msg: str) -> None:
    print(f"[share] {msg}", flush=True)


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def ensure_cloudflared(explicit: str | None = None) -> Path:
    """返回可用的 cloudflared 路径：显式指定 > 环境变量 > 缓存 > PATH > 自动下载。"""
    explicit = explicit or os.getenv("CLOUDFLARED_BIN")
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise SystemExit(f"指定的 cloudflared 不存在：{p}")
        return p

    CACHE.mkdir(parents=True, exist_ok=True)
    exe = CACHE / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
    if exe.exists() and exe.stat().st_size > 1_000_000:
        return exe

    import shutil

    found = shutil.which("cloudflared")
    if found:
        log(f"复用 PATH 中的 cloudflared：{found}")
        return Path(found)

    key = (platform.system(), platform.machine())
    asset = CF_ASSETS.get(key)
    if asset is None:
        raise SystemExit(f"暂不支持的平台 {key}，请手动安装 cloudflared 后重试")

    log(f"下载 cloudflared（{asset}）...")
    url = CF_RELEASE + asset
    if asset.endswith(".tgz"):
        import tarfile
        import urllib.request

        tmp = CACHE / asset
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("cloudflared"))
            member.name = exe.name
            tf.extract(member, CACHE)
    else:
        import urllib.request

        urllib.request.urlretrieve(url, exe)

    if os.name != "nt":
        exe.chmod(0o755)
    log(f"已就绪：{exe}")
    return exe


def start_app(port: int) -> subprocess.Popen | None:
    """端口空闲才起服务；已占用则复用，避免起两份模型。"""
    if port_open("127.0.0.1", port):
        log(f"{port} 端口已有服务在跑，直接复用")
        return None

    log("启动 Gradio 服务 ...")
    log_path = CACHE / "app.log"
    fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py"), "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for _ in range(120):
        if port_open("127.0.0.1", port):
            log("服务已就绪")
            return proc
        if proc.poll() is not None:
            raise SystemExit(f"服务启动失败，日志：{log_path}")
        time.sleep(1)
    raise SystemExit(f"等待服务超时，日志：{log_path}")


def start_tunnel(cf: Path, port: int) -> tuple[subprocess.Popen, str]:
    log("建立 Cloudflare 隧道 ...")
    proc = subprocess.Popen(
        [str(cf), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 90
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise SystemExit("隧道进程意外退出，请检查网络是否能访问 Cloudflare")
            continue
        m = pattern.search(line)
        if m:
            return proc, m.group(0)
    raise SystemExit("90 秒内没拿到公网地址，请检查网络是否能访问 trycloudflare.com")


def main() -> None:
    parser = argparse.ArgumentParser(description="把本地界面发布成公网链接")
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--no-app", action="store_true", help="不启动本地服务，只建隧道")
    parser.add_argument(
        "--cloudflared",
        default=None,
        help="已有 cloudflared 可执行文件路径（GitHub 下载慢时用这个指定）",
    )
    args = parser.parse_args()

    cf = ensure_cloudflared(args.cloudflared)
    app_proc = None if args.no_app else start_app(args.port)
    tunnel_proc, url = start_tunnel(cf, args.port)

    URL_FILE.write_text(url + "\n", encoding="utf-8")
    print()
    print("=" * 62)
    print(f"  公网地址：{url}")
    print(f"  已写入：{URL_FILE}")
    print("  提示：临时隧道，关掉本窗口即失效；域名每次重新生成")
    print("=" * 62)
    print("按 Ctrl+C 结束\n")

    try:
        while tunnel_proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        log("正在关闭 ...")
    finally:
        tunnel_proc.terminate()
        if app_proc is not None:
            app_proc.terminate()


if __name__ == "__main__":
    main()
