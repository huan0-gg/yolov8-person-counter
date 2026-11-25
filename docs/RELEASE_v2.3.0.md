## 这一版解决的问题

代码在你机器上能跑，不代表别人能看。v2.3 把"让别人打开这个项目"做成三条可执行路径，
检测逻辑一行没改，v2.2 的全部指标与测试结论保持不变。

## 三种访问方式

| 方式 | 拿到什么 | 有效期 | 需要账号 |
|---|---|---|---|
| `python scripts/share.py` | `https://xxx.trycloudflare.com` | 进程存活期间 | 不需要 |
| GitHub Codespaces 一键运行 | `https://<codespace>-7860.app.github.dev` | 代码空间存活期间 | GitHub 账号 |
| Docker 镜像 | 自己的域名或 IP:7860 | 长期 | 需要服务器 |

## 新增

- **启动参数可配置**：`app.py` 支持 `--host` / `--port` / `--share` / `--no-share`，
  以及 `GRADIO_SERVER_NAME`、`GRADIO_SERVER_PORT`、`GRADIO_SHARE` 环境变量；
  默认仍是 `127.0.0.1:7860`，容器里用 `--host 0.0.0.0`。
- **`scripts/share.py`**：自动复用已在跑的服务（不会起第二份模型），首次运行自动下载
  `cloudflared`，打印公网地址并写入 `PUBLIC_URL.txt`（已 gitignore，防过期链接入库）。
  支持 `--port` / `--no-app` / `--cloudflared <路径>`。
- **`.devcontainer/`**：Codespaces 一键创建，7860 端口 `visibility: public`，
  `postCreate` 幂等装依赖、`postStart` 幂等起服务。
- **`Dockerfile` + `requirements-app.txt`**：CPU 版 torch（约 200MB，不是 2GB 的 CUDA 轮子），
  带 `HEALTHCHECK`，任何容器平台可直接用。
- **`docs/DEPLOY.md`**：三种方式的取舍、免费额度、公网摄像头限制、排障。

## 为什么不走 Gradio 自带的 `--share`

它要从 `cdn-media.huggingface.co` 下载 `frpc_windows_amd64_v0.3`，本机网络超时，
报 `Could not create share link`。HuggingFace Spaces 同理（`huggingface.co` 不可达），
所以改用 Cloudflare 隧道 + Codespaces 两条路。

## 验证

- `python scripts/share.py` 端到端跑通，公网地址返回 **HTTP 200**（页面 109 KB）；
- `python -m pytest -q` → **31 passed**；
- Codespaces 配置采用官方标准写法，端口公开无需再手动改可见性。

## 使用

```bash
git clone https://github.com/huan0-gg/yolov8-person-counter
cd yolov8-person-counter
pip install -r requirements.txt
python scripts/share.py
```
