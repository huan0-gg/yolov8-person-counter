# 部署与公网访问

三种方式，按「给别人看」的需求从轻到重排列。所有方式都不需要云服务器备案，
也不依赖 HuggingFace（本机网络对 HF 不可达，`gradio.live` 隧道因此用不了）。

| 方式 | 拿到什么 | 有效期 | 需要账号 | 适合场景 |
|---|---|---|---|---|
| `python scripts/share.py` | `https://xxx.trycloudflare.com` | 进程活着期间 | 不需要 | 临时发给同学/老师看一眼 |
| GitHub Codespaces | 云端实例的公网转发地址 | 代码空间存在期间 | GitHub 账号 | 别人从仓库一键跑起来看 |
| Docker / 自建服务器 | 自己的域名或 IP:7860 | 长期 | 需要服务器 | 长期演示、答辩现场备份 |

---

## 1. 临时公网链接（最快）

```bash
python scripts/share.py
```

脚本会做三件事：

1. 检查 7860 端口，没有服务就自己起一个（有就直接复用，不会起两份模型）；
2. 首次运行会把 `cloudflared` 下到 `~/.cache/person-counter/`；
3. 建立 Cloudflare Quick Tunnel，打印公网地址并写入 `PUBLIC_URL.txt`。

换端口：`python scripts/share.py --port 7861`
只建隧道、不起服务：`python scripts/share.py --no-app`

**注意**：这是临时隧道，关掉终端就断，域名每次都不一样。别把这个地址写进文档或 PPT 定稿。

---

## 2. GitHub Codespaces（推荐给"别人直接使用"）

仓库根目录已带 `.devcontainer/`，别人打开仓库后：

1. 点绿色 **Code** → **Codespaces** → **Create codespace on main**；
2. 等 3～5 分钟（首次装 CPU 版 torch 最慢，之后有缓存）；
3. 容器启动会自动执行 `scripts/dev_start.sh` 拉起服务；
4. 端口面板里 **7860** 已设为 `public` + 自动打开预览，
   地址形如 `https://<codespace-name>-7860.app.github.dev`，直接发给别人就能看。

关键配置（`.devcontainer/devcontainer.json`）：

- `portsAttributes."7860".visibility = "public"` —— 否则链接会要求登录 GitHub；
- `postCreateCommand: bash scripts/dev_setup.sh` —— 幂等安装依赖，重开不重装；
- `postStartCommand: bash scripts/dev_start.sh` —— 端口已占用则跳过，避免重复启动。

免费额度：个人账号每月 120 core-hours（2 核机器约 60 小时），够演示用；
不用时记得在 Codespaces 页面 Stop，否则会一直计费。

---

## 3. Docker（长期部署）

```bash
docker build -t person-counter .
docker run --rm -p 7860:7860 person-counter
```

镜像基于 `python:3.12-slim`，只装 CPU 版 torch（约 200MB，不是 2GB 的 CUDA 版），
`requirements-app.txt` 只含界面所需依赖，不含训练/评测用的 pandas、matplotlib、pycocotools。

任何支持容器的平台（Render、Railway、Fly.io、阿里云/腾讯云轻量服务器）都能直接用它，
平台会自动读取 `EXPOSE 7860`。

---

## 常见问题

**Q：为什么不用 Gradio 自带的 `--share`？**
A：它依赖 `cdn-media.huggingface.co` 下载隧道二进制，本机网络不可达，会报
`Could not create share link. Missing file: ... frpc_windows_amd64_v0.3`。

**Q：为什么不用 HuggingFace Spaces？**
A：部署需要 `HF_TOKEN`，且本机到 `huggingface.co` 超时。如果之后能访问，
把本仓库当 Space 推上去即可，`Dockerfile` 与 `app.py` 已按 Spaces 要求支持
`--host 0.0.0.0` 和环境变量配置端口。

**Q：公网链接能撑住答辩现场吗？**
A：临时隧道受本地网络和 Cloudflare 波动影响。答辩建议**本地跑为主**，
公网链接作为"扫码自己玩"的补充；必要时提前用 Codespaces 备一份。

**Q：摄像头页签在公网能用吗？**
A：浏览器只允许 `https` 或 `localhost` 调用摄像头。公网地址是 https，可以用；
但摄像头画面要先传到本地机器再推理，延迟取决于上行带宽。
