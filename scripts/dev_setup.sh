#!/usr/bin/env bash
# 云端容器（Codespaces / Docker / CI）首次构建：安装 CPU 版依赖
# 幂等：已装过就跳过，避免每次重建都重下载 torch
set -euo pipefail

cd "$(dirname "$0")/.."

if python -c "import gradio, ultralytics, cv2" >/dev/null 2>&1; then
  echo "[dev_setup] 依赖已就绪，跳过安装"
  exit 0
fi

echo "[dev_setup] 安装 CPU 版 torch ..."
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu

echo "[dev_setup] 安装界面依赖 ..."
python -m pip install --no-cache-dir -r requirements-app.txt

echo "[dev_setup] 完成"
