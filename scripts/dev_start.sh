#!/usr/bin/env bash
# 云端容器每次启动时拉起 Gradio 服务（幂等：端口已占用就不再起第二个）
set -uo pipefail

cd "$(dirname "$0")/.."

PORT="${GRADIO_SERVER_PORT:-7860}"
LOG="${TMPDIR:-/tmp}/app.log"

if ! python -c "import gradio, ultralytics, cv2" >/dev/null 2>&1; then
  echo "[dev_start] 依赖未装好，先执行 scripts/dev_setup.sh"
  exit 0
fi

if (echo >"/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
  echo "[dev_start] ${PORT} 已在监听，跳过启动"
  exit 0
fi

echo "[dev_start] 启动服务，日志：${LOG}"
nohup python app.py --host 0.0.0.0 --port "${PORT}" >"${LOG}" 2>&1 &

sleep 12
tail -n 15 "${LOG}" || true
