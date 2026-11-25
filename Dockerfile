# 云端部署镜像：任何支持 Docker 的平台（Hugging Face Spaces / Render / Fly.io / 自建服务器）通用
# 构建：docker build -t person-counter .
# 运行：docker run -p 7860:7860 person-counter
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    OMP_NUM_THREADS=2

WORKDIR /app

# libgomp1 是 torch 的运行时依赖；opencv 用 headless 版，无需 libGL
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 torch，避免默认源拉 2GB+ 的 CUDA 轮子
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
