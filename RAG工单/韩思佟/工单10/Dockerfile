# 工单10 —— 金融问答系统容器化部署
# 人工智能NLP-RAG-金融问答系统部署
#
# 构建： docker build -t rag-fin-qa:1.0 .
# 运行： docker run -d --name rag-fin-qa -p 8000:8000 -v ragdata:/app/data rag-fin-qa:1.0

FROM python:3.11-slim

LABEL name="rag-fin-qa" \
      description="人工智能NLP-RAG-金融问答系统部署" \
      version="1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TZ=Asia/Shanghai

WORKDIR /app

# 只装运行时依赖（torch CPU 版控制镜像体积）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# 代码
COPY src ./src
COPY scripts ./scripts
COPY data ./data

# 运行时目录（用卷持久化，见 docker-compose.yml）
VOLUME ["/app/outputs", "/app/logs", "/app/cache"]

EXPOSE 8000

# 健康检查：/healthz 由 src.web_api 提供
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.web_api", "--host", "0.0.0.0", "--port", "8000"]
