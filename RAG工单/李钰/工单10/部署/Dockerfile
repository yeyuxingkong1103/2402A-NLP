# ========== 基础镜像 ==========
FROM python:3.9-slim

# ========== 元信息 ==========
LABEL maintainer="RAG Team"
LABEL version="v10.0"
LABEL description="金融问答系统 - V9 Graph RAG 优化版"

# ========== 环境变量 ==========
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_PORT=5008
ENV FLASK_HOST=0.0.0.0
ENV RAG_DATA_DIR=/app/data
ENV RAG_CACHE_DIR=/app/cache
ENV RAG_LOG_DIR=/app/logs
ENV RAG_SHARED_DIR=/app/shared

# ========== 工作目录 ==========
WORKDIR /app

# ========== 复制依赖 ==========
COPY requirements.txt /app/requirements.txt

# ========== 安装依赖 ==========
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# ========== 复制应用代码 ==========
COPY 研发/ /app/

# ========== 创建持久化目录 ==========
RUN mkdir -p /app/data /app/cache /app/logs /app/shared

# ========== 复制入口脚本 ==========
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ========== 暴露端口 ==========
EXPOSE 5008

# ========== 健康检查 (Docker HEALTHCHECK) ==========
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5008/api/health')" || exit 1

# ========== 入口 ==========
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "app_wrapper.py"]
