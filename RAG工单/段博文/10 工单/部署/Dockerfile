# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# ============================================================
# RAG 金融问答系统 Dockerfile
# ============================================================
# 构建：docker build -t rag-finance:latest .
# 运行：docker run -d -p 8000:8000 -v rag-data:/app/data -v rag-models:/models --network rag-net rag-finance:latest
# ============================================================

FROM python:3.10-slim

LABEL maintainer="RAG Project"
LABEL description="RAG 金融问答系统 - 混合检索 + 多轮对话 + RAGAS 评估"

# 设置工作目录
WORKDIR /app

# 设置环境变量（避免交互式问题）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 安装系统依赖（PyMuPDF、jieba、OCR 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 并安装 Python 依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# 复制应用代码
COPY src/ /app/src/
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/.env.example /app/.env.example

# 赋予执行权限
RUN chmod +x /app/entrypoint.sh

# 创建数据目录（将被 Docker 卷挂载）
RUN mkdir -p /app/data /models

# 暴露服务端口
EXPOSE 8000

# 健康检查：每 30 秒检查一次，超时 10 秒，3 次失败判定不健康
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)" || exit 1

# 启动命令
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
