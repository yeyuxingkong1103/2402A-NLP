# 基于 PDF 文档的问答系统 - Dockerfile
# 工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量 (构建时可通过 --build-arg 覆盖)
ENV PYTHONUNBUFFERED=1 \
    PDF_PATH=/data/招股说明书1.pdf \
    LLM_API_KEY= \
    LLM_BASE_URL=https://api.openai.com/v1 \
    LLM_MODEL=gpt-3.5-turbo

# 安装系统依赖 (pdfplumber 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 拷贝依赖清单并安装
COPY 部署/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 拷贝研发目录代码
COPY 研发 /app

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://127.0.0.1:5000/api/health').raise_for_status()" || exit 1

# 启动命令
CMD ["python", "app.py"]
