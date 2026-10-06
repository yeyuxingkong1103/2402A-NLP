# 基于 PDF 文档的问答系统优化版 V2 - Dockerfile
# 工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PDF_PATH=/data/招股说明书1.pdf \
    LLM_API_KEY= \
    LLM_BASE_URL=https://api.openai.com/v1 \
    LLM_MODEL=gpt-3.5-turbo

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY 部署/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY 研发 /app

EXPOSE 5001

CMD ["python", "app_v2.py"]
