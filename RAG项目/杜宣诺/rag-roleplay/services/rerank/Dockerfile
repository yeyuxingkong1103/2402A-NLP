FROM python:3.11-slim

WORKDIR /app

# 先装 CPU 版 torch，避免 FlagEmbedding 拉 GPU 版
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries 8 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app.py .

ENV RERANK_MODEL=BAAI/bge-reranker-base
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
