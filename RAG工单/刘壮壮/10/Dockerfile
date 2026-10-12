# 工单编号：人工智能NLP-RAG-金融问答系统部署
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    PORT=8000

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码
COPY rag_core.py .
COPY app.py .

# 数据卷挂载点
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "app.py"]
