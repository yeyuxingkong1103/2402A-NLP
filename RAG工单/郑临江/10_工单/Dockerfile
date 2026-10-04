# 工单编号：人工智能NLP-RAG-金融问答系统部署
FROM python:3.12-slim

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码
COPY . .

# 数据通过卷挂载到 /data（不打包进镜像，保证数据可持久化、可共享）
ENV DATA_DIR=/data \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# 容器数据卷声明
VOLUME ["/data"]

CMD ["python", "app.py"]
