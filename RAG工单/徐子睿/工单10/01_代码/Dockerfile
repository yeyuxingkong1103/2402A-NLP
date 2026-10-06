# 工单编号：人工智能NLP-RAG-金融问答系统部署任务
# 关联工单：人工智能NLP-RAG-金融问答系统部署任务
# 说明：金融问答系统（Graph RAG）运行镜像。索引/图谱为离线产物，镜像只含服务端代码。
#       嵌入与生成默认走外部 Ollama（bge-m3 / qwen2:7b），通过 OLLAMA_BASE 指定。
FROM python:3.11-slim

# 工单编号：人工智能NLP-RAG-金融问答系统部署任务
LABEL work_order="人工智能NLP-RAG-金融问答系统部署任务"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    DATA_DIR=/data \
    OLLAMA_BASE=http://host.docker.internal:11434

WORKDIR /app

COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# 索引与图谱挂载到 /data（data/index、data/graph、data/ccf…）
VOLUME ["/data"]

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/api/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8100"]
