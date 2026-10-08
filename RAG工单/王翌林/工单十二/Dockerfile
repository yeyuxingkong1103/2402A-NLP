# 工单编号：人工智能NLP-RAG-金融问答系统部署
# Dockerfile —— 工单十 金融问答系统（FastAPI v6 + Streamlit v6）一体化镜像
# 基础镜像 python:3.10-slim；模型权重通过 bind mount /home/dabaie/models 复用，
# 不打入镜像；数据目录 /app/data 由 named volume rag-data 持久化。
FROM python:3.10-slim

# 工单十：构建参数与基础环境
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    HF_HUB_OFFLINE=1 \
    RAG_EMBED_DEVICE=cpu

WORKDIR /app

# 工单十：分层安装依赖（requirements 变更频率低，充分利用构建缓存）
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/requirements-docker.txt

# 工单十：拷贝应用代码（.dockerignore 已排除 附件/日志/缓存/.env 等）
COPY src /app/src
COPY app /app/app
COPY scripts /app/scripts
COPY data /app/data

EXPOSE 8006 8506

# 工单十：默认启动 FastAPI v6；Streamlit 服务由 docker-compose 覆盖 command
CMD ["python", "-m", "uvicorn", "src.api_v6:app", "--host", "0.0.0.0", "--port", "8006"]
