# 工单编号：人工智能NLP-RAG-金融问答系统部署
# Dockerfile - 金融问答系统容器化部署

FROM python:3.12-slim

LABEL maintainer="人工智能NLP-RAG项目"
LABEL description="金融问答系统 - Graph RAG"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目代码
COPY *.py ./
COPY docs/ ./docs/

# 创建数据目录
RUN mkdir -p /app/data /app/models /app/logs

# 暴露端口
EXPOSE 6006

# 环境变量
ENV HF_HOME=/app/models
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONUNBUFFERED=1

# 数据卷（持久化）
VOLUME ["/app/data", "/app/models", "/app/logs"]

# 启动命令
CMD ["python3", "app.py"]
