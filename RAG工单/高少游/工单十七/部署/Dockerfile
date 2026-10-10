# 工单编号: 人工智能 NLP-RAG-解决API服务并发瓶颈与资源泄漏
# RAG API 服务镜像（API Server 与 Task Executor 共用）
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖：PDF 解析（pdfplumber/pdfminer）所需的字体与图形库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 代码与索引（vector_db / data 由 compose 以只读卷挂载覆盖）
COPY . /app

EXPOSE 9380

# 默认启动优化版 API 服务（等价 run_server.py --variant optimized）
CMD ["uvicorn", "api.app_optimized:app", "--host", "0.0.0.0", "--port", "9380", "--workers", "1"]