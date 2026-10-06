# 工单编号：人工智能NLP-RAG-金融问答系统部署
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 DATA_DIR=/app/data OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY ["工单9/requirements.txt", "/app/requirements.txt"]
RUN pip install --no-cache-dir -r requirements.txt
COPY ["工单9/app.py", "工单9/index.html", "/app/"]
COPY ["工单9/data/", "/app/data/"]
VOLUME ["/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
