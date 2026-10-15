# 工单编号：人工智能NLP-RAG-金融问答系统部署
FROM python:3.11-slim
WORKDIR /app
COPY 研发/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY 研发/app /app/app
COPY 研发/static /app/static
COPY 研发/project.json /app/project.json
ENV EMBEDDING_BACKEND=tfidf PYTHONUNBUFFERED=1 LLM_PROVIDER=ollama LLM_MODEL=Qwen3.5:4B LLM_BASE_URL=http://host.docker.internal:11434 LLM_TIMEOUT=120
EXPOSE 8774
CMD ["python","-m","app.main","serve","--host","0.0.0.0","--port","8774"]
