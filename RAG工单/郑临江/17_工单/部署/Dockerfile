FROM python:3.12-slim
WORKDIR /app
COPY 研发/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY 研发/ .
EXPOSE 8000
CMD ["python", "api_server.py"]
