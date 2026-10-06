FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_REQUIRE_HASHES=0 PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
COPY app ./app
COPY celery_worker.py .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
