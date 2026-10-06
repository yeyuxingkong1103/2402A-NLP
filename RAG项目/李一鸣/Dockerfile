FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY run.py .
COPY .env.example .
RUN mkdir -p data/uploads data/indexes logs

EXPOSE 8000
CMD ["python", "run.py"]
