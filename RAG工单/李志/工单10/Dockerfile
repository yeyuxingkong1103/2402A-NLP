FROM python:3.12-slim
WORKDIR /app
COPY main.py /app/main.py
COPY ragkit.py /app/ragkit.py
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "main.py"]
