FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's#http://deb.debian.org/debian#http://mirrors.aliyun.com/debian#g; s#http://deb.debian.org/debian-security#http://mirrors.aliyun.com/debian-security#g' /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

COPY backend ./backend
COPY data_pipeline ./data_pipeline
COPY frontend ./frontend
COPY run.py ./run.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=7294

EXPOSE 7294
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "run.py"]
