FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app.py corpus.py evaluation.py graph_rag.py prepare.py questions.py ./
COPY --chown=appuser:appuser index.html eval_questions.md sample_questions.md evaluation_report.json ./
COPY --chown=appuser:appuser data ./data

USER appuser
EXPOSE 8020

HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 \
  CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8020/api/health',timeout=5))['ready']"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "1"]
