@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Starting Redis...
docker compose -f deploy\docker-compose.redis.yml up -d

echo [2/3] Starting Milvus...
docker compose -f deploy\docker-compose.milvus.yml up -d

start "Education RAG API" cmd /k "cd /d %~dp0 && uvicorn src.edu_rag_ingest.app.rag_api_backend:app --host 0.0.0.0 --port 8000"
start "Education RAG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --host 0.0.0.0"

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
pause
