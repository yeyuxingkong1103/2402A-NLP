@echo off
REM 工单编号：人工智能NLP-RAG-功能测试及评估 —— 一键启动后端（FastAPI + 前端静态页）
REM 环境：本机 Milvus(19530) 与 Redis(6379) 已在运行，不要再执行 docker compose
chcp 65001 >nul
cd /d "%~dp0.."
set PY=D:/myconda1/python.exe
echo [1/2] 探活 Redis ...
"%PY%" -c "import redis;redis.Redis.from_url('redis://localhost:6379/0',protocol=2).ping();print('  Redis OK')"
if errorlevel 1 (echo   Redis 未启动，请先启动 Redis 后重试 & pause & exit /b 1)
echo [2/2] 启动 API： http://127.0.0.1:8010
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8010
