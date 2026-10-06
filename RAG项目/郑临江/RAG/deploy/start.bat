@echo off
chcp 65001 >nul
rem 切换到项目根目录（本文件位于 deploy\ 下）
cd /d "%~dp0.."

set PY=D:\an\envs\rags_\python.exe

echo [1/3] 检查 Redis ...
%PY% -c "import redis; r=redis.Redis(host='127.0.0.1', port=6379); r.ping(); print('    Redis OK')" 2>nul
if errorlevel 1 (
    echo    [警告] Redis 不可用，请先启动 Redis 服务。
)

echo [2/3] 检查 Ollama ...
%PY% -c "import httpx; httpx.get('http://localhost:11434/api/tags', timeout=3).raise_for_status(); print('    Ollama OK')" 2>nul
if errorlevel 1 (
    echo    [警告] Ollama 不可用，请先启动 Ollama。
)

echo [3/3] 启动服务（http://127.0.0.1:8000）...
%PY% run.py
pause
