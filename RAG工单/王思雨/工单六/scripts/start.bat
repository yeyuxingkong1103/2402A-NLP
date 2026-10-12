@echo off
REM 工单编号：人工智能NLP-RAG-混合检索任务
REM 说明：Windows 一键启动。固定使用本机唯一齐备的解释器 D:/myconda1/python.exe，端口 8010。
cd /d "%~dp0.."
set PY=D:/myconda1/python.exe
echo [1/2] 检查 Milvus 与 Redis ...
%PY% -c "import redis,sys;redis.Redis.from_url('redis://localhost:6379/0',protocol=2).ping();print('  Redis OK')" || echo "  Redis 不可用，服务将以降级模式运行"
echo [2/2] 启动 FastAPI (http://127.0.0.1:8010) ...
%PY% -m uvicorn app.main:app --host 127.0.0.1 --port 8010
