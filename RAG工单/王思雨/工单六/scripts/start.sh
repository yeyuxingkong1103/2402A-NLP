#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# 说明：Git Bash / Linux 一键启动。固定端口 8010（8000 已被本机其它服务占用）。
set -e
cd "$(dirname "$0")/.."
PY=${PY:-D:/myconda1/python.exe}
echo "[1/2] 检查 Redis ..."
"$PY" -c "import redis;redis.Redis.from_url('redis://localhost:6379/0',protocol=2).ping();print('  Redis OK')" \
  || echo "  Redis 不可用，服务将以降级模式运行"
echo "[2/2] 启动 FastAPI (http://127.0.0.1:8010) ..."
exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8010
