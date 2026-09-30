#!/usr/bin/env bash
# RAGLoRA 一键启动：Redis → 依赖探活 → FastAPI 后端
#
# 用法:  bash backend/run.sh            # 前台启动
#        bash backend/run.sh --bg       # 后台启动（日志写 logs/uvicorn.log）
set -u

BACKEND_DIR="D:/桌面/RAGLoRA/backend"
ROOT_DIR="D:/桌面/RAGLoRA"
PY="D:/anaconda3/envs/rag_env/python.exe"
PORT="${RAGLORA_PORT:-8000}"

echo "============================================================"
echo " RAGLoRA 启动"
echo "============================================================"

# ---------- 0. 环境自检 ----------
if [ ! -x "$PY" ]; then
  echo "[错误] 找不到 Python 环境: $PY"
  echo "       本项目固定使用 rag_env，请勿改用其它环境"
  exit 1
fi
echo "[1/4] Python 环境 OK  ($("$PY" -c 'import sys;print(sys.version.split()[0])'))"

# ---------- 1. Redis（短期记忆，非必须）----------
echo "[2/4] 启动 Redis ..."
bash "$ROOT_DIR/tools/redis/start_redis.sh" || echo "      Redis 启动失败，短期记忆将降级为 MySQL 回源"

# ---------- 2. 依赖探活 ----------
echo "[3/4] 依赖探活 ..."
if netstat -ano 2>/dev/null | grep -qE ":3306\s.*LISTENING"; then
  echo "      MySQL    OK (127.0.0.1:3306)"
else
  echo "      MySQL    未监听 —— 后端将无法启动，请先启动 MySQL 服务"
  exit 1
fi

if curl -s --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "      Ollama   OK (127.0.0.1:11434)"
else
  echo "      Ollama   未响应 —— 生成功能不可用，请先启动 Ollama"
fi

# 向量库探活。默认是 Milvus（需 Docker），起不来时给出可执行的补救命令，
# 而不是让用户到第一次检索才撞上 500。
STORE="${RAGLORA_VECTOR_STORE:-milvus}"
if [ "$STORE" = "qdrant" ]; then
  echo "      Qdrant   嵌入式模式，随后端进程启动（无需单独服务）"
elif netstat -ano 2>/dev/null | grep -qE ":19530\s.*LISTENING"; then
  echo "      Milvus   OK (127.0.0.1:19530)"
else
  echo "      Milvus   未监听（VECTOR_STORE=$STORE）—— 检索将不可用"
  echo "               补救：启动 Docker Desktop 后执行  bash tools/demo/up.sh milvus"
  echo "               或临时回退：RAGLORA_VECTOR_STORE=qdrant bash backend/run.sh --bg"
fi

# ---------- 3. 启动后端 ----------
echo "[4/4] 启动 FastAPI ..."
cd "$BACKEND_DIR" || exit 1

if [ "${1:-}" = "--bg" ]; then
  mkdir -p logs
  # ⚠️ 必须用 PowerShell Start-Process 真正脱离父进程。
  # 早期版本用 `nohup ... &`，在 Git Bash on Windows 下子进程仍依附于调用它的 shell，
  # shell 一退出后端就可能被一并收走——表现为「进程静默消失、无任何 Python 堆栈、
  # Windows 事件日志里也没有崩溃记录」，极易被误判成程序 bug。
  powershell -NoProfile -Command \
    "Start-Process -FilePath '$PY' -ArgumentList '-m','uvicorn','app.main:app','--host','0.0.0.0','--port','$PORT' -WorkingDirectory 'D:\桌面\RAGLoRA\backend' -WindowStyle Hidden -RedirectStandardOutput 'D:\桌面\RAGLoRA\backend\logs\uvicorn.log' -RedirectStandardError 'D:\桌面\RAGLoRA\backend\logs\uvicorn.err.log'"
  echo "      后台启动中（已脱离父进程）"
  for _ in $(seq 1 30); do          # 最多等 30s（首次启动要建表+种角色）
    sleep 1
    netstat -ano 2>/dev/null | grep -qE ":${PORT}\s.*LISTENING" && break
  done
  if netstat -ano 2>/dev/null | grep -qE ":${PORT}\s.*LISTENING"; then
    echo ""
    echo "============================================================"
    echo " 启动成功"
    echo "   接口文档  http://127.0.0.1:${PORT}/docs"
    echo "   健康检查  http://127.0.0.1:${PORT}/api/health?deep=1"
    echo "   日志      $BACKEND_DIR/logs/uvicorn.log"
    echo "============================================================"
  else
    echo "      启动失败，请查看 logs/uvicorn.log"
    exit 1
  fi
else
  echo "      前台启动（Ctrl+C 停止）"
  echo ""
  exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --log-level info
fi
