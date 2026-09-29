#!/usr/bin/env bash
# RAGLoRA 一键停止：FastAPI 后端 → Redis
# 用法: bash backend/shutdown.sh
set -u

ROOT_DIR="D:/桌面/RAGLoRA"
PORT="${RAGLORA_PORT:-8000}"

echo "============================================================"
echo " RAGLoRA 停止"
echo "============================================================"

stop_port() {
  local port="$1" name="$2"
  local pid
  pid=$(netstat -ano 2>/dev/null | grep -E ":${port}\s.*LISTENING" | head -1 | awk '{print $NF}' | tr -d '\r')
  if [ -n "$pid" ]; then
    echo "[$name] 结束 PID $pid (端口 $port)"
    taskkill //PID "$pid" //F >/dev/null 2>&1
    sleep 1
  else
    echo "[$name] 未在运行"
  fi
}

stop_port "$PORT" "backend"
bash "$ROOT_DIR/tools/redis/stop_redis.sh"

echo ""
echo "已全部停止。"
