#!/usr/bin/env bash
# ============================================================
# 停止脚本：停止 FastAPI 后端进程
# 用法：bash scripts/shutdown.sh
# ============================================================
# set -euo pipefail 严格模式：出错即停、未定义变量即报错、管道失败即失败
set -euo pipefail

# 定位项目根目录，PID 文件固定放在 data/ 下（与 run.sh 写入的位置一致）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/data/uvicorn.pid"

# 无 PID 文件时回退：按进程名模糊匹配停止，避免残留后台进程
if [ ! -f "$PID_FILE" ]; then
  echo "[i] 未找到 PID 文件，尝试按进程名停止 ..."
  pkill -f "uvicorn src.main:app" && echo "[√] 已停止 uvicorn 进程" || echo "[i] 没有运行中的服务"
  exit 0
fi

PID="$(cat "$PID_FILE")"
# kill -0 只探测进程是否存在、不发送终止信号，用于确认 PID 仍有效
if kill -0 "$PID" 2>/dev/null; then
  echo "[*] 停止服务 PID=$PID ..."
  # 先发 SIGTERM（kill 默认信号），让 uvicorn 优雅收尾：释放端口、落盘、清理连接
  kill "$PID"
  # 轮询最多 10 秒，等待进程响应 SIGTERM 自行退出；一旦退出即 break
  for _ in $(seq 1 10); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  # 超过宽限期仍未退出，说明进程卡死/忽略 SIGTERM，改用 SIGKILL(-9) 强制结束
  if kill -0 "$PID" 2>/dev/null; then
    echo "[!] 进程未退出，强制结束"
    kill -9 "$PID" || true
  fi
  echo "[√] 服务已停止"
else
  # PID 文件里的进程已不存在，可能是上次异常退出留下的残留文件
  echo "[i] PID $PID 不存在，服务可能已停止"
fi

# 清理 PID 文件，避免下次启动时因"文件存在但进程无效"产生误判
rm -f "$PID_FILE"