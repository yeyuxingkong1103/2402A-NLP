#!/usr/bin/env bash
# ============================================================
# shutdown.sh — 停止服务
#
#   bash scripts/shutdown.sh
#
# 与 start.sh 对称，把 FastAPI 与 Streamlit 两个后台进程都关掉。
# 先发 TERM 让服务优雅退出，1 秒后还在的再发 KILL 兜底。
#
# 环境变量：
#   API_PORT    FastAPI 端口，默认 8080
#   WEB_PORT    Streamlit 端口，默认 8501
# ============================================================
set -uo pipefail

API_PORT="${API_PORT:-8080}"
WEB_PORT="${WEB_PORT:-8501}"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# 按进程命令行匹配停止一组服务，返回 0 表示有进程被关掉
stop_group() {
  local label="$1" pattern="$2" pids

  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    info "${label} 未在运行，跳过"
    return 1
  fi

  info "停止 ${label}（PID: ${pids//$'\n'/ }）"
  kill $pids 2>/dev/null || true
  sleep 1

  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    warn "${label} 未响应 TERM，强制结束（PID: ${pids//$'\n'/ }）"
    kill -9 $pids 2>/dev/null || true
  fi
  ok "${label} 已停止"
  return 0
}

echo
info "开始停止服务（端口 ${API_PORT} / ${WEB_PORT}）"
echo

stopped=0
stop_group "FastAPI (uvicorn)" "uvicorn server:app" && stopped=$((stopped + 1))
stop_group "Streamlit" "streamlit run streamlit_app.py" && stopped=$((stopped + 1))

echo
if [[ "$stopped" -eq 0 ]]; then
  warn "没有找到正在运行的服务进程。"
else
  ok "已停止 ${stopped} 组服务，端口 ${API_PORT} / ${WEB_PORT} 已释放。"
fi
