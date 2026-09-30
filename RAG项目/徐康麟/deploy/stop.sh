#!/usr/bin/env bash
# ============================================================================
# deploy/stop.sh —— **结束脚本**：把服务停干净
#
# 停止顺序（**不能反**）：
#   [1/4] API   —— 它持有 Milvus Lite 的**单进程文件锁**、连着 Redis；
#                  先停它才能保证"锁被释放、写盘完成"。
#   [2/4] vLLM  —— 等它把显存释放干净（否则下次启动会 OOM）。
#   [3/4] Redis —— 最后停（API 收尾时还要写它）。
#   [4/4] 残留检查 + 运行时快照（GPU 显存应回到 0）。
#
# 为什么顺序不能反（真机踩过）：API 还在跑时另一个进程去开 Milvus Lite 会拿不到锁，
# 而 `MILVUS_FALLBACK_TO_MEMORY=True` 会**静默回落内存库** —— 于是后面的读数全是假的
# （"库是空的"但服务看起来正常）。
#
# 用法：
#   bash deploy/stop.sh                     # 停 API + vLLM + Redis
#   bash deploy/stop.sh --keep-redis        # 只停应用与模型，保留 Redis
#   bash deploy/stop.sh --keep-vllm         # 只停 API（例如要换 API 代码重起）
#   bash deploy/stop.sh --sweep             # 兜底：按进程名扫一遍（pid 文件丢了时用）
#   bash deploy/stop.sh --dry-run
#
# 退出码：0 停干净；1 参数错；40 停完仍有残留进程。
#
# 日志：logs/stop-YYYYmmdd.log（目录可用 LEGAL_RAG_LOG_DIR 覆盖）。
# ============================================================================
set -euo pipefail

LR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/lib.sh
source "$LR_LIB_DIR/lib.sh"
lr_init "stop"

KEEP_REDIS=0
KEEP_VLLM=0
SWEEP=0
GRACE=25

usage() {
  cat <<'USAGE'
结束脚本（Linux）：按"API → vLLM → Redis"的顺序停干净，并检查残留。

用法：bash deploy/stop.sh [选项]

选项：
  --keep-redis     保留 Redis（只停应用与 vLLM）
  --keep-vllm      保留 vLLM（只停 API）
  --sweep          pid 文件丢了时的兜底：按进程名扫一遍再停
  --grace SECONDS  SIGTERM 最长等待（默认 25s，之后 SIGKILL）
  --dry-run        只打印将执行的操作
  -h, --help       显示本帮助

退出码：0 停干净；1 参数错；40 停完仍有残留进程。
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-redis) KEEP_REDIS=1 ;;
    --keep-vllm) KEEP_VLLM=1 ;;
    --sweep) SWEEP=1 ;;
    --grace) shift; GRACE="${1:?--grace 需要值}" ;;
    --dry-run) LR_DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; lr_error "未知参数：$1"; exit 1 ;;
  esac
  shift
done
export LR_DRY_RUN

cd "$LR_ROOT"
RUN_DIR="$LR_ROOT/run"
API_PID_FILE="$RUN_DIR/api.pid"
VLLM_PID_FILE="$RUN_DIR/vllm.pid"

lr_section "停止服务（项目根：$LR_ROOT）"
lr_snapshot_runtime "停止前"
[[ "$LR_DRY_RUN" == "1" ]] && lr_warn "当前是 --dry-run：只打印，不动进程"

# ---------------------------------------------------------------------------
# [1/4] 停 API（先停它：它持锁、还连着 Redis）
# ---------------------------------------------------------------------------
lr_section "[1/4] 停止 API"
API_PID="$(lr_read_pid "$API_PID_FILE")"
if lr_pid_alive "$API_PID"; then
  lr_stop_pid "$API_PID" "API" "$GRACE"
  [[ "$LR_DRY_RUN" != "1" ]] && rm -f "$API_PID_FILE"
elif [[ "$SWEEP" == "1" ]]; then
  lr_warn "pid 文件里没有活进程，--sweep：按 'scripts/run_api.py' 扫一遍"
  # 只认**真 python** 进程：pgrep -f 会匹配到 bash -c 包装进程（真机教训，见 lib.sh）
  FOUND="$(lr_python_pids 'scripts/run_api\.py' || true)"
  if [[ -n "$FOUND" ]]; then
    for pid in $FOUND; do lr_stop_pid "$pid" "API(扫出)" "$GRACE"; done
  else
    lr_ok "没有扫到残留的 API 进程"
  fi
else
  lr_info "API 没在跑（也没有 pid 文件活进程）。若怀疑有残留，加 --sweep"
fi

# ---------------------------------------------------------------------------
# [2/4] 停 vLLM（等它释放显存）
# ---------------------------------------------------------------------------
lr_section "[2/4] 停止 vLLM"
if [[ "$KEEP_VLLM" == "1" ]]; then
  lr_info "按参数保留 vLLM（--keep-vllm）"
else
  VLLM_PID="$(lr_read_pid "$VLLM_PID_FILE")"
  if lr_pid_alive "$VLLM_PID"; then
    lr_stop_pid "$VLLM_PID" "vLLM" "$GRACE"
    [[ "$LR_DRY_RUN" != "1" ]] && rm -f "$VLLM_PID_FILE"
  elif [[ "$SWEEP" == "1" ]]; then
    FOUND="$(lr_python_pids 'vllm serve' || true)"
    if [[ -n "$FOUND" ]]; then
      for pid in $FOUND; do lr_stop_pid "$pid" "vLLM(扫出)" "$GRACE"; done
    else
      lr_ok "没有扫到残留的 vLLM 进程"
    fi
  else
    lr_info "vLLM 没在跑（没启动过，或已停）"
  fi
fi

# ---------------------------------------------------------------------------
# [3/4] 停 Redis（最后停）
# ---------------------------------------------------------------------------
lr_section "[3/4] 停止 Redis"
if [[ "$KEEP_REDIS" == "1" ]]; then
  lr_info "按参数保留 Redis（--keep-redis）"
elif ! lr_have redis-cli; then
  lr_info "没有 redis-cli，跳过（本机本来就没 Redis）"
elif ! redis-cli ping >/dev/null 2>&1; then
  lr_info "Redis 本来就没在跑"
else
  # `shutdown nosave`：**不落盘**。会话窗口是短期数据，落盘没有价值；
  # 真要保留就往 .env 里配 Redis 的持久化，而不是在这里 save。
  lr_run redis-cli shutdown nosave || lr_warn "redis-cli shutdown 失败，尝试 pkill"
  sleep 1
  if redis-cli ping >/dev/null 2>&1; then
    lr_warn "Redis 还在响应，退回 pkill"
    lr_run pkill -TERM -f redis-server || true
    sleep 2
  fi
  if redis-cli ping >/dev/null 2>&1; then
    lr_warn "Redis 仍未停（可能有 systemd 在拉起它：systemctl stop redis-server）"
  else
    lr_ok "Redis 已停"
  fi
fi

# ---------------------------------------------------------------------------
# [4/4] 残留检查
#   这里刻意**再查一遍**而不是复用上面的结果：停机最容易出的事就是
#   "以为停了、其实还有半截进程"（然后下次启动撞锁/撞端口）。
#
#   ⚠️ 2026-09-29 真机自检抓到的缺陷：`--dry-run` 下**不能**做这个检查 ——
#   干跑本来就**没有**停任何进程，检查必然报"残留" ⇒ 退出码 40。
#   （同一类缺陷在 install/deploy/start 里也各有一处，已一并修掉：
#    干跑只报告"会做什么/缺什么"，不拿它没动过的状态当失败。）
# ---------------------------------------------------------------------------
lr_section "[4/4] 残留检查"
LEFTOVER=0
check_left() {
  local label="$1" pattern="$2" pids
  pids="$(lr_python_pids "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    lr_error "$label 仍有残留进程：$(echo "$pids" | tr '\n' ' ')"
    LEFTOVER=1
  else
    lr_ok "$label 无残留"
  fi
}
if [[ "$LR_DRY_RUN" == "1" ]]; then
  lr_info "[dry-run] 跳过残留检查（干跑不会真的停进程，进程还在是预期的）"
else
  check_left "API" 'scripts/run_api\.py'
  [[ "$KEEP_VLLM" == "1" ]] || check_left "vLLM" 'vllm serve'
  if [[ "$KEEP_REDIS" != "1" ]] && lr_have redis-cli; then
    redis-cli ping >/dev/null 2>&1 && { lr_error "Redis 仍在运行"; LEFTOVER=1; } || lr_ok "Redis 无残留"
  fi
fi

lr_snapshot_runtime "停止后"
if [[ "$LEFTOVER" == "1" ]]; then
  lr_error "停完仍有残留（见上）。可加 --sweep 重跑，或人工确认这些进程是谁的。"
  exit 40
fi
if [[ "$LR_DRY_RUN" == "1" ]]; then
  lr_section "干跑结束（未停任何进程）"
else
  lr_section "已全部停止"
fi
lr_info "  API 日志： $LR_ROOT/logs/api.log"
lr_info "  脚本日志：$LR_LOG_FILE"
lr_info "  重新启动：bash deploy/start.sh"
