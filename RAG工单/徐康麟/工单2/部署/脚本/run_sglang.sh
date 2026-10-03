#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 阶段：部署 / 算力云 Linux + GPU —— SGLang 推理服务（**备选**后端）
# =============================================================================
# 用途
#   与 run_vllm.sh 等价，但使用 SGLang 的 OpenAI 兼容服务（默认端口 30000）。
#   当 vLLM 在目标卡型/模型上不可用（如 AWQ 支持差异、显存不足）时作为备选。
#   SGLang 的 OpenAI 兼容端点为 http://<host>:<port>/v1。
#   本机 Windows **没有** SGLang，本脚本为「已交付待云端验证」。
#
# 前置
#   pip install "sglang[all]>=0.3.0"
#   nvidia-smi 正常；模型权重已下载到 MODEL_PATH
#
# 用法
#   ./run_sglang.sh --help
#   MODEL_PATH=/models/Qwen2.5-7B-Instruct ./run_sglang.sh
#   SGLANG_TP_SIZE=2 SGLANG_PORT=30000 ./run_sglang.sh --no-wait
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${SGLANG_MODEL_PATH:-/models/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SGLANG_SERVED_NAME:-qwen2.5-7b-instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LEN="${SGLANG_CONTEXT_LENGTH:-8192}"
MAX_RUNNING="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
DTYPE="${SGLANG_DTYPE:-auto}"
QUANTIZATION="${SGLANG_QUANTIZATION:-}"
API_KEY="${SGLANG_API_KEY:-}"
EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
WAIT_TIMEOUT="${SGLANG_WAIT_TIMEOUT:-600}"
DO_WAIT=1

usage() {
  cat <<'EOF'
用法: run_sglang.sh [选项]

选项:
  -m, --model PATH        模型权重目录（默认 $SGLANG_MODEL_PATH 或 /models/Qwen2.5-7B-Instruct）
  -n, --name NAME         OpenAI 接口暴露的模型名（默认 $SGLANG_SERVED_NAME）
      --host HOST         监听地址（默认 0.0.0.0）
  -p, --port PORT         监听端口（默认 30000，SGLang 惯例）
      --tp N             tensor parallel 张量并行数（默认 1）
      --mem-fraction F   静态显存占比（默认 0.85）
      --context-len N    最大上下文长度（默认 8192）
      --max-running N    最大并发请求数（默认 64）
      --dtype D          auto/float16/bfloat16（默认 auto）
      --quant Q          awq/gptq/fp8 等（默认空=不指定）
      --api-key KEY      接口鉴权 key（默认空=不鉴权）
      --extra "ARGS"     追加传给 SGLang 的参数
      --no-wait          启动后不等待 /v1/models 就绪
      --wait-timeout N   就绪等待超时秒数（默认 600）
  -h, --help             显示本帮助

环境变量: SGLANG_MODEL_PATH SGLANG_SERVED_NAME SGLANG_HOST SGLANG_PORT SGLANG_TP_SIZE
          SGLANG_MEM_FRACTION_STATIC SGLANG_CONTEXT_LENGTH SGLANG_MAX_RUNNING_REQUESTS
          SGLANG_DTYPE SGLANG_QUANTIZATION SGLANG_API_KEY SGLANG_EXTRA_ARGS
          SGLANG_WAIT_TIMEOUT PYTHON_BIN（默认 python3）

退出码: 0 正常退出 | 2 前置检查失败 | 3 模型目录不存在 | 4 就绪等待超时
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL_PATH="$2"; shift 2 ;;
    -n|--name) SERVED_NAME="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    -p|--port) PORT="$2"; shift 2 ;;
    --tp) TP_SIZE="$2"; shift 2 ;;
    --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
    --context-len) CONTEXT_LEN="$2"; shift 2 ;;
    --max-running) MAX_RUNNING="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --quant) QUANTIZATION="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    --no-wait) DO_WAIT=0; shift ;;
    --wait-timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[错误] 未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
log()  { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[失败] $*" >&2; exit "${2:-2}"; }

# ---------------- 前置检查 ----------------
log "=== 前置检查 ==="
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "找不到 $PYTHON_BIN（可用 PYTHON_BIN 覆盖）"
log "python      : $(command -v "$PYTHON_BIN") $("$PYTHON_BIN" -V 2>&1)"

if ! "$PYTHON_BIN" -c "import sglang" >/dev/null 2>&1; then
  fail "未检测到 sglang 包。云端请先安装：pip install 'sglang[all]>=0.3.0'（本机不具备该依赖）"
fi
log "sglang      : $("$PYTHON_BIN" -c 'import sglang; print(getattr(sglang, "__version__", "unknown"))' 2>/dev/null || echo unknown)"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' '; ')"
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ "$TP_SIZE" -gt "$GPU_COUNT" ]]; then
    fail "tp=${TP_SIZE} 超过可见 GPU 数 ${GPU_COUNT}"
  fi
else
  log "GPU         : 未检测到 nvidia-smi（SGLang 需要 NVIDIA GPU）"
fi

[[ -d "$MODEL_PATH" ]] || fail "模型目录不存在: ${MODEL_PATH}" 3
log "模型目录    : ${MODEL_PATH}"

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  fail "端口 ${PORT} 已被占用"
fi

# ---------------- 组装并启动 ----------------
ARGS=( -m sglang.launch_server
       --model-path "$MODEL_PATH"
       --served-model-name "$SERVED_NAME"
       --host "$HOST"
       --port "$PORT"
       --tp-size "$TP_SIZE"
       --mem-fraction-static "$MEM_FRACTION"
       --context-length "$CONTEXT_LEN"
       --max-running-requests "$MAX_RUNNING"
       --dtype "$DTYPE" )
[[ -n "$QUANTIZATION" ]] && ARGS+=( --quantization "$QUANTIZATION" )
[[ -n "$API_KEY" ]] && ARGS+=( --api-key "$API_KEY" )
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && ARGS+=( $EXTRA_ARGS )

log "=== 启动 SGLang ==="
log "命令: ${PYTHON_BIN} ${ARGS[*]}"
log "对应应用侧环境变量："
log "  RAG_LLM__BACKEND=openai"
log "  RAG_LLM__BASE_URL=http://127.0.0.1:${PORT}/v1"
log "  RAG_LLM__MODEL=${SERVED_NAME}"

"$PYTHON_BIN" "${ARGS[@]}" &
SGLANG_PID=$!
trap 'log "收到停止信号，终止 SGLang (pid=${SGLANG_PID})"; kill "${SGLANG_PID}" 2>/dev/null || true' INT TERM

if [[ "$DO_WAIT" -eq 1 ]]; then
  log "=== 等待 /v1/models 就绪（超时 ${WAIT_TIMEOUT}s）==="
  DEADLINE=$(( $(date +%s) + WAIT_TIMEOUT ))
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
      wait "$SGLANG_PID" || true
      fail "SGLang 进程已退出（启动失败），请查看上方日志" 4
    fi
    if [[ "$(date +%s)" -ge "$DEADLINE" ]]; then
      log "就绪等待超时；可手动检查：curl http://127.0.0.1:${PORT}/v1/models"
      exit 4
    fi
    sleep 3
  done
  log "SGLang 已就绪："
  curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400
  echo
fi

wait "$SGLANG_PID"
