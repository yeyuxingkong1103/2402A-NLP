#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 阶段：部署 / 算力云 Linux + GPU —— vLLM 推理服务（**首选**后端）
# =============================================================================
# 用途
#   在算力云（Linux + NVIDIA GPU）上起一个 OpenAI 兼容的 vLLM 服务，
#   供本仓库的 RAG 应用通过 RAG_LLM__BACKEND=openai + RAG_LLM__BASE_URL 调用。
#   本机 Windows **没有** vLLM（无 CUDA），本脚本为「已交付待云端验证」。
#
# 前置
#   pip install "vllm>=0.6.0"    # 云端安装；本机不可用且断网，无法安装
#   nvidia-smi 正常；模型权重已下载到 MODEL_PATH
#
# 用法
#   ./run_vllm.sh --help
#   MODEL_PATH=/models/Qwen2.5-7B-Instruct-AWQ ./run_vllm.sh
#   VLLM_TP_SIZE=2 VLLM_PORT=8000 ./run_vllm.sh --no-wait
#
# 全部参数均可用同名环境变量覆盖；命令行参数优先级更高。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${VLLM_MODEL_PATH:-/models/Qwen2.5-7B-Instruct-AWQ}"
SERVED_NAME="${VLLM_SERVED_NAME:-qwen2.5-7b-instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TP_SIZE:-1}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
DTYPE="${VLLM_DTYPE:-auto}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
API_KEY="${VLLM_API_KEY:-}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
WAIT_TIMEOUT="${VLLM_WAIT_TIMEOUT:-600}"
DO_WAIT=1

usage() {
  cat <<'EOF'
用法: run_vllm.sh [选项]

选项:
  -m, --model PATH       模型权重目录（默认 $VLLM_MODEL_PATH 或 /models/Qwen2.5-7B-Instruct-AWQ）
  -n, --name NAME        OpenAI 接口暴露的模型名（默认 $VLLM_SERVED_NAME 或 qwen2.5-7b-instruct）
      --host HOST        监听地址（默认 0.0.0.0）
  -p, --port PORT        监听端口（默认 8000）
      --tp N             tensor parallel 张量并行数（默认 1；多卡时设为 GPU 数）
      --gpu-util F       GPU 显存占用上限（默认 0.90）
      --max-len N        最大上下文长度（默认 8192）
      --max-num-seqs N   并发序列数（默认 64）
      --dtype D          数据类型 auto/float16/bfloat16（默认 auto）
      --quant Q          量化方式 awq/gptq/fp8（默认空=不指定）
      --api-key KEY      接口鉴权 key（默认空=不鉴权；设为随机串更安全）
      --extra "ARGS"     追加传给 vLLM 的参数（原样拼接）
      --no-wait          启动后不等待 /v1/models 就绪
      --wait-timeout N   就绪等待超时秒数（默认 600）
  -h, --help             显示本帮助

环境变量: VLLM_MODEL_PATH VLLM_SERVED_NAME VLLM_HOST VLLM_PORT VLLM_TP_SIZE
          VLLM_GPU_UTIL VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_DTYPE
          VLLM_QUANTIZATION VLLM_API_KEY VLLM_EXTRA_ARGS VLLM_WAIT_TIMEOUT
          PYTHON_BIN（默认 python3）

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
    --gpu-util) GPU_UTIL="$2"; shift 2 ;;
    --max-len) MAX_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
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

if ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
  fail "未检测到 vllm 包。云端请先安装：pip install 'vllm>=0.6.0'（本机 Windows 无 CUDA，不具备该依赖）"
fi
log "vllm        : $("$PYTHON_BIN" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' '; ')"
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  log "GPU 数量    : ${GPU_COUNT}（tp=${TP_SIZE}）"
  if [[ "$TP_SIZE" -gt "$GPU_COUNT" ]]; then
    fail "tp=${TP_SIZE} 超过可见 GPU 数 ${GPU_COUNT}"
  fi
else
  log "GPU         : 未检测到 nvidia-smi（vLLM 需要 NVIDIA GPU）"
fi

[[ -d "$MODEL_PATH" ]] || fail "模型目录不存在: ${MODEL_PATH}（示例：huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir \$MODEL_PATH）" 3
log "模型目录    : ${MODEL_PATH}"

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  fail "端口 ${PORT} 已被占用"
fi

# ---------------- 组装并启动 ----------------
ARGS=( -m vllm.entrypoints.openai.api_server
       --model "$MODEL_PATH"
       --served-model-name "$SERVED_NAME"
       --host "$HOST"
       --port "$PORT"
       --tensor-parallel-size "$TP_SIZE"
       --gpu-memory-utilization "$GPU_UTIL"
       --max-model-len "$MAX_LEN"
       --max-num-seqs "$MAX_NUM_SEQS"
       --dtype "$DTYPE" )
[[ -n "$QUANTIZATION" ]] && ARGS+=( --quantization "$QUANTIZATION" )
[[ -n "$API_KEY" ]] && ARGS+=( --api-key "$API_KEY" )
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && ARGS+=( $EXTRA_ARGS )

log "=== 启动 vLLM ==="
log "命令: ${PYTHON_BIN} ${ARGS[*]}"
log "对应应用侧环境变量："
log "  RAG_LLM__BACKEND=openai"
log "  RAG_LLM__BASE_URL=http://127.0.0.1:${PORT}/v1"
log "  RAG_LLM__MODEL=${SERVED_NAME}"

"$PYTHON_BIN" "${ARGS[@]}" &
VLLM_PID=$!
trap 'log "收到停止信号，终止 vLLM (pid=${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null || true' INT TERM

# ---------------- 就绪等待 ----------------
if [[ "$DO_WAIT" -eq 1 ]]; then
  log "=== 等待 /v1/models 就绪（超时 ${WAIT_TIMEOUT}s）==="
  DEADLINE=$(( $(date +%s) + WAIT_TIMEOUT ))
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      wait "$VLLM_PID" || true
      fail "vLLM 进程已退出（启动失败），请查看上方日志" 4
    fi
    if [[ "$(date +%s)" -ge "$DEADLINE" ]]; then
      log "就绪等待超时；进程仍在运行，可手动检查：curl http://127.0.0.1:${PORT}/v1/models"
      exit 4
    fi
    sleep 3
  done
  log "vLLM 已就绪："
  curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400
  echo
fi

wait "$VLLM_PID"
