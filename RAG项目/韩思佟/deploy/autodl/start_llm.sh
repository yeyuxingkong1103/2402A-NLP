#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ -f deploy/autodl/.env ]] && { set -a; source deploy/autodl/.env; set +a; }

PYTHON="${DEMO_VLLM_PYTHON:-/root/miniconda3/envs/vllm/bin/python}"
MODEL="${QWEN_MODEL_PATH:-/root/autodl-tmp/Qwen3.8-27b}"
NAME="${QWEN_SERVED_MODEL_NAME:-Qwen3.8-27b}"
PORT="${VLLM_PORT:-8001}"
mkdir -p logs

ready() {
  curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(not any(x.get("id")==sys.argv[1] for x in d.get("data",[])))' "$NAME"
}
if ready; then
  echo "MODEL_READY: $NAME 已经运行在本机端口 $PORT"
  exit 0
fi
[[ -x "$PYTHON" ]] || { echo "找不到 vLLM Python: $PYTHON" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "找不到模型目录: $MODEL" >&2; exit 1; }

help="$($PYTHON -m vllm.entrypoints.openai.api_server --help=all 2>logs/vllm-help.log)"
mm=()
if [[ "$help" == *'--language-model-only'* ]]; then
  mm=(--language-model-only)
elif [[ "$help" == *'--limit-mm-per-prompt'* ]]; then
  mm=(--limit-mm-per-prompt '{"image":0,"video":0}')
fi

VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port "$PORT" --model "$MODEL" --served-model-name "$NAME" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.88}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" --max-num-seqs 1 --enforce-eager \
  "${mm[@]}" >logs/vllm.log 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >logs/vllm.pid

for second in $(seq 5 5 600); do
  if ready; then
    echo "MODEL_READY: $NAME 已启动。现在在本地 Ubuntu 运行 deploy/local/tunnel.sh"
    exit 0
  fi
  kill -0 "$pid" 2>/dev/null || { tail -n 80 logs/vllm.log >&2; exit 1; }
  echo "等待 Qwen 加载：${second}s"
  sleep 5
done
tail -n 80 logs/vllm.log >&2
echo "模型 600 秒内未就绪；进程保留，检查 logs/vllm.log" >&2
exit 1
