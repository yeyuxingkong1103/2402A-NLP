#!/usr/bin/env bash
# Start the web demo with existing base/vllm conda environments; install nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f deploy/autodl/.env ]]; then
  set -a
  source <(sed 's/\r$//' deploy/autodl/.env)
  set +a
fi

API_PYTHON="${DEMO_API_PYTHON:-/root/miniconda3/bin/python}"
VLLM_PYTHON="${DEMO_VLLM_PYTHON:-/root/miniconda3/envs/vllm/bin/python}"
MODEL_PATH="${QWEN_MODEL_PATH:-/root/autodl-tmp/Qwen3.8-27b}"
MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-Qwen3.8-27b}"
API_PORT=6006
LLM_PORT=8001

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ "${1:-}" == '' || "${1:-}" == '--check' ]] || fail "Usage: bash deploy/autodl/start_demo.sh [--check]"
[[ -x "$API_PYTHON" ]] || fail "Missing base Python: $API_PYTHON"
[[ -x "$VLLM_PYTHON" ]] || fail "Missing vllm Python: $VLLM_PYTHON"
echo "Checking files and existing environments; no installs or downloads."
"$API_PYTHON" deploy/autodl/preflight.py --model-path "$MODEL_PATH" \
  --api-python "$API_PYTHON" --vllm-python "$VLLM_PYTHON"
if [[ "${1:-}" == '--check' ]]; then
  echo "无卡预检通过。已完成文件和依赖检查；真实问答将在带卡启动时验证。"
  exit 0
fi

mkdir -p logs
command -v curl >/dev/null || fail "curl is required."
command -v flock >/dev/null || fail "flock is required."
exec 9>logs/demo-start.lock
flock -n 9 || fail "Another start_demo.sh is running; follow its output."

port_open() {
  "$API_PYTHON" - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

ready() {
  if [[ "$1" == api ]]; then
    curl -fsS --max-time 3 "http://127.0.0.1:$API_PORT/" |
      "$API_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("status")=="ok" and d.get("service")=="RAG 角色扮演系统" else 1)' 2>/dev/null &&
      curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:$API_PORT/chat" &&
      curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:$API_PORT/static/app.js" &&
      curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:$API_PORT/static/app.css" &&
      curl -fsS --max-time 3 "http://127.0.0.1:$API_PORT/api/roles" |
      "$API_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if len(d)==1 and d[0].get("name")=="医生" else 1)' 2>/dev/null
  else
    curl -fsS --max-time 3 "http://127.0.0.1:$LLM_PORT/v1/models" |
      "$API_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(m.get("id")==sys.argv[1] for m in d.get("data",[])) else 1)' "$MODEL_NAME" 2>/dev/null
  fi
}

owned_pid() {
  # A stale PID file alone is not proof: verify command and project directory.
  "$API_PYTHON" - "$1" "$2" "$PWD" <<'PY'
from pathlib import Path
import sys
try:
    pid = Path(sys.argv[1]).read_text().strip()
    if not pid.isdecimal():
        raise ValueError("invalid pid")
    proc = Path("/proc") / pid
    args = proc.joinpath("cmdline").read_bytes().split(b"\0")
    if sys.argv[2].encode() not in args or proc.joinpath("cwd").resolve() != Path(sys.argv[3]):
        raise ValueError("unrelated process")
    print(pid)
except (OSError, ValueError):
    sys.exit(1)
PY
}

wait_ready() {
  local kind="$1" pid="$2" log="$3" limit="$4" started=$SECONDS
  while ! ready "$kind" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 60 "$log" >&2
      fail "$kind process exited; see $log."
    fi
    if (( SECONDS - started >= limit )); then
      tail -n 60 "$log" >&2
      fail "$kind not ready after ${limit}s. Process left running; inspect $log before retrying."
    fi
    echo "Waiting for $kind... $((SECONDS - started))s (log: $log)"
    sleep 5
  done
}

echo "[1/4] Checking runtime and GPU."
CUDA_VISIBLE_DEVICES='' "$API_PYTHON" -c 'import fastapi, openai, pymilvus, sentence_transformers; print("RAG imports OK")'

llm_pid=''
if ready llm >/dev/null 2>&1; then
  echo "Existing vLLM model is ready on port $LLM_PORT; reusing it."
elif llm_pid=$(owned_pid logs/demo-vllm.pid vllm.entrypoints.openai.api_server); then
  echo "Existing demo vLLM process $llm_pid is still starting; will wait."
else
  port_open "$LLM_PORT" && fail "Port $LLM_PORT is occupied by another service; no process was stopped."
  "$API_PYTHON" deploy/autodl/preflight.py --model-path "$MODEL_PATH" \
    --api-python "$API_PYTHON" --vllm-python "$VLLM_PYTHON" --gpu
fi

echo "[2/4] Starting web/API on port $API_PORT using base Python and CPU embeddings."
# Apply the freshly uploaded code. Only stop a process owned by this launcher.
if api_pid=$(owned_pid logs/demo-api.pid app.main:app); then
  echo "Restarting verified demo API PID $api_pid to load the uploaded update."
  kill "$api_pid"
  stop_started=$SECONDS
  while owned_pid logs/demo-api.pid app.main:app >/dev/null 2>&1; do
    (( SECONDS - stop_started < 30 )) || fail "API is still stopping; retry after the current request finishes."
    sleep 1
  done
  port_stop_started=$SECONDS
  while port_open "$API_PORT"; do
    (( SECONDS - port_stop_started < 30 )) || fail "API port $API_PORT is still occupied after the owned process stopped."
    sleep 1
  done
fi
port_open "$API_PORT" && fail "Port $API_PORT is occupied by another service; no unrelated process was stopped."
CUDA_VISIBLE_DEVICES='' RAG_EMBEDDING_DEVICE=cpu \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LLM_BASE_URL="http://127.0.0.1:$LLM_PORT/v1" LLM_API_KEY=EMPTY \
  LLM_MODEL="$MODEL_NAME" LLM_ENABLE_THINKING=false \
  nohup "$API_PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" \
  >logs/demo-api.log 2>&1 </dev/null 9>&- &
api_pid=$!
printf '%s\n' "$api_pid" >logs/demo-api.pid
wait_ready api "$api_pid" logs/demo-api.log 90

echo "[3/4] Starting Qwen from local model files; first load can take several minutes."
if ! ready llm >/dev/null 2>&1; then
  if [[ -z "$llm_pid" ]]; then
    # Discover flags from the installed vLLM, rather than assuming a release's CLI.
    # vLLM's ordinary --help shows group names only; --help=all exposes flags.
    vllm_help=$("$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server --help=all 2>logs/demo-vllm-help.log) || fail "Cannot read vLLM options; see logs/demo-vllm-help.log."
    mm_args=()
    if [[ "$vllm_help" == *'--language-model-only'* ]]; then
      mm_args=(--language-model-only)
    elif [[ "$vllm_help" == *'--limit-mm-per-prompt'* ]]; then
      mm_args=(--limit-mm-per-prompt '{"image":0,"video":0}')
    else
      echo "Notice: this vLLM has no text-only multimodal flag; continuing with its default model loader."
    fi
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      nohup "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port "$LLM_PORT" \
      --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
      --max-model-len 8192 --max-num-seqs 1 --enforce-eager "${mm_args[@]}" \
      >logs/demo-vllm.log 2>&1 </dev/null 9>&- &
    llm_pid=$!
    printf '%s\n' "$llm_pid" >logs/demo-vllm.pid
  fi
  wait_ready llm "$llm_pid" logs/demo-vllm.log 600
fi

echo "[4/4] Testing actual doctor chat, citations, history and web assets."
"$API_PYTHON" scripts/check_demo.py --base-url "http://127.0.0.1:$API_PORT" || {
  tail -n 60 logs/demo-api.log >&2
  fail "真实问答验收未通过。请发送上面的错误；日志：logs/demo-api.log 和 logs/demo-vllm.log。"
}
echo "READY: 医生网页、知识库检索和真实模型问答验收通过。"
echo "Open AutoDL Custom Service -> HTTP port 6006, then append /chat to its URL."
echo "Logs: logs/demo-api.log and logs/demo-vllm.log"
echo "GPU billing continues while the instance is on; shut it down in AutoDL when finished."
