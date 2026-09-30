#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${RAG_VENV:-/home/lenovo/.venvs/rag-roleplay}"
cd "$ROOT"
MODE="${1:-}"
[[ -f .env.local ]] || cp .env.local.example .env.local
[[ -x "$VENV/bin/python" ]] || { echo "请先运行 bash deploy/local/setup.sh" >&2; exit 1; }

project_env=()
while IFS= read -r env_line; do
  env_line="${env_line%$'\r'}"
  [[ "$env_line" =~ ^(RAG_|LLM_|LOCAL_LLM_)[A-Za-z0-9_]*= ]] || continue
  project_env+=("$env_line")
done < .env.local
project_env+=("RAG_ENV_FILE=$ROOT/.env.local")
run_python() {
  env "${project_env[@]}" "$VENV/bin/python" "$@"
}

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  echo "需要 sudo 访问 Docker；请输入 Ubuntu 密码。"
  sudo -v
  docker_cmd=(sudo docker)
fi
compose=("${docker_cmd[@]}" compose --env-file .env.local -f deploy/docker-compose.yml)

service_args=(mysql redis etcd minio milvus)
if [[ "$MODE" == "--visual" ]]; then
  "${compose[@]}" --profile visual up -d --wait "${service_args[@]}" redisinsight attu
  echo "VISUAL_READY: RedisInsight http://127.0.0.1:5540 (redis:6379, DB0)"
  echo "VISUAL_READY: Attu http://127.0.0.1:3000 (milvus:19530)"
  exit 0
fi

if [[ "$MODE" != "--check" ]]; then
  "${compose[@]}" up -d --wait "${service_args[@]}"
fi
llm_provider="$(sed -n 's/^LLM_PROVIDER=//p' .env.local | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
llm_provider="${llm_provider:-local_hf}"
if [[ "$MODE" != "--check" && "$MODE" != "--services-only" \
      && ( "$llm_provider" == "local_hf" || "$llm_provider" == "local" ) ]]; then
  model_path="$(sed -n 's/^LOCAL_LLM_MODEL_PATH=//p' .env.local | tail -n 1 | tr -d '\r')"
  model_path="${model_path:-/mnt/d/AI_Friend/models/Qwen/Qwen3-0.6B}"
  [[ -f "$model_path/config.json" ]] || {
    echo "找不到本地模型：$model_path" >&2
    exit 1
  }
  mkdir -p logs
  llm_cmdline() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    [[ -z "$cwd" || "$cwd" == "$ROOT" ]] || return 1
    tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq 'scripts.local_llm_server'
  }
  if [[ -f logs/local-llm.pid ]] && llm_cmdline "$(cat logs/local-llm.pid)"; then
    echo "本地 Qwen 服务已运行，PID $(cat logs/local-llm.pid)"
  else
    rm -f logs/local-llm.pid
    nohup env "${project_env[@]}" \
      "$VENV/bin/python" -m scripts.local_llm_server >logs/local-llm.log 2>&1 </dev/null &
    llm_pid=$!
    printf '%s\n' "$llm_pid" >logs/local-llm.pid
  fi
  for attempt in {1..60}; do
    curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1 && break
    sleep 1
  done
  expected_model="$(sed -n 's/^LLM_MODEL=//p' .env.local | tail -n 1 | tr -d '\r')"
  expected_model="${expected_model:-Qwen3-0.6B}"
  models_json="$(curl -fsS http://127.0.0.1:8001/v1/models)" || {
    tail -n 80 logs/local-llm.log
    exit 1
  }
  grep -Fq '"id":"'"$expected_model"'"' <<<"$models_json" || {
    echo "8001 返回的模型不是 $expected_model：$models_json" >&2
    exit 1
  }
  echo "LOCAL_LLM_READY: 已使用本地 Qwen 权重，无需 Ollama 镜像。"
elif [[ "$MODE" != "--check" && "$MODE" != "--services-only" ]]; then
  if [[ -f logs/local-llm.pid ]]; then
    old_llm_pid="$(cat logs/local-llm.pid)"
    if [[ "$old_llm_pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$old_llm_pid/cmdline" ]] \
      && tr '\0' ' ' <"/proc/$old_llm_pid/cmdline" | grep -Fq 'scripts.local_llm_server'; then
      kill "$old_llm_pid" 2>/dev/null || true
    fi
    rm -f logs/local-llm.pid
  fi
  echo "ONLINE_LLM: 使用 $llm_provider 在线模型，不启动本地 Qwen。"
fi
if [[ "$MODE" != "--check" ]]; then
  # 只有缺库或空库才导入快照；已有库保持稳定主键，不在启动时覆盖。
  run_python - <<'PY'
from app.offline_pipeline import COLLECTION, load_env, sync_milvus

load_env()
import os
import time
from pymilvus import MilvusClient

client = MilvusClient(uri=os.environ.get("RAG_MILVUS_URI", "http://127.0.0.1:19530"))
try:
    populated = False
    for attempt in range(30):
        try:
            populated = client.has_collection(COLLECTION) and bool(
                client.query(COLLECTION, filter="id >= 0", output_fields=["id"], limit=1)
            )
            break
        except Exception:
            if attempt == 29:
                raise
            time.sleep(1)
finally:
    client.close()
if populated:
    print("KNOWLEDGE_KEEP: 保留已有知识库；更新请使用 app.offline_pipeline sync")
else:
    report = sync_milvus()
    print(f"KNOWLEDGE_INITIALIZED: {report['snapshot']['records']} 个知识块")
PY
fi
if [[ "$MODE" == "--check" || "$MODE" == "--services-only" ]]; then
  run_python -m scripts.check_services
  exit 0
fi
run_python -m scripts.check_services --require-llm

mkdir -p logs
api_cmdline() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ -z "$cwd" || "$cwd" == "$ROOT" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Eq 'uvicorn app\.(main|single_app):app([[:space:]]|$)' \
    && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq -- '--port 8000'
}

api_port_pids() {
  ss -ltnp 2>/dev/null \
    | awk '$4 ~ /(^|:)8000$/ {print}' \
    | grep -o 'pid=[0-9]*' \
    | cut -d= -f2 \
    | sort -u || true
}

api_process_pids() {
  {
    api_port_pids
    ps -eo pid=,args= 2>/dev/null \
      | awk '$0 ~ /[u]vicorn app\.(main|single_app):app([[:space:]]|$)/ && $0 ~ /--port 8000/ {print $1}'
  } | sort -u
}

stop_api_pid() {
  local pid="$1"
  if api_cmdline "$pid"; then
    kill "$pid" 2>/dev/null || sudo kill "$pid" 2>/dev/null || true
    for wait in {1..10}; do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" 2>/dev/null || true
  fi
}

if [[ -f logs/local-api.pid ]]; then
  old_pid="$(cat logs/local-api.pid)"
  if api_cmdline "$old_pid"; then
    stop_api_pid "$old_pid"
  fi
  rm -f logs/local-api.pid
fi
while read -r port_pid; do
  [[ -n "$port_pid" ]] || continue
  if api_cmdline "$port_pid"; then
    stop_api_pid "$port_pid"
  else
    echo "端口 8000 已被非本项目进程占用（PID $port_pid），请先检查后再启动。" >&2
    exit 1
  fi
done < <(api_process_pids)

sleep 1
if curl -fsS --max-time 1 http://127.0.0.1:8000/ >/dev/null 2>&1; then
  echo "端口 8000 仍被占用，无法确认归属；请先停止旧的项目 API。" >&2
  exit 1
fi

nohup env "${project_env[@]}" "$VENV/bin/python" -m uvicorn app.single_app:app --host 127.0.0.1 --port 8000 \
  >logs/local-api.log 2>&1 </dev/null &
api_pid=$!
printf '%s\n' "$api_pid" >logs/local-api.pid
api_ready=false
for attempt in {1..90}; do
  if ! kill -0 "$api_pid" 2>/dev/null; then
    break
  fi
  if api_cmdline "$api_pid" && curl -fsS http://127.0.0.1:8000/ >/dev/null 2>&1; then
    api_ready=true
    break
  fi
  sleep 1
done
if [[ "$api_ready" != true ]]; then
  echo "API 启动失败，未确认新进程 PID $api_pid 正在监听 8000。" >&2
  stop_api_pid "$api_pid"
  rm -f logs/local-api.pid
  tail -n 60 logs/local-api.log
  exit 1
fi
echo "LOCAL_READY: 打开 http://127.0.0.1:8000/chat"
echo "模型服务、MySQL、Redis、Milvus、API 和网页均已就绪。"
