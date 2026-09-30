#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
stop_project_api() {
  local pid="$1"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]] \
    && { cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"; [[ -z "$cwd" || "$cwd" == "$ROOT" ]]; } \
    && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Eq 'uvicorn app\.(main|single_app):app([[:space:]]|$)' \
    && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq -- '--port 8000'; then
    kill "$pid" 2>/dev/null || sudo kill "$pid" 2>/dev/null || true
  fi
}
if [[ -f logs/local-api.pid ]]; then
  stop_project_api "$(cat logs/local-api.pid)"
  rm -f logs/local-api.pid
fi
while read -r pid; do
  [[ -n "$pid" ]] || continue
  stop_project_api "$pid"
done < <({
  ss -ltnp 2>/dev/null | awk '$4 ~ /(^|:)8000$/ {print}' | grep -o 'pid=[0-9]*' | cut -d= -f2 || true
  ps -eo pid=,args= 2>/dev/null | awk '$0 ~ /[u]vicorn app\.(main|single_app):app([[:space:]]|$)/ && $0 ~ /--port 8000/ {print $1}'
} | sort -u)
if [[ -f logs/local-llm.pid ]]; then
  pid="$(cat logs/local-llm.pid)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]] \
    && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q 'scripts.local_llm_server'; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f logs/local-llm.pid
fi
docker_cmd=(docker)
docker info >/dev/null 2>&1 || docker_cmd=(sudo docker)
"${docker_cmd[@]}" compose --env-file .env.local -f deploy/docker-compose.yml stop
echo "STOPPED: API、可选本地模型和数据容器已停止，Docker 命名卷中的数据仍保留。"
