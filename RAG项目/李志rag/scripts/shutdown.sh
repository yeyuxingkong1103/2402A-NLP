#!/usr/bin/env bash
set -euo pipefail
for file in .api.pid .ui.pid; do
  if [ -f "$file" ]; then kill "$(cat "$file")" 2>/dev/null || true; rm -f "$file"; fi
done
# 只停止容器，不删除容器；Docker Desktop 中会继续显示 medical-rag。
docker compose stop
