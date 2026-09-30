#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_FILE="deploy/autodl/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy deploy/autodl/env.example to $ENV_FILE and edit it first."
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

mkdir -p logs

nohup python -m uvicorn app.main:app \
  --host "$RAG_HOST" \
  --port "$RAG_PORT" \
  > logs/rag-api.log 2>&1 &

echo $! > logs/rag-api.pid
echo "RAG API started, pid=$(cat logs/rag-api.pid), log=logs/rag-api.log"
