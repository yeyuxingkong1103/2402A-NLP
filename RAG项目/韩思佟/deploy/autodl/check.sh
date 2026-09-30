#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_FILE="deploy/autodl/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

VLLM_PORT="${VLLM_PORT:-8001}"
RAG_PORT="${RAG_PORT:-8000}"
LLM_MODEL="${LLM_MODEL:-qwen3-27b}"

echo "Checking vLLM models endpoint..."
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models"
echo

echo "Checking RAG root endpoint..."
curl -s "http://127.0.0.1:${RAG_PORT}/"
echo

echo "Checking chat completions endpoint..."
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，用一句话介绍你自己\"}],\"max_tokens\":64,\"temperature\":0.2}"
echo
