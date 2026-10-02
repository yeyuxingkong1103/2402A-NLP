#!/usr/bin/env bash
# =============================================================================
# 启动 SGLang 推理服务（vLLM 的备选方案）
# =============================================================================
# 工单给出的备选推理加速方案。SGLang 在结构化输出与高并发前缀复用上表现较好，
# 同样提供 OpenAI 兼容接口，因此 RAG 侧只需改 RAG_LLM__BASE_URL 即可切换。
#
# 安装（与主环境隔离）：
#     conda create -n sglang python=3.11 -y
#     conda activate sglang
#     pip install "sglang[all]"
#
# 用法：
#     bash scripts/run_sglang.sh
#     MODEL=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 bash scripts/run_sglang.sh
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
# 4090 24GB：量化权重约 5GB，mem-fraction-static 控制 KV Cache 占比
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=============================================="
echo " 启动 SGLang 服务"
echo "   模型        : ${MODEL}"
echo "   监听        : ${HOST}:${PORT}"
echo "   显存占比    : ${MEM_FRACTION}"
echo "   上下文长度  : ${CONTEXT_LENGTH}"
echo "   最大并发    : ${MAX_RUNNING_REQUESTS}"
echo "=============================================="
echo
echo "RAG 系统侧配置："
echo "   export RAG_LLM__BASE_URL=http://127.0.0.1:${PORT}/v1"
echo "   export RAG_LLM__MODEL=${MODEL##*/}"
echo

exec python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --served-model-name "$(basename "${MODEL}")" \
    --trust-remote-code
