#!/usr/bin/env bash
# =============================================================================
# 启动 vLLM 推理服务（OpenAI 兼容接口）
# =============================================================================
# 工单要求：6B 左右小模型 + vLLM 加速，部署在算力云 4090 机器上。
#
# 重要：vLLM 不要装进 gao6gongdan 主环境，单独建环境避免 CUDA/torch 冲突：
#     conda create -n vllm python=3.11 -y
#     conda activate vllm
#     pip install vllm
#
# 用法：
#     bash scripts/run_vllm.sh                     # 默认 Qwen2.5-7B-Instruct-AWQ
#     MODEL=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 bash scripts/run_vllm.sh
#     PORT=8001 bash scripts/run_vllm.sh
#
# 启动后自检：
#     curl http://127.0.0.1:8000/v1/models
# =============================================================================
set -euo pipefail

# ------------------------------ 可调参数 ------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
# 4090 是 24GB 显存：7B 的 AWQ/GPTQ-Int4 权重约 5GB，
# 剩余显存用于 KV Cache，显存利用率给 0.90 比较稳。
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# 中等并发：4090 单卡建议 16~32，配合 --max-num-seqs
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
DTYPE="${DTYPE:-float16}"
# 国内下载模型较慢时可指定镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=============================================="
echo " 启动 vLLM 服务"
echo "   模型        : ${MODEL}"
echo "   监听        : ${HOST}:${PORT}"
echo "   显存利用率  : ${GPU_MEMORY_UTILIZATION}"
echo "   最大长度    : ${MAX_MODEL_LEN}"
echo "   最大并发序列: ${MAX_NUM_SEQS}"
echo "   HF_ENDPOINT : ${HF_ENDPOINT}"
echo "=============================================="
echo
echo "启动成功后，RAG 系统侧配置（无需改代码，用环境变量覆盖即可）："
echo "   export RAG_LLM__BASE_URL=http://127.0.0.1:${PORT}/v1"
echo "   export RAG_LLM__MODEL=${MODEL##*/}"
echo

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --served-model-name "$(basename "${MODEL}")" \
    --trust-remote-code \
    --disable-log-requests
