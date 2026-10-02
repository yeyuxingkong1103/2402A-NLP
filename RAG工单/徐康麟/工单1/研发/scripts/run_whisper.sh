#!/usr/bin/env bash
# =============================================================================
# 启动 Whisper 语音识别服务（OpenAI 兼容接口）
# =============================================================================
# 工单追加要求：支持语音输入。
#
# 为什么单独起一个服务而不是在应用里加载模型：
#   1. 4090 的显存要优先留给 LLM；Whisper 单独进程便于按需启停、限制显存；
#   2. vLLM 提供 OpenAI 兼容的 /v1/audio/transcriptions，
#      应用侧用同一套 OpenAI 客户端即可，无需新增依赖。
#
# 用法：
#     bash scripts/run_whisper.sh                     # 默认 whisper-large-v3
#     MODEL=openai/whisper-medium PORT=8001 bash scripts/run_whisper.sh
#
# 启动后自检：
#     curl http://127.0.0.1:8001/v1/models
#
# 应用侧配置（无需改代码）：
#     export RAG_ASR__BASE_URL=http://127.0.0.1:8001/v1
#     export RAG_ASR__MODEL=openai/whisper-large-v3
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
# Whisper-large-v3 约 3GB 权重；4090 上给 0.25 显存即可（约 6GB）
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
# 单次请求最长音频秒数
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-120}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=============================================="
echo " 启动 Whisper 语音识别服务"
echo "   模型        : ${MODEL}"
echo "   监听        : ${HOST}:${PORT}"
echo "   显存利用率  : ${GPU_MEMORY_UTILIZATION}"
echo "=============================================="
echo
echo "应用侧配置："
echo "   export RAG_ASR__BASE_URL=http://127.0.0.1:${PORT}/v1"
echo "   export RAG_ASR__MODEL=${MODEL}"
echo

# 优先用 vLLM（与 LLM 一致的部署方式）；没有则回退 faster-whisper 服务
if python -c "import vllm" 2>/dev/null; then
    exec python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --served-model-name "${MODEL}" \
        --trust-remote-code \
        --disable-log-requests
else
    echo "[提示] 未检测到 vLLM，尝试 faster-whisper-server（需 pip install faster-whisper-server）"
    exec faster-whisper-server \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}"
fi
