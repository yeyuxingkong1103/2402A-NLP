# 工单编号：人工智能NLP-RAG-金融问答系统部署
#
# 金融问答系统镜像。
#
# 基础镜像用 python:3.11-slim，而不是 nvidia/cuda 或 pytorch/pytorch：
#   · Python 3.11 与开发环境（conda rag_gd，3.11.16）一致
#   · Linux 上 PyPI 的 torch 轮子自带 CUDA 12 运行时（依赖里那串 nvidia-*-cu12
#     包就是它），所以**不需要** nvidia/cuda 基础镜像，容器照样能用 GPU ——
#     宿主机的驱动由 NVIDIA Container Toolkit 在 `--gpus all` 时注入容器
#   · nvidia/cuda、pytorch/pytorch 这类镜像单个就 2~4GB，本机 registry 加速器
#     拉它们反复 TLS 超时；换成 slim 基础镜像后，大头改从 pip 源（国内可达）下载
#
# 模型和知识库**不打进镜像**：
#   · bge-m3 有 2.1GB（排除 onnx/ 后），打进去镜像就没法快速重建/分发；
#   · 知识库是「关键数据」，按工单验收要求要放 Docker 卷持久化，不能固化在镜像层。
#   两者都在运行时挂进来，见 docker-compose.yml 与《部署文档》。
#
# 构建（在工单10 目录下执行，构建上下文是工单根目录）：
#   docker build -f 部署/Dockerfile -t rag-finance:1.0 .

FROM python:3.11-slim

# PYTHONUNBUFFERED 日志实时刷出，否则 docker logs 要等缓冲满才看得到
# PIP_NO_CACHE_DIR 不留 pip 缓存，镜像小几百 MB
# TRANSFORMERS_VERBOSITY / TOKENIZERS_PARALLELISM
#                  压掉 transformers 的告警。验收标准要求「容器运行过程中无异常
#                  日志输出」，而 transformers 默认会打一条 XLMRobertaTokenizerFast
#                  的提示，看着像异常，实际无害，直接压掉
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    BGE_M3_PATH=/models/bge-m3 \
    KB_CACHE_DIR=/kb

# libgl1 / libglib2.0-0：Pillow、PyMuPDF 的运行时依赖
# libgomp1：OpenMP，torch 与 ctranslate2（faster-whisper 后端）都要
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 单独一层先装：它是最大的一块（含 CUDA 运行时约 2.5GB），
# 放前面能让改依赖时复用它这层缓存，不用重下。
# 锁 2.5.1 是为了和 transformers 4.44.2 / FlagEmbedding 1.3.3 同一时期，
# 装最新版有踩到不兼容的风险。
RUN pip install --no-cache-dir torch==2.5.1

# 依赖清单直接用研发目录那一份 —— 不另抄一份，抄了迟早对不上。
COPY 研发/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 应用代码
COPY 研发/*.py /app/

# 知识库种子。首次启动时若数据卷里没有知识库，入口脚本会把这份种子复制进去，
# 这样单纯的 `docker run`（不带任何挂载配置）也能直接跑起来。
COPY 研发/kb_cache /app/seed_kb

COPY 部署/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 7860 是 Gradio 默认端口，可用 PORT 环境变量覆盖
EXPOSE 7860

# 健康检查：探 Gradio 首页。容器「运行中」不等于「服务可用」，
# 验收要求「在指定端口提供服务」，用健康检查把这件事说清楚。
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/" >/dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
