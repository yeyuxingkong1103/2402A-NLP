# =============================================================================
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 阶段：部署 / Docker 镜像（应用侧；推理服务由 docker-compose 的 vllm 容器提供）
# =============================================================================
# 构建（在**仓库根目录**执行，构建上下文必须是仓库根）：
#     docker build -f 部署/docker/Dockerfile -t rag-pdf-qa:2.0.0 .
# 运行（单独跑应用，推理服务需另起）：
#     docker run --rm -p 8600:8600 \
#       -e RAG_LLM__BACKEND=openai -e RAG_LLM__BASE_URL=http://host.docker.internal:8000/v1 \
#       -v "$PWD/研发/data:/workspace/研发/data" -v "$PWD/部署/日志:/workspace/部署/日志" \
#       rag-pdf-qa:2.0.0
#
# 说明
#   - 本镜像**未在本机构建**：本机 Docker 守护进程不可访问（named pipe 权限拒绝），
#     且本机断网无法拉取基础镜像。状态：**已交付待云端验证**。
#   - GPU 不在本镜像内声明：推理走独立的 vLLM 容器（见 docker-compose.yml），
#     应用容器只需 CPU + 网络。若要让嵌入/重排也走 GPU，另行挂载 nvidia runtime。
# =============================================================================
FROM python:3.11-slim

LABEL org.opencontainers.image.title="rag-pdf-qa" \
      org.opencontainers.image.description="工单2 人工智能NLP-RAG-基于PDF文档的问答系统优化（应用侧）" \
      org.opencontainers.image.version="2.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8

# 运行期需要的系统包：
#   curl  —— 容器健康检查（探测 LLM 后端 / 界面端口）
#   libgomp1 —— numpy/torch CPU 版的 OpenMP 运行时
#   fonts-noto-cjk —— 中文渲染（若在容器内出图/导出 PDF 时需要）
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- 依赖分层：先装依赖，源码变更不会让依赖层失效 ----
COPY 研发/requirements.txt /tmp/requirements-core.txt
COPY 部署/配置/requirements-cloud.txt /tmp/requirements-cloud.txt
# 核心依赖必装；云端额外依赖（streamlit/loguru/pdfplumber/ragas/chromadb/pandas…）
# 若因网络或体积原因失败，可注释掉第二行，应用仍可退回标准库界面与等价实现。
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements-core.txt \
 && pip install -r /tmp/requirements-cloud.txt

# ---- 源码与交付物 ----
COPY 研发/ /workspace/研发/
COPY 测试/ /workspace/测试/
COPY 优化/ /workspace/优化/
COPY 部署/ /workspace/部署/
COPY 设计/ /workspace/设计/
COPY 环境事实.md /workspace/环境事实.md

# 索引与日志默认落点（用卷覆盖以持久化；索引必须先构建）
RUN mkdir -p /workspace/研发/data/index /workspace/研发/data/processed /workspace/部署/日志

# 默认走云端组合：OpenAI 兼容后端（vLLM/SGLang）+ Streamlit 界面
ENV RAG_LLM__BACKEND=openai \
    RAG_LLM__BASE_URL=http://vllm:8000/v1 \
    RAG_LLM__MODEL=qwen2.5-7b-instruct \
    RAG_EMBEDDING__BACKEND=ollama \
    RAG_EMBEDDING__OLLAMA_BASE_URL=http://ollama:11434 \
    RAG_APP__LOG_JSON=true

EXPOSE 8600

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8600/ >/dev/null || exit 1

# 界面入口：Streamlit（云端已装）；无 streamlit 时改用：
#   CMD ["python", "研发/app/ui/serve_fallback.py", "--host", "0.0.0.0", "--port", "8600"]
CMD ["python", "-m", "streamlit", "run", "研发/app/ui/streamlit_app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8600", "--server.headless", "true"]
