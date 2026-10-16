# =============================================================================
# 工单：人工智能NLP-RAG-图像内容解析及检索优化
# 文件：部署/docker/Dockerfile —— FastAPI 服务镜像（**参考交付，未在本机构建**）
#
# ⚠️ 构建状态（如实标注，勿在 README/核验里写成「已构建」）：
#   本机 Docker Desktop 在运行，但 docker CLI 走命名管道
#   npipe:////./pipe/dockerDesktopLinuxEngine，DSH 沙箱默认拒绝：
#       $ docker version
#       → permission denied while trying to connect to the docker API at
#         npipe:////./pipe/dockerDesktopLinuxEngine   （退出码 1）
#   → 因此本文件 **未执行过 `docker build`**；只做过静态校验（见 部署/交付物核验.md §3）：
#     ① `docker compose -f 部署/docker/docker-compose.yml config`（YAML/字段/schema 解析通过）
#     ② 依赖清单与版本取自本机实测（设计/环境事实.md §2.3），非臆造
#   要真正构建，请在**可访问 Docker 守护进程**的环境执行：
#       docker build -f 部署/docker/Dockerfile -t gongdan4-rag-api:local .
#   （构建上下文 = 仓库根 E:\gao6gongdan\工单4）
#
# 设计取舍（都与「本机实况」有关，别当成通用最佳实践）：
#   1) 镜像**不烘焙**索引与语料（研发/data 数 GB，且含只读语料红线）→ 由 compose 以只读卷挂载；
#   2) 镜像**不烘焙**任何模型权重 → 嵌入/生成走宿主机 Ollama（127.0.0.1:11434）；
#   3) torch / transformers 只在**离线图像解析**阶段用（多模态编码在运行时才 import），
#      本镜像作为 **API 服务** 不装（省数 GB）；需要解析图像时用 `--build-arg WITH_MULTIMODAL=1`。
# =============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/研发 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖清单：版本号与本机实测一致（设计/环境事实.md §2.3；T8 复核）
COPY 部署/docker/requirements.txt /app/部署/docker/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/部署/docker/requirements.txt

# 可选：离线图像解析所需的重依赖（torch/transformers/Pillow），默认不装
ARG WITH_MULTIMODAL=0
COPY 部署/docker/requirements-multimodal.txt /app/部署/docker/requirements-multimodal.txt
RUN if [ "$WITH_MULTIMODAL" = "1" ]; then \
        python -m pip install --no-cache-dir -r /app/部署/docker/requirements-multimodal.txt; \
    fi

# 代码与配置（索引/语料/权重由卷挂载，见 docker-compose.yml）
COPY 研发 /app/研发
COPY 测试 /app/测试
COPY 部署 /app/部署
COPY run_py.ps1 /app/run_py.ps1
COPY README.md /app/README.md

# 无 root 运行（卷挂载的 部署/日志 需可写）
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/部署/日志 && chown -R rag:rag /app/部署/日志
USER rag

EXPOSE 8600

# 健康检查：/health 返回 200（status=ok）才算健康；start_period 给索引加载与预热留时间
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8600/health', timeout=10).status == 200 else 1)"

CMD ["python", "部署/脚本/serve_api.py", "--host", "0.0.0.0", "--port", "8600"]
