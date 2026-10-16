# syntax=docker/dockerfile:1
# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/docker/Dockerfile —— 金融问答系统（RAG）容器镜像定义
#
# 构建（在 工单10/ 目录下执行）：
#     docker build -f 部署/docker/Dockerfile -t rag-finqa:1.0.0 .
#
# 关键设计（每条都对应验收标准）：
#   1. 多阶段构建：builder 装编译依赖并产出 wheel，runtime 只搬运 wheel，
#      不把 gcc/g++/cmake 带进最终镜像 → 体积显著下降（优化前后对比见 优化/部署优化说明.md）。
#   2. 非 root 用户：uid/gid 10001 的 rag 用户，以 USER rag 运行 → 容器逃逸风险收敛。
#   3. HEALTHCHECK：内置 python 探针（不额外装 curl），命中 /healthz。
#   4. 时区 + 中文字体：TZ=Asia/Shanghai、fonts-wqy-zenhei，保证中文日志与年报文本渲染不乱码。
#   5. PYTHONUNBUFFERED=1：日志实时刷出，docker logs 才看得到（否则管道缓冲会吞日志）。
#   6. 业务代码零复制：直接 COPY 研发/app 与研发/service（与宿主机同一份源码）。
# =============================================================================

# ---------------------------------------------------------------------------
# 阶段一：builder —— 编译 wheel（带 gcc/g++/python3-dev/libgomp）
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONDONTWRITEBYTECODE=1

# 国内镜像加速（构建超时是实测踩过的坑，见 部署/过程问题记录.md 问题3）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential python3-dev; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY 部署/docker/requirements.txt ./requirements.txt

# 只装运行时真正需要的包：numpy（向量运算）/ faiss-cpu（向量检索）/ fastapi+uvicorn（API）
# / pymupdf（PDF 解析）/ jieba（中文分词）/ redis（会话共享，可选但装上更省事）
RUN pip install --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        --upgrade pip setuptools wheel \
    && pip wheel --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        --wheel-dir /wheels -r requirements.txt

# ---------------------------------------------------------------------------
# 阶段二：runtime —— 只带运行时依赖，非 root，含健康检查
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="金融问答系统（RAG）" \
      org.opencontainers.image.description="工单编号：人工智能NLP-RAG 项目-金融问答系统部署" \
      org.opencontainers.image.version="1.0.0" \
      work_order="人工智能NLP-RAG 项目-金融问答系统部署"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # —— 容器内路径（与 docker-compose.yml 的卷挂载点一一对应）——
    APP_HOME=/app \
    RAG_DATA__INDEX_DIR=/app/data/index \
    RAG_LOG__DIR=/app/logs \
    DEPLOY_API_HOST=0.0.0.0 \
    DEPLOY_API_PORT=8200 \
    DEPLOY_LOG_DIR=/app/logs \
    DEPLOY_SESSION_DIR=/app/sessions \
    DEPLOY_UPLOAD_DIR=/app/data/uploads \
    DEPLOY_MODEL_DIR=/app/data/models \
    DEPLOY_REDIS_URL=redis://redis:6379/0 \
    # 宿主机 Ollama（Windows 上用 host.docker.internal，见 部署/过程问题记录.md 问题4）
    RAG_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 运行时系统包：时区数据 + 中文字体（wqy 文泉驿，开源可商用）+ libgomp（faiss 依赖）
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        tzdata fonts-wqy-zenhei libgomp1 ca-certificates; \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime; echo "$TZ" > /etc/timezone; \
    rm -rf /var/lib/apt/lists/*

# 非 root 用户（固定 uid/gid，绑定挂载卷的属主可预测）
RUN set -eux; \
    groupadd -g 10001 rag; \
    useradd -u 10001 -g 10001 -m -s /bin/bash rag

COPY --from=builder /wheels /wheels
COPY 部署/docker/requirements.txt /tmp/requirements.txt
RUN pip install --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        --no-index --find-links=/wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt

WORKDIR /app

# 业务源码（与宿主机 研发/ 同一份，零复制）
COPY 研发/app ./app
COPY 研发/service ./service
COPY 部署/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY 部署/docker/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/logs /app/sessions /app/data/index /app/data/models /app/data/uploads \
    && chown -R rag:rag /app

# 卷挂载点（compose 里与具名卷/绑定挂载对应；这里声明为 VOLUME 以便 docker run 也自动建卷）
VOLUME ["/app/logs", "/app/sessions", "/app/data/index", "/app/data/models", "/app/data/uploads"]

USER rag

# 暴露两个端口：8200 = FastAPI 正式 API；8201 = 工单7 Web 界面（UI 容器用同一镜像换命令）
EXPOSE 8200 8201

# 健康检查：std 库探针访问 /healthz（不装 curl，减小体积）
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "/usr/local/bin/healthcheck.py"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8200"]
