# -*- syntax=docker/dockerfile:1 -*-
# 【金融问答服务镜像 · Dockerfile】python:3.11-slim 轻量基座 + 非root + 健康检查 + 时区
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 构建命令（国内网络可显式覆盖 pip 源）：
#   docker build -t finqa-service:1.0 .
#   docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ -t finqa-service:1.0 .

# ---------- 基础镜像层：slim 变体体积小，纯 CPU 场景无需完整发行版工具链 ----------
FROM python:3.11-slim

# pip 源构建参数（默认清华源，可按构建环境覆盖）；不写入运行期 ENV，避免污染容器环境
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# ---------- 运行期环境变量 ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    FINQA_DATA_DIR=/data \
    FINQA_INDEX_DIR=/app/index \
    FINQA_PORT=8000

# ---------- 系统层：仅安装时区数据 ----------
# 说明：本服务只做 PDF 文本抽取与 JSON 输出，不做页面栅格化，因此不安装
# fonts-noto-cjk（约 300MB）；如后续扩展页面渲染/OCR 截图能力，再追加中文字体包。
# C.UTF-8 已完整支持中文编解码，无需安装 locales 重新生成 zh_CN.UTF-8。
# 国内构建网络访问 deb.debian.org 不稳定，先把 Debian 源替换为阿里云镜像；
# 兼容 bookworm 的 sources.list 与 trixie 的 deb822 格式 debian.sources。
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi; \
    apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo "${TZ}" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 依赖层：先单独拷贝 requirements 以利用 Docker 分层缓存 ----------
COPY requirements.txt ./
RUN pip install --no-cache-dir \
      --index-url ${PIP_INDEX_URL} \
      --trusted-host ${PIP_TRUSTED_HOST} \
      -r requirements.txt

# ---------- 应用层 ----------
COPY app ./app

# ---------- 非 root 用户与卷目录 ----------
# 固定 uid=1000 的普通用户（便于宿主机按 uid 配卷权限；不加 --system 以避开
# Debian SYS_UID_MAX=999 的 useradd 告警）
RUN useradd --uid 1000 --create-home --home-dir /home/app --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/index /data \
    && chown -R appuser:appuser /app /home/app \
    && chmod 755 /data
USER appuser

# 声明持久化挂载点：/data 放 PDF 语料，/app/index 放索引（支持容器间共享同一卷）
VOLUME ["/data", "/app/index"]

EXPOSE 8000

# ---------- 健康检查：首次建库最长宽限 180 秒 ----------
# 用标准库 urllib 探活 /health，无需在镜像内安装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4); sys.exit(0 if r.status==200 else 1)"

# ---------- 入口 ----------
ENTRYPOINT ["python", "app/service.py"]
