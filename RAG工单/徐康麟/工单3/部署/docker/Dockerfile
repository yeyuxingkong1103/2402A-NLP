# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/docker/Dockerfile —— RAG 服务的容器镜像定义（CPU 基底；GPU 推理另行起服务）
#
# 设计要点
#   1. **不把模型塞进镜像**：本服务的在线主路径是 HTTP 调 Ollama / vLLM / SGLang
#      （研发/app/core/llm_client.py 只用标准库 urllib），镜像里只有 Python 依赖 + 代码 + 索引；
#   2. 依赖分两层：`requirements.txt`（本机基线，已验证版本）为必需；
#      GPU 场景在容器里再用 requirements-cloud.txt + CUDA 版 torch（见文件末尾注释）；
#   3. 预热是硬要求：`app.main` 的 startup 事件会调用 `build_engine(warmup=True)`
#      （jieba 分词、嵌入后端、LLM 后端探测），因此容器启动即完成预热，无需额外步骤；
#   4. 索引与语料通过 volume 挂载（镜像不带 PDF/索引）：`/app/研发/data`、`/app/部署/日志`。
#
# ⚠️ 诚实声明（不得改写）：本机 Docker CLI 存在（29.7.2）但**守护进程在本沙箱不可达**
#   （`docker images` 报 permission denied: npipe），且本机断网无法拉取基础镜像，
#   因此本 Dockerfile **未在本机构建/运行过**。请在算力云按下方命令自验：
#       docker build -f 部署/docker/Dockerfile -t rag-pdf-qa:1.0 .
#       docker run --rm -p 8600:8600 rag-pdf-qa:1.0 check
#
# 构建（**必须在仓库根 E:\gao6gongdan\工单3 下执行**，否则 COPY 路径不成立）：
#   # ① 让 .dockerignore 生效：Docker 只认「构建上下文根目录」下的 .dockerignore
#   Copy-Item 部署/docker/.dockerignore .dockerignore      # PowerShell
#   docker build -f 部署/docker/Dockerfile -t rag-pdf-qa:1.0 .
#   Remove-Item .dockerignore
# =============================================================================
FROM python:3.11-slim

# 工单编号（镜像标签，便于跨工单检索；中文注释规约见 部署/配置/环境事实.md §7）
LABEL work-order="人工智能NLP-RAG-PDF文档的表格解析及检索优化" \
      maintainer="devops" \
      description="PDF 表格解析 + 混合检索 + 带引用答案（Ollama/vLLM/SGLang 可插拔后端）"

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# PYTHONUNBUFFERED=1（上方 ENV 已含）：让 print/stdout 诊断输出立即刷出，便于 docker logs 观察启动过程。
# 说明（T18）：**日志文件**的并发安全不依赖本变量 —— 研发/app/core/logging_conf.py 用
# open(path,"ab",buffering=0)（O_APPEND）+ 单次 write + 跨进程文件锁（msvcrt.locking/flock）落盘，
# 与 Python 层缓冲无关；但两者保持一致便于排障。详见 部署/日志/日志字段说明.md §8「并发写入策略」。
# 注意：日志目录必须挂在**支持文件锁的本地卷**上（见 docker-compose.yml 的 部署/日志 卷）；
# 网络共享卷（SMB/NFS）上文件锁语义可能不生效，此时应改为每进程独立 RAG_LOG__DIR。

WORKDIR /app

# 先装依赖（利用层缓存）：只装 requirements.txt 里的核心项，重型的本地嵌入降级栈
# （torch/transformers/sentence-transformers）按需在构建参数里打开，CPU 场景默认不装。
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/*

COPY 部署/配置/requirements.txt /app/部署/配置/requirements.txt
ARG INSTALL_OPTIONAL_EMBEDDING=0
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      pymupdf==1.28.2 numpy==1.26.4 jieba==0.42.1 \
      fastapi==0.109.0 uvicorn==0.27.0 \
 && if [ "$INSTALL_OPTIONAL_EMBEDDING" = "1" ]; then \
        python -m pip install torch==2.11.0 transformers==4.57.6 sentence-transformers==2.3.1 sentencepiece==0.2.1; \
    fi

# 复制代码与配置（语料/索引/日志走 volume，不复制；见 .dockerignore）
COPY 研发 /app/研发
COPY 部署/脚本 /app/部署/脚本
COPY 部署/配置 /app/部署/配置
COPY 部署/docker/entrypoint.sh /app/部署/docker/entrypoint.sh
COPY run_py.ps1 /app/run_py.ps1
RUN chmod +x /app/部署/docker/entrypoint.sh

# 容器内默认参数（**必须 0.0.0.0**，否则端口映射进不来；HOST 与 PORT 与 config.py 默认同源）
ENV RAG_SERVER__HOST=0.0.0.0 \
    RAG_SERVER__PORT=8600 \
    RAG_DATA__RAW_DIR=/app/研发/data/raw \
    RAG_DATA__PROCESSED_DIR=/app/研发/data/processed \
    RAG_DATA__INDEX_DIR=/app/研发/data/index \
    RAG_LOG__DIR=/app/部署/日志 \
    RAG_LLM__BACKEND=auto \
    RAG_LLM__OLLAMA_BASE_URL=http://ollama:11434 \
    RAG_LLM__PROBE_TIMEOUT_S=0.5 \
    RAG_APP_MODE=api

# 只暴露 FastAPI/标准库界面的端口；vLLM(8000) / SGLang(30000) / Ollama(11434) 是独立服务
EXPOSE 8600

# 健康检查：/api/health（api 模式）。**依赖 curl 或 python，二选一**，这里用 python（镜像自带）
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import json,urllib.request,os; \
url='http://127.0.0.1:'+os.environ.get('RAG_SERVER__PORT','8600')+'/api/health'; \
data=json.load(urllib.request.urlopen(url,timeout=4)); \
assert data.get('ok') is True, 'health not ok'; print('ok', data['retriever']['chunks'])" || exit 1

# tini 作为 1 号进程，保证 Ctrl+C / docker stop 能正确转发信号（避免 uvicorn 变僵尸）
ENTRYPOINT ["/usr/bin/tini", "--", "/app/部署/docker/entrypoint.sh"]
CMD ["api"]

# =============================================================================
# GPU / 算力云用法（镜像本身不装 CUDA 版 torch，避免 CPU 场景拖到数 GB）：
#   # ① 起 vLLM（OpenAI 兼容）——见 compose 的 profile=gpu 或 部署/脚本/run_vllm.sh
#   # ② 让本服务改走 vLLM：
#   docker run --rm -p 8600:8600 \
#     -e RAG_LLM__BACKEND=openai \
#     -e RAG_LLM__OPENAI_BASE_URL=http://<vllm-host>:8000/v1 \
#     -e RAG_LLM__OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct \
#     -v "$PWD/研发/data:/app/研发/data:ro" -v "$PWD/部署/日志:/app/部署/日志" \
#     rag-pdf-qa:1.0 api
#   嵌入：容器内默认走 Ollama（RAG_LLM__OLLAMA_BASE_URL）；若云上不跑 Ollama，
#   可用 RAG_LLM__BACKEND=openai + 专门的嵌入服务，或构建时 INSTALL_OPTIONAL_EMBEDDING=1 走本地模型。
# =============================================================================
