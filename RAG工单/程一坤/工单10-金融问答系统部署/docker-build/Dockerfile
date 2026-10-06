# 金融问答系统 Docker 镜像
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 构建命令：docker build -t rag-qa:1.0 .
# 基础镜像：python:3.11-slim（精简版，体积小、无多余系统工具）
FROM python:3.11-slim

# 设定容器内工作目录，后续 COPY/RUN 的相对路径均基于此
WORKDIR /app

# 系统依赖（PyMuPDF 需要的运行库）
# libglib2.0-0/libgl1：PyMuPDF 渲染 PDF 时的底层图形库依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*   # 清理 apt 缓存，减小镜像体积；--no-install-recommends 不装推荐包

# Python 依赖
# 先单独 COPY requirements.txt 再 pip install，是为了利用 Docker 层缓存：依赖不变时重建镜像不用重装
COPY requirements.txt /app/requirements.txt
# 用清华 pip 镜像加速国内构建；--no-cache-dir 不留 pip 缓存，减小镜像体积
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 应用代码与预构建的向量索引
# 公共模块（config/pdf_parser/chunker/vector_store/ollama_client/rag_engine 等）
COPY common /app/common
# 工单01应用代码（app.py：Flask 实例与 /api/ask 路由）
COPY app /app/app
# 预构建的向量索引（避免容器内重建，启动即用）
COPY index_store /app/index_store
# 容器入口启动脚本（挂 sys.path 并启动 Flask）
COPY server.py /app/server.py

# Ollama 服务地址：宿主机部署时通过 docker run 环境变量覆盖
# host.docker.internal 指向宿主机（Linux 下需 compose extra_hosts 配合）
ENV OLLAMA_BASE=http://host.docker.internal:11434
# LLM 模型名：Ollama 中的 qwen2.5 7B 指令版
ENV LLM_MODEL=qwen2.5:7b-instruct
# Embedding 模型：bge-m3（Ollama 拉取的量化版，dim=1024）
ENV EMBED_MODEL=dengcao/bge-m3:567m

# 声明容器监听 8601 端口（与 server.py 的 app.run(port=8601) 一致）
EXPOSE 8601

# 健康检查：每 30s 探测一次、5s 超时、连续失败 3 次判为不健康
# 用容器内 python 直接请求本机 8601 根路径，验证 Flask 服务存活
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8601/')" || exit 1

# 容器默认启动命令：运行入口脚本启动 Flask 服务
CMD ["python", "/app/server.py"]
