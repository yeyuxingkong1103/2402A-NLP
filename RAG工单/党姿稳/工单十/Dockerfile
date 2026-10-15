# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 镜像说明：基于 python:3.8-slim 构建金融问答服务镜像，采用多阶段构建减小最终镜像体积
# 构建命令：docker build -t rag-finance-qa:1.0.0 .
# 运行命令：docker run -d -p 8000:8000 --name rag-finance-api rag-finance-qa:1.0.0
# 注意：BuildKit 不允许在续行（行尾反斜杠）后追加注释，故注释统一写在指令上方

# ---------------------------------------------------------------------------
# 阶段一：依赖构建（预下载全部 wheel，运行镜像据此离线安装）
# 说明：全部依赖均提供 CPython 3.8 预编译 wheel，无需容器内编译，故不安装编译工具链
# ---------------------------------------------------------------------------

# 使用官方精简 Python 3.8 基础镜像（Python 版本与项目运行环境保持一致）
FROM python:3.8-slim AS builder

# 设置工作目录为 wheel 输出目录
WORKDIR /wheels

# 复制依赖清单到构建目录
COPY requirements.txt /wheels/requirements.txt

# torch 单独从官方 CPU 源获取，避免默认源拉取 CUDA 版本导致镜像膨胀数 GB
# 说明：官方 CPU 索引页给出的链接会 302 跳转到 download-r2.pytorch.org，该 CDN 在部分网络环境不可达，
#       故这里直连 download.pytorch.org 的静态包地址；如需换版本，同步修改下方文件名与 requirements.txt
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp38-cp38-linux_x86_64.whl', '/wheels/torch-2.4.1+cpu-cp38-cp38-linux_x86_64.whl')"

# 其余依赖统一从国内镜像源预编译为 wheel，加速构建
# --find-links /wheels 让 torch==2.4.1+cpu 直接命中上一步下载的本地 wheel；
# 不能像早期版本那样把 torch 行从清单中删掉——删掉后 sentence-transformers 的
# torch>=1.11.0 依赖会失去约束，pip 会转而下载 797MB 的 CUDA 版
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    --find-links /wheels \
    -r /wheels/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---------------------------------------------------------------------------
# 阶段二：运行镜像（仅保留运行时必要内容，体积更小、攻击面更低）
# ---------------------------------------------------------------------------

# 运行阶段同样使用精简基础镜像
FROM python:3.8-slim

# 标注维护者信息
LABEL maintainer="人工智能NLP-RAG项目"

# 标注工单编号
LABEL workorder="人工智能NLP-RAG-金融问答系统部署"

# 标注镜像版本
LABEL version="1.0.0"

# 关闭输出缓冲便于 docker logs 实时查看，并设置时区
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV TZ=Asia/Shanghai

# 设置应用工作目录
WORKDIR /app

# 从构建阶段复制 wheel 文件
COPY --from=builder /wheels /wheels

# 离线安装全部依赖（torch 与其余依赖的 wheel 均已就位），随后删除 wheel 以减小体积
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && rm -rf /wheels

# 复制应用代码
COPY app/ /app/

# 复制内置语料（运行时可被数据卷或挂载目录覆盖）
COPY corpus/ /app/corpus/

# 创建非 root 用户运行服务以降低容器逃逸风险，并预建数据、日志与共享目录
RUN useradd -m -u 1000 raguser \
    && mkdir -p /app/data /app/logs /app/shared \
    && chown -R raguser:raguser /app

# 切换到非 root 用户运行
USER raguser

# 声明服务监听端口
EXPOSE 8000

# 健康检查：调用 /health 接口，容器编排与 docker ps 可据此判断服务可用性
# start-period 必须大于冷启动索引构建耗时：本机 1533 块 CPU 编码实测约 440 秒，
# 设成 180 秒会让健康检查在建索引期间连续失败、把正常服务误判为 unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status==200 else 1)"

# 以 uvicorn 启动服务；--workers 1 因索引与模型为进程内单例，多 worker 会成倍占用内存
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
