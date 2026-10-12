#!/bin/bash
# 工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏
# 把剖析与监控依赖装进 RAGFlow 容器
#
# ## 装什么
#
#   py-spy              CPU 热点采样（工单任务二要的「找出 CPU 热点函数」）
#   objgraph            内存对象跟踪（工单任务二要的「对象创建与留存」）
#   prometheus-client   暴露 /metrics（产出物三要的「集成 Prometheus 监控」）
#
# ## 为什么需要这个脚本（两个坑）
#
# 1. **容器里有 uv 没有 pip**：镜像的 venv 是 uv 建的，没有 pip 模块，
#    只能用 `uv pip install --python /ragflow/.venv/bin/python`。
# 2. **容器一重建，装进去的包就没了**：venv 在容器的可写层里。
#    所以每次 `docker compose up -d --force-recreate` 之后要重新跑这个脚本。
#    （这是有意的：不动镜像，保持镜像干净。）
#
# ## 前置条件
#
# 容器必须带 SYS_PTRACE，否则 py-spy 会报
#   Failed to copy Py_Version symbol: Permission denied
# 这一条已经写进 研发/ragflow/docker/docker-compose.yml 的 ragflow-cpu 服务里，
# 用那份 compose 起的容器就自带权限。
#
# ## 用法
#
#   bash install_extras.sh          # 装齐并重启 API 进程（让 /metrics 生效）
#   bash install_extras.sh --check  # 只检查装了什么
set -e

CONTAINER="${CONTAINER:-docker-ragflow-cpu-1}"
PY="/ragflow/.venv/bin/python"

check() {
    # 注意：命令要包在 sh -c 里跑。直接 `docker exec 容器 /ragflow/...`
    # 会被 Git Bash 的路径转换改写成 `C:/Program Files/Git/ragflow/...`，
    # 报 "no such file or directory"。
    docker exec "${CONTAINER}" sh -c \
        "${PY} -c \"import objgraph; print('  objgraph OK')\" 2>&1 | tail -1;
         ${PY} -c \"import prometheus_client; print('  prometheus-client OK')\" 2>&1 | tail -1;
         /ragflow/.venv/bin/py-spy --version 2>&1 | tail -1"
}

if [ "${1:-}" = "--check" ]; then
    echo "==> 容器 ${CONTAINER} 里的剖析/监控依赖："
    check || true
    exit 0
fi

echo "==> 检查容器是否在跑"
docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}" || {
    echo "[错误] 容器 ${CONTAINER} 没在运行" >&2; exit 1; }

echo "==> 安装 py-spy / objgraph / prometheus-client（容器内，约 3MB）"
docker exec "${CONTAINER}" sh -c \
    "uv pip install --python ${PY} py-spy objgraph prometheus-client 2>&1 | tail -2"

echo "==> 重启 API 进程，让 /metrics 开始暴露"
# prom_metrics.install() 是在 ragflow_server.py 启动时调用的，
# 装完 prometheus-client 必须重启进程才生效。
docker exec "${CONTAINER}" sh -c "pkill -f 'api/ragflow_server.py' || true"
sleep 3
docker exec -d "${CONTAINER}" sh -c "cd /ragflow && ${PY} api/ragflow_server.py"
echo "    等它起来（约 40 秒）..."
sleep 40

echo "==> 验证"
check
if curl -s --max-time 10 http://localhost:9390/metrics 2>/dev/null | grep -q "^ragflow_"; then
    echo "  /metrics 端点正常"
else
    echo "  ⚠ /metrics 还没数据，再等一会儿或看容器日志"
fi
