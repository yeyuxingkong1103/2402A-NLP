#!/usr/bin/env bash
# ============================================================
# scripts/deploy.sh —— 简化版部署脚本（Linux，非 root 也能跑）
#
# 用法：bash scripts/deploy.sh
# 环境变量：APP_DIR（默认 /opt/rag-roleplay）
#
# 与 deploy/install.sh 的区别（两者都做部署，定位不同）：
#   deploy/install.sh  完整安装：装系统包、拉代码、建 venv、注册 systemd 服务
#                      需要 root，适合全新服务器的一次性初始化
#   本脚本            轻量部署：只拉起中间件容器 + 建 venv + 装依赖 + 探活
#                      不需要 root，适合"机器上已有代码、只想起服务"
# ============================================================
set -euo pipefail
# set -euo pipefail 说明同 start.sh

APP_DIR="${APP_DIR:-/opt/rag-roleplay}"

# command -v 探测命令是否存在。注意这里用 >/dev/null 屏蔽掉路径输出，
# 只关心退出码 —— 找不到时 command -v 返回非 0，配合 set -e 会中断脚本，
# 所以下面紧接着就是手写的友好提示（比让脚本裸奔失败好得多）。
if ! command -v docker >/dev/null; then
  echo "请先安装 Docker 和 Docker Compose"
  exit 1
fi

cd "$APP_DIR"

# 只拉中间件（MySQL / Redis / Milvus / etcd / minio / Attu）。
# 应用本身不在容器里跑，而是直接用宿主机上的 Python 进程 —— 这样改代码不用重建镜像，
# 更适合开发/演示场景。
docker compose -f docker-compose.middleware.yml up -d

# 建独立 venv 而不是装进系统 Python：避免污染系统环境，也便于清理
python3 -m venv .venv
. .venv/bin/activate
# 先升级 pip 本身：老版本 pip 解析新版依赖的元数据时可能出错
python -m pip install -U pip
python -m pip install -r requirements.txt

# 探活。末尾 `|| true` 是刻意的：此时应用还没启动（本脚本只准备环境，不起服务），
# 探活必然失败。加 || true 让脚本不会因此中断，
# 同时把探测结果打印出来，让人看到"环境就绪，但服务尚未启动"这个状态。
bash scripts/health_check.sh || true
