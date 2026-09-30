#!/usr/bin/env bash
# ============================================================
# 环境安装脚本：检查 WSL/Ubuntu、Python、依赖、MySQL/Redis/Milvus 连通性
# 依赖已安装在固定虚拟环境：/home/dabaie/code/my_project/.venv
# 用法：bash scripts/install.sh
# ============================================================
# set -euo pipefail 严格模式：
#   -e  任何命令返回非零就立即退出，避免带病继续执行；
#   -u  引用未定义变量即报错，尽早暴露拼写错误；
#   -o pipefail  管道中任一命令失败即视为整条管道失败，防止中间出错被吞掉。
set -euo pipefail

# 定位项目根目录（脚本所在目录的上一级），再 cd 进去，保证后续所有相对路径都以项目根为基准
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 读取 .env 环境变量文件：set -a 会把 source 进来的变量自动 export，
# 使后续 python 脚本、端口探测等子进程能读到 DB_HOST / DB_PORT 等配置；set +a 恢复默认。
if [ -f .env ]; then
  set -a; source .env; set +a
else
  # 首次部署可能还没有 .env，从模板复制一份再加载，让脚本能继续跑下去
  echo "[!] 未找到 .env，正在从 .env.example 复制"
  cp .env.example .env
  set -a; source .env; set +a
fi

# 固定虚拟环境路径；${VAR:-默认值} 表示"已设置则用环境变量，否则用默认值"，便于外部覆盖
VENV_PATH="${VENV_PATH:-/home/dabaie/code/my_project/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
PIP_BIN="$VENV_PATH/bin/pip"

echo "=============================================="
echo " 基于 RAG 的心理医生多角色陪伴系统 - 环境安装"
echo "=============================================="

# 1. 检查操作系统
if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
  echo "[!] 当前系统不是 Ubuntu，脚本按 Ubuntu 22.04/24.04 设计，请确认后继续"
else
  . /etc/os-release
  echo "[√] 操作系统：$PRETTY_NAME"
fi

# 2. 检查虚拟环境
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[x] 虚拟环境不存在：$VENV_PATH"
  echo "    如未创建，请先执行： conda create -p $VENV_PATH python=3.10 -y"
  exit 1
fi
echo "[√] Python：$("$PYTHON_BIN" -V 2>&1)"

# 3. 安装依赖
echo "[*] 安装 Python 依赖（requirements.txt）..."
"$PIP_BIN" install --upgrade pip >/dev/null
"$PIP_BIN" install -r requirements.txt
echo "[√] 依赖安装完成"

# 4. 检查本地模型
for model_path in "${EMBEDDING_MODEL_PATH}" "${RERANKER_MODEL_PATH}"; do
  if [ -d "$model_path" ]; then
    echo "[√] 模型存在：$model_path"
  else
    echo "[x] 模型缺失：$model_path（请先下载 BGE-M3 / BGE-Reranker-v2-M3）"
    exit 1
  fi
done

# 5. 检查 MySQL / Redis / Milvus
# 端口探测函数：利用 bash 内置 /dev/tcp 伪设备尝试建立 TCP 连接，无需额外安装 nc/telnet。
# cat < /dev/null 表示不发送任何数据、立即关闭写端，只验证"能否连上"。
# timeout 3 限定 3 秒超时，避免对无响应的端口无限等待。
check_port() {
  local host="$1" port="$2" name="$3"
  if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
    echo "[√] $name 可连通：$host:$port"
  else
    echo "[x] $name 无法连接：$host:$port（请先启动服务）"
  fi
}
check_port "${DB_HOST:-127.0.0.1}" "${DB_PORT:-3307}" "MySQL"
check_port "${REDIS_HOST:-127.0.0.1}" "${REDIS_PORT:-6379}" "Redis"
check_port "${MILVUS_HOST:-127.0.0.1}" "${MILVUS_PORT:-19530}" "Milvus"

# 6. 初始化数据库与角色数据
echo "[*] 初始化 MySQL 表结构 / Milvus Collection / 三个心理医生角色 ..."
"$PYTHON_BIN" scripts/init_db.py

echo
echo "=============================================="
echo " 安装完成 ✅"
echo " 启动服务： bash scripts/run.sh"
echo " 停止服务： bash scripts/shutdown.sh"
echo " 接口文档： http://127.0.0.1:${API_PORT:-8000}/docs"
echo "=============================================="