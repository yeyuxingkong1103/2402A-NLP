#!/usr/bin/env bash
# ============================================================
# run.sh —— 服务启动脚本（第 11 步实现）
# 作用：在后台拉起后端 FastAPI 与前端 Vite，并把 PID 写进 logs/ 便于停止
# 目标环境：WSL + Ubuntu（部署环境按技术选型选的是 WSL + Ubuntu）
# 用法：bash scripts/run.sh
# ============================================================

set -euo pipefail                 # 失败即退出，避免半启动状态

# ---------- 全局变量 ----------
ENV_NAME="power_rag"              # conda 环境名，与 install.sh 保持一致
MINICONDA_DIR="$HOME/miniconda3"  # Miniconda 安装目录

cd "$(dirname "$0")/.."           # 切到项目根目录
PROJECT_DIR="$(pwd)"              # 项目根目录绝对路径
mkdir -p logs                     # 日志目录不存在就创建

# ---------- 输出辅助 ----------
info() { echo -e "\033[32m[信息]\033[0m $*"; }        # 绿色：正常进度
warn() { echo -e "\033[33m[警告]\033[0m $*"; }        # 黄色：警告但不中断
err()  { echo -e "\033[31m[错误]\033[0m $*" >&2; }    # 红色：错误信息

has_cmd() { command -v "$1" >/dev/null 2>&1; }        # 命令是否存在

# 从 .env 读取一个键的值；没有该键时输出空串
env_get() {
  grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-   # 取第一处匹配，等号右边整段
}

# 探测 host:port 是否可连；通返回 0，不通返回 1
port_open() {
  local host="$1" port="$2"                          # 主机与端口
  timeout 3 bash -c "exec 3<>/dev/tcp/$host/$port" >/dev/null 2>&1   # 用 bash 内建 TCP，3 秒超时
}

# ============================================================
# 第 1 步：检查并激活 conda 环境
# ============================================================
echo "===== 1/5 检查 conda 环境 ====="

if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then           # conda 本身不在预期位置
  err "找不到 conda：$MINICONDA_DIR/bin/conda；请先执行 bash scripts/install.sh"
  exit 1
fi

# shellcheck disable=SC1091
eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"   # 加载 conda 的 shell 钩子

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then  # 环境列表里没有目标环境
  err "conda 环境 $ENV_NAME 不存在；请先执行 bash scripts/install.sh"
  exit 1
fi

conda activate "$ENV_NAME"                             # 激活环境
info "已激活 conda 环境：$CONDA_DEFAULT_ENV（Python $(python -V 2>&1 | cut -d' ' -f2)）"

# ============================================================
# 第 2 步：检查 .env 配置文件
# ============================================================
echo "===== 2/5 检查配置文件 ====="

if [ ! -f .env ]; then                                 # 没有 .env 就无法读数据库与模型配置
  err "配置文件 .env 不存在；请执行 cp .env.example .env 并填写真实值后重试"
  exit 1
fi
info ".env 已就位"

# 从 .env 里取各项依赖的地址与端口，取不到就用默认值
API_PORT="$(env_get API_PORT)";   API_PORT="${API_PORT:-8000}"     # 后端端口
REDIS_HOST="$(env_get REDIS_HOST)"; REDIS_HOST="${REDIS_HOST:-127.0.0.1}"   # Redis 主机
REDIS_PORT="$(env_get REDIS_PORT)"; REDIS_PORT="${REDIS_PORT:-6379}"        # Redis 端口
MYSQL_HOST="$(env_get MYSQL_HOST)"; MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}"   # MySQL 主机
MYSQL_PORT="$(env_get MYSQL_PORT)"; MYSQL_PORT="${MYSQL_PORT:-3306}"        # MySQL 端口
MILVUS_HOST="$(env_get MILVUS_HOST)"; MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"   # Milvus 主机
MILVUS_PORT="$(env_get MILVUS_PORT)"; MILVUS_PORT="${MILVUS_PORT:-19530}"      # Milvus 端口

# ============================================================
# 第 3 步：探测三项依赖的连通性（连不上只警告，不强制退出）
# ============================================================
echo "===== 3/5 探测依赖连通性 ====="

check_dep() {                                          # 入参：名称、主机、端口、补充说明
  local label="$1" host="$2" port="$3" hint="$4"       # 拆出四个入参
  if port_open "$host" "$port"; then                   # 端口通
    info "$label 可连（$host:$port）"
  else                                                 # 端口不通
    warn "$label 连不上（$host:$port）；$hint"          # 只警告，继续往下启动
  fi
}

check_dep "Redis"  "$REDIS_HOST"  "$REDIS_PORT"  "短期记忆与答案缓存会降级：缓存读不到、记忆写不进"
check_dep "MySQL"  "$MYSQL_HOST"  "$MYSQL_PORT"  "用户/会话/消息接口会返回 500"
check_dep "Milvus" "$MILVUS_HOST" "$MILVUS_PORT" "检索会拿不到上下文，知识库状态会显示 0 条"

# ============================================================
# 第 4 步：后台启动后端
# ============================================================
echo "===== 4/5 启动后端 ====="

if has_cmd uvicorn; then                               # 环境里有 uvicorn 命令就直接用
  UVICORN_CMD="uvicorn"
else                                                   # 否则退回模块方式，等价且更稳
  UVICORN_CMD="python -m uvicorn"
fi

# 后台启动并落日志；$! 取到刚启动进程的 PID
nohup $UVICORN_CMD app:app --host 0.0.0.0 --port "$API_PORT" > logs/backend.log 2>&1 &
BACKEND_PID=$!                                         # 记下 PID
echo "$BACKEND_PID" > logs/backend.pid                 # 写入 PID 文件，供 shutdown.sh 使用
info "后端已启动（PID $BACKEND_PID），日志：logs/backend.log"

sleep 8                                                # 等 8 秒让模型与依赖初始化
if kill -0 "$BACKEND_PID" 2>/dev/null; then            # 进程还活着
  info "后端进程存活"
else                                                   # 进程已退出
  err "后端启动失败，请查看 logs/backend.log 末尾的报错"
  exit 1
fi

# ============================================================
# 第 5 步：后台启动前端
# ============================================================
echo "===== 5/5 启动前端 ====="

if ! has_cmd node; then                                # 前端需要 Node
  warn "未检测到 node，跳过前端启动；请先安装 Node.js 18+（推荐 20+）"
else
  cd "$PROJECT_DIR/frontend"                           # 进入前端目录
  if [ ! -d node_modules ]; then                       # 依赖没装过
    info "未发现 node_modules，先执行 npm install（首次会慢一些）"
    npm install                                        # 安装前端依赖
  else                                                 # 已装过
    info "node_modules 已存在，跳过 npm install"
  fi
  # 后台启动 Vite 开发服务器，日志写到项目根的 logs 下
  nohup npm run dev > ../logs/frontend.log 2>&1 &
  FRONTEND_PID=$!                                      # npm 进程的 PID
  echo "$FRONTEND_PID" > ../logs/frontend.pid          # 写入 PID 文件
  info "前端已启动（PID $FRONTEND_PID），日志：logs/frontend.log"
  cd "$PROJECT_DIR"                                    # 回到项目根目录
  sleep 5                                              # 等 Vite 完成启动
fi

# ============================================================
# 打印访问地址
# ============================================================
echo
echo "==================== 启动完成 ===================="
echo "  后端服务   http://localhost:${API_PORT}"
echo "  API 文档   http://localhost:${API_PORT}/docs"
echo "  前端页面   http://localhost:5173"
echo "-------------------------------------------------"
echo "  查看日志   tail -f logs/backend.log"
echo "            tail -f logs/frontend.log"
echo "  停止服务   bash scripts/shutdown.sh"
echo "================================================="
