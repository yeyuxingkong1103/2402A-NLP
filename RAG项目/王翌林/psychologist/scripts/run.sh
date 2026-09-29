#!/usr/bin/env bash
# ============================================================
# 启动脚本：启动 FastAPI 后端（uvicorn），加载 BGE-M3 与 Reranker
# 用法：bash scripts/run.sh [--reload] [--port 8000]
# ============================================================
# set -euo pipefail 严格模式：命令出错立即退出、禁止未定义变量、管道任一环节失败即失败
set -euo pipefail

# 定位项目根目录并进入，保证 PID 文件、日志、相对路径都以项目根为基准
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 加载 .env：set -a 让变量自动 export 给子进程，便于 uvicorn/python 读取数据库等配置
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# ${VAR:-默认值}：环境变量已设置则沿用，否则用默认值，方便部署时覆盖
PYTHON_BIN="${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
# PID 文件用于记录后台 uvicorn 的进程号，供 shutdown.sh 精确停止、避免误杀
PID_FILE="$PROJECT_ROOT/data/uvicorn.pid"
LOG_FILE="$LOG_DIR/uvicorn.out"

# 解析命令行参数：把用户传入的 --reload / --port 转成 uvicorn 需要的启动参数
RELOAD=""
for arg in "$@"; do
  case "$arg" in
    --reload) RELOAD="--reload" ;;
    --port) shift; API_PORT="$1" ;;
  esac
done

# 确保日志目录与数据目录存在（-p 已存在也不报错），否则 nohup 重定向日志会失败
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/data"

# 防重复启动：PID 文件存在且进程存活（kill -0 只探测不发送信号）则视为已在运行
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[!] 服务已在运行，PID=$(cat "$PID_FILE")"
  exit 0
fi

echo "[*] 检查依赖服务 ..."
# 依次探测 MySQL/Redis/Milvus 端口（/dev/tcp 无需额外工具，timeout 2 防止卡死）
for pair in "${DB_HOST:-127.0.0.1}:${DB_PORT:-3307}:MySQL" \
            "${REDIS_HOST:-127.0.0.1}:${REDIS_PORT:-6379}:Redis" \
            "${MILVUS_HOST:-127.0.0.1}:${MILVUS_PORT:-19530}:Milvus"; do
  # 用冒号拆分 host:port:name 三部分（注意 host 是 IP，无冒号，拆分安全）
  IFS=':' read -r h p name <<< "$pair"
  if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$h/$p" 2>/dev/null; then
    echo "[√] $name OK ($h:$p)"
  else
    echo "[!] $name 不可用 ($h:$p)，相关功能可能降级"
  fi
done

echo "[*] 启动后端：http://$API_HOST:$API_PORT （日志：$LOG_FILE）"
cd "$PROJECT_ROOT"
# nohup 让进程脱离终端、忽略 SIGHUP，终端关闭后服务仍存活；
# 末尾 & 放入后台运行，>> 将 stdout 和 stderr 一并追加到日志文件
PYTHONPATH="$PROJECT_ROOT" nohup "$PYTHON_BIN" -m uvicorn src.main:app \
  --host "$API_HOST" --port "$API_PORT" --workers "${API_WORKERS:-1}" $RELOAD \
  >> "$LOG_FILE" 2>&1 &

# $! 是刚启动的后台进程 PID，写入 PID 文件供 shutdown.sh 使用
echo $! > "$PID_FILE"
# 留 3 秒让 uvicorn 完成启动（加载模型、绑定端口），随后再做存活探测
sleep 3

# kill -0 发送空信号（不终止进程），仅探测进程是否存在，用于判断是否启动成功
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[√] 服务已启动 PID=$(cat "$PID_FILE")"
  echo "    健康检查： curl http://127.0.0.1:$API_PORT/health"
  echo "    接口文档： http://127.0.0.1:$API_PORT/docs"
else
  # 进程已退出，说明启动失败，打印日志尾部便于排查
  echo "[x] 启动失败，请查看日志：$LOG_FILE"
  tail -n 40 "$LOG_FILE" || true
  exit 1
fi