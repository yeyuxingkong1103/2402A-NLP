#!/usr/bin/env bash
# ============================================================
# shutdown.sh —— 服务停止脚本（第 11 步实现）
# 作用：按 run.sh 写下的 PID 文件，先 SIGTERM 再 SIGKILL 停掉前后端
# 用法：bash scripts/shutdown.sh
# ============================================================

set -uo pipefail                  # 不启用 -e：停止过程允许"进程已不在"这类失败继续往下走

cd "$(dirname "$0")/.."           # 切到项目根目录
PROJECT_DIR="$(pwd)"              # 项目根目录绝对路径

# ---------- 输出辅助 ----------
info() { echo -e "\033[32m[信息]\033[0m $*"; }        # 绿色：正常进度
warn() { echo -e "\033[33m[警告]\033[0m $*"; }        # 黄色：可以容忍的异常
err()  { echo -e "\033[31m[错误]\033[0m $*" >&2; }    # 红色：需要人工处理

# 探测 host:port 是否还开着；开着返回 0
port_open() {
  local host="$1" port="$2"                           # 主机与端口
  timeout 3 bash -c "exec 3<>/dev/tcp/$host/$port" >/dev/null 2>&1   # bash 内建 TCP，3 秒超时
}

# ============================================================
# 核心函数：按 PID 文件停掉一个服务
# 入参：PID 文件路径、服务显示名
# ============================================================
stop_by_pidfile() {
  local pidfile="$1" label="$2"                       # 拆出两个入参

  if [ ! -f "$pidfile" ]; then                        # PID 文件不存在
    warn "$label：找不到 $pidfile，可能没用 run.sh 启动过"
    return 0                                          # 不算错误，继续处理下一个
  fi

  local pid                                           # 待停止的进程号
  pid="$(tr -d '[:space:]' < "$pidfile")"             # 读文件并去掉空白与换行
  if [ -z "$pid" ] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then   # 内容不是纯数字
    warn "$label：$pidfile 内容不是合法 PID（[$pid]），只清理该文件"
    rm -f "$pidfile"                                  # 删掉坏文件
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then               # 进程已经不在了
    warn "$label：PID $pid 已不存在，直接清理 PID 文件"
    rm -f "$pidfile"                                  # 清理文件
    return 0
  fi

  info "$label：正在停止 PID $pid ..."
  pkill -P "$pid" 2>/dev/null || true                 # 先杀子进程：npm 会派生出 vite，只杀父进程会留孤儿
  kill -15 "$pid" 2>/dev/null || true                 # 先礼：SIGTERM 让进程自己收尾

  local i                                             # 等待计数器
  for i in 1 2 3 4 5; do                              # 最多等 5 秒
    if ! kill -0 "$pid" 2>/dev/null; then break; fi   # 已经退出就立刻跳出
    sleep 1                                           # 每秒复检一次
  done

  if kill -0 "$pid" 2>/dev/null; then                 # 5 秒后还活着
    warn "$label：5 秒内未退出，改用 kill -9 强制结束"
    kill -9 "$pid" 2>/dev/null || true                # 后兵：SIGKILL
    sleep 1                                           # 给内核一点回收时间
  fi

  if kill -0 "$pid" 2>/dev/null; then                 # 强杀后仍在
    err "$label：PID $pid 仍未停止，请手动执行 kill -9 $pid 检查"
  else                                                # 已停止
    info "$label：已停止（原 PID $pid）"
  fi

  rm -f "$pidfile"                                    # 无论结果如何都清掉 PID 文件，避免下次误用
}

echo "===== 停止服务 ====="

# 停后端与前端；两个都各自独立处理，一个失败不影响另一个
stop_by_pidfile "$PROJECT_DIR/logs/backend.pid"  "后端 uvicorn"
stop_by_pidfile "$PROJECT_DIR/logs/frontend.pid" "前端 vite"

# ============================================================
# 打印停止结果：再看一眼端口是否真的释放了
# ============================================================
echo
echo "===== 停止结果 ====="
for pair in "后端 8000" "前端 5173"; do               # 逐个检查端口
  set -- $pair                                        # 拆成名称与端口两个位置参数
  label="$1"; port="$2"                               # 赋给具名变量，便于阅读
  if port_open 127.0.0.1 "$port"; then                # 端口仍在监听
    warn "$label：端口 $port 仍在监听，可能有非 run.sh 启动的进程占用"
  else                                                # 端口已释放
    info "$label：端口 $port 已释放"
  fi
done
echo
info "如需重启，执行：bash scripts/run.sh"
