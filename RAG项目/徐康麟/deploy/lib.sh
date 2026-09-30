#!/usr/bin/env bash
# shellcheck shell=bash
# ============================================================================
# deploy/lib.sh —— 四个部署脚本共用的**公共库**
#
# 为什么单独一个库：安装 / 部署 / 启动 / 停止四件事都要「写日志、判存活、等健康、
# 优雅停进程」。复制四份必然走偏（一处改了另外三处忘改），所以集中在这里。
#
# 用法（在脚本里）：
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
#   lr_init "install"          # 初始化：确定日志文件、解析公共环境变量
#   lr_section "第 1 步 ..."    # 打一条醒目分隔
#   lr_info "..."              # 普通信息
#
# 约定：
#   * 所有函数名以 `lr_` 开头，避免与业务命令/其它脚本撞名；
#   * **日志同时打屏与落盘**（`logs/<脚本名>-YYYYmmdd.log`，目录可用
#     `LEGAL_RAG_LOG_DIR` 覆盖）；
#   * 支持 `--dry-run`：`lr_run` 只打印不执行（安装/部署脚本靠它做"先看后跑"）。
#
# 目标平台：**Linux**（Ubuntu 为主）。在 Windows 上用 Git Bash/WSL 也能跑，
# 但 `/proc` 相关函数（识别真 python 进程）只在 Linux 上可靠。
# bash 版本要求：**≥ 4.4**（空数组配 `set -u` 的展开行为；Ubuntu 18.04+ 自带 4.4+）。
# ============================================================================

# 严格模式：未定义变量、管道任一环失败、命令失败都立刻停。
# 这是本项目"绝不静默失败"的底线 —— 宁可停在这里，也不要带着半成品继续往下走。
set -euo pipefail

# ---------------------------------------------------------------------------
# 公共环境变量（都给了默认值；调用方可在 source 之前覆盖）
# ---------------------------------------------------------------------------
#: 日志目录（默认 <项目根>/logs）。放这里而不是 /tmp：重启后还能查。
: "${LEGAL_RAG_LOG_DIR:=}"
#: 是否 dry-run（1=只打印不执行）。由脚本解析 --dry-run 后设置。
: "${LR_DRY_RUN:=0}"
#: 日志级别阈值：仅 "debug" 时额外输出 DEBUG 行，其它一律 info 起。
: "${LR_LOG_LEVEL:=info}"

LR_LEVEL_DEBUG=10
LR_LEVEL_INFO=20
LR_LEVEL_WARN=30
LR_LEVEL_ERROR=40

# ---------------------------------------------------------------------------
# lr_init <脚本名>
#   初始化日志。必须在任何 lr_log 之前调用。
#   参数：$1 = 脚本名（用于日志文件名，例如 install / deploy / start / stop）
# ---------------------------------------------------------------------------
lr_init() {
  LR_NAME="${1:?lr_init 需要脚本名，例如 lr_init install}"
  # 项目根：lib.sh 在 <root>/deploy/ 下
  LR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  LR_ROOT="$(cd "$LR_LIB_DIR/.." && pwd)"

  if [[ -z "$LEGAL_RAG_LOG_DIR" ]]; then
    LEGAL_RAG_LOG_DIR="$LR_ROOT/logs"
  fi
  LR_LOG_FILE="$LEGAL_RAG_LOG_DIR/${LR_NAME}-$(date +%Y%m%d).log"
  mkdir -p "$LEGAL_RAG_LOG_DIR"
  export LEGAL_RAG_LOG_DIR LR_DRY_RUN LR_LOG_LEVEL LR_NAME
}

# ---------------------------------------------------------------------------
# lr_log <级别> <消息...>
#   同时写 stderr（人能立刻看到）与日志文件（事后能查）。
#   ⚠️ 不写 stdout：stdout 留给"机器可读"的输出（例如 pid、URL），
#      这样 `PID=$(bash deploy/start.sh ... | tail -1)` 这类用法不会被日志污染。
# ---------------------------------------------------------------------------
lr_log() {
  local level="$1"; shift || true
  local stamp; stamp="$(date '+%Y-%m-%d %H:%M:%S')"
  local line="$stamp | ${level} | ${LR_NAME:-?} | $*"
  printf '%s\n' "$line" >&2
  if [[ -n "${LR_LOG_FILE:-}" ]]; then
    printf '%s\n' "$line" >>"$LR_LOG_FILE" 2>/dev/null || true
  fi
}

# 级别过滤：默认 info（debug 只在 LR_LOG_LEVEL=debug 时输出）。
# 不搞数值映射表 —— 那玩意儿容易写成"自己跟自己比"的假过滤（曾这么写过）。
lr_debug() {
  [[ "${LR_LOG_LEVEL}" == "debug" ]] || return 0
  lr_log DEBUG "$@"
}
lr_info()  { lr_log INFO  "$@"; }
lr_ok()    { lr_log OK    "$@"; }
lr_warn()  { lr_log WARN  "$@"; }
lr_error() { lr_log ERROR "$@"; }

# lr_ok_if_ran <消息...>：**真执行过**才报 OK；`--dry-run` 时只说明"这一步没有真做"。
#
# 为什么必须有这条（2026-09-29 真机 `bash deploy/selfcheck.sh` 之后的补强）：
#   `lr_run` 在干跑时**只打印并 `return 0`**，于是 `if lr_run …; then lr_ok "安装完成"`
#   会一路打出"pip 安装完成 / 自检通过 / 索引建立完成"—— 而那一步**根本没执行**。
#   这正是本项目明令禁止的**虚报通过**（比不报更糟：运维会以为环境好了）。
#   判据很简单：`lr_ok` 只在"这一步真的跑过"时才允许用，否则用这个助手。
lr_ok_if_ran() {
  if [[ "$LR_DRY_RUN" == "1" ]]; then
    lr_info "[dry-run] 未真正执行（故不报成功）：$*"
    return 0
  fi
  lr_ok "$@"
}

# lr_die <消息...>：记 ERROR 并以 1 退出（调用方可用 `|| lr_die` 兜底）
lr_die() { lr_error "$@"; exit 1; }

# ---------------------------------------------------------------------------
# 步骤分隔：让日志一眼能看出"现在到哪一步了"
# ---------------------------------------------------------------------------
lr_section() {
  lr_log INFO "============================================================"
  lr_log INFO "== $*"
  lr_log INFO "============================================================"
}

# ---------------------------------------------------------------------------
# lr_have <命令名>：命令是否存在（不打印）
# ---------------------------------------------------------------------------
lr_have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# lr_need <命令名> <人话说明>：不存在就 die（附带安装建议由调用方给）
# ---------------------------------------------------------------------------
lr_need() {
  local cmd="$1"; shift
  lr_have "$cmd" || lr_die "缺少命令 '$cmd'：$*"
}

# ---------------------------------------------------------------------------
# lr_run <命令...>：执行命令；dry-run 时只打印
#   ⚠️ 刻意把整条命令作为参数数组传入（`lr_run pip install -r x.txt`），
#      而不是 `eval` 字符串 —— eval 遇到路径里的空格/引号就出事。
# ---------------------------------------------------------------------------
lr_run() {
  if [[ "$LR_DRY_RUN" == "1" ]]; then
    lr_info "[dry-run] 将执行：$*"
    return 0
  fi
  lr_info "执行：$*"
  "$@"
}

# ---------------------------------------------------------------------------
# lr_free_gb <路径>：该路径所在盘可用空间（GB，四舍五入到整数）
# ---------------------------------------------------------------------------
lr_free_gb() {
  local path="$1"
  df -Pk "$path" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

# ---------------------------------------------------------------------------
# lr_need_gb <路径> <最少GB> <用途说明>
#   真机教训（N.6）：pip 的 cache/tmp 与 29G 模型**同盘**，安装峰值把盘写满，
#   装到一半报 `No space left on device`，而安装脚本收尾清理后又显示"还有空间"
#   —— 于是**看起来像随机失败**。所以安装前必须**预检**并打印读数。
# ---------------------------------------------------------------------------
lr_need_gb() {
  local path="$1" need="$2" what="${3:-}"
  local free; free="$(lr_free_gb "$path")"
  if [[ -z "$free" ]]; then
    lr_warn "读不到 $path 的可用空间（df 失败）—— 跳过空间预检，请自行确认"
    return 0
  fi
  if (( free < need )); then
    lr_error "空间不足：$path 所在盘可用 ${free}G，需要至少 ${need}G（$what）"
    lr_error "  处理建议：清 pip 缓存（pip cache purge 只清默认目录！自定义目录要手动 rm）、"
    lr_error "            换 TMPDIR/PIP_CACHE_DIR 到别的盘、或扩容后再装。"
    return 1
  fi
  lr_info "空间预检通过：$path 可用 ${free}G（要求 ≥ ${need}G，$what）"
}

# ---------------------------------------------------------------------------
# lr_port_listening <端口>：本机是否有人在监听该端口
#   优先 ss/lsof；都没有时退回 bash 的 /dev/tcp（纯 bash，无需额外命令）。
# ---------------------------------------------------------------------------
lr_port_listening() {
  local port="$1"
  if lr_have ss; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$" && return 0 || return 1
  fi
  if lr_have lsof; then
    lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0 || return 1
  fi
  (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1 && return 0 || return 1
}

# ---------------------------------------------------------------------------
# lr_wait_http <URL> <超时秒> [期望状态码前缀，默认 2]
#   轮询直到 HTTP 状态码匹配。**用 /livez 而不是 /health**：
#   实测 /health 在依赖不可达时单次要 2.2s 且占住工作线程（见 docs/LOAD-TEST.md §3），
#   拿它做探活会把线程池吃光。
# ---------------------------------------------------------------------------
lr_wait_http() {
  local url="$1" timeout="${2:-60}" expect="${3:-2}"
  local waited=0 code
  while (( waited < timeout )); do
    code="$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$url" 2>/dev/null || true)"
    if [[ "$code" == "$expect"* ]]; then
      lr_ok "健康检查通过：$url -> HTTP $code（等待 ${waited}s）"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited % 10 == 0 )); then
      lr_info "  等待 $url … 已 ${waited}s（最近状态码 '${code}'）"
    fi
  done
  lr_error "健康检查超时：$url（等了 ${timeout}s，期望 ${expect}xx）"
  return 1
}

# ---------------------------------------------------------------------------
# lr_pid_alive <pid>：进程是否还在
# ---------------------------------------------------------------------------
lr_pid_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# ---------------------------------------------------------------------------
# lr_read_pid <pid文件>：读出 pid（文件不存在/内容非法都返回空串，不报错）
# ---------------------------------------------------------------------------
lr_read_pid() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local pid; pid="$(tr -d '[:space:]' <"$file" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] && printf '%s' "$pid"
}

# ---------------------------------------------------------------------------
# lr_stop_pid <pid> <名字> [优雅等待秒数]
#   先 SIGTERM 让它收尾（API 要落盘、vLLM 要释放显存），超时才 SIGKILL。
# ---------------------------------------------------------------------------
lr_stop_pid() {
  local pid="$1" name="$2" grace="${3:-20}" waited=0
  if ! lr_pid_alive "$pid"; then
    lr_info "$name（pid=$pid）已经不在运行"
    return 0
  fi
  if [[ "$LR_DRY_RUN" == "1" ]]; then
    lr_info "[dry-run] 将停止 $name（pid=$pid）"
    return 0
  fi
  lr_info "停止 $name（pid=$pid）：先发 SIGTERM，最多等 ${grace}s"
  kill -TERM "$pid" 2>/dev/null || true
  while (( waited < grace )); do
    lr_pid_alive "$pid" || { lr_ok "$name 已优雅退出（${waited}s）"; return 0; }
    sleep 1
    waited=$((waited + 1))
  done
  lr_warn "$name 在 ${grace}s 内没退出，改用 SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  sleep 2
  lr_pid_alive "$pid" && lr_error "$name（pid=$pid）SIGKILL 后仍在？请人工确认" || \
    lr_ok "$name 已强制退出"
}

# ---------------------------------------------------------------------------
# lr_python_pids <匹配串>：列出**真正的 python 进程** pid
#
# ⚠️ 真机教训（本轮被骗过两次）：`pgrep -f run_api.py | head -1` 取到的是
#    `bash -c '... run_api.py ...'` 这个**包装进程**，它的 `/proc/<pid>/environ`
#    是 source .env **之前**的旧值 —— 于是"读环境变量"读到的全是错的。
#    判据：/proc/<pid>/exe 指向 python（Linux 上可靠）。
# ---------------------------------------------------------------------------
lr_python_pids() {
  local pattern="$1" pid exe
  for pid in $(pgrep -f "$pattern" 2>/dev/null || true); do
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    case "$exe" in
      *python*) printf '%s\n' "$pid" ;;
    esac
  done
}

# ---------------------------------------------------------------------------
# lr_snapshot_runtime <标签>：把"当前进程/端口/显存"打一份快照进日志
#   排障时最想知道的就是"当时到底是什么状态"，所以关键节点都留一份。
# ---------------------------------------------------------------------------
lr_snapshot_runtime() {
  local label="${1:-快照}"
  lr_info "---- 运行时快照：$label ----"
  lr_info "  项目根：$LR_ROOT  日志：${LR_LOG_FILE:-（未初始化）}"
  lr_info "  时间：$(date '+%F %T %Z')"
  lr_info "  主机：$(uname -a 2>/dev/null || echo '未知')"
  if lr_have nvidia-smi; then
    local gpu; gpu="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null | head -3 | tr '\n' ';' || true)"
    lr_info "  GPU：${gpu:-（读不到）}"
  else
    lr_info "  GPU：无 nvidia-smi（无卡环境属正常）"
  fi
}
