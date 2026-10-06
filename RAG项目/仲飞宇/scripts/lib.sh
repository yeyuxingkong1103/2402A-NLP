#!/usr/bin/env bash
# scripts/ 下启动/停止类脚本的公共部分。
#
# 用法：先 cd 到仓库根，再 `source scripts/lib.sh`（本文件里的相对路径按仓库根解析）。
# 刻意没有 main、不直接执行：它是库，不是脚本。
#
# 只有真正重复的才放这里。颜色常量**不**放：全仓只有 start_all.sh 输出带色，
# 搬过来只会让其它脚本平白依赖一堆用不到的变量。

# ---------- pid 文件 ----------
# 读 pid 文件里的 pid；文件不存在/为空都返回空串，且**恒返回 0**。
# 恒返回 0 是必需的：调用方普遍开着 `set -euo pipefail`，若这里返回非 0，
# `p="$(pid_of ...)"` 这种赋值会把整支脚本带走。
pid_of() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    cat "$f" 2>/dev/null || true
    return 0
}

# 该 pid 文件对应的进程是否还活着（文件在、pid 非空、kill -0 通过）
is_running() {
    local p
    p="$(pid_of "$1")"
    [[ -n "$p" ]] || return 1
    kill -0 "$p" 2>/dev/null
}

# 按 pid 文件停进程：stop_by_pidfile <显示名> <pidfile> [兜底匹配串]
#
# 有 pid 文件就按 pid 停（进程已不在也照常清掉文件，免得下次误判）；
# 没有 pid 文件时才退回 pkill 匹配串——那只在脚本被强杀、pidfile 没落盘时发生。
# 没给匹配串就只报告，不做任何兜底 kill：宁可留个孤儿让人看见，也别误杀无关进程。
stop_by_pidfile() {
    local name="$1" pidfile="$2" pattern="${3-}"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid="$(cat "$pidfile" 2>/dev/null || echo "")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "已停止 $name（pid $pid）"
        else
            echo "$name 进程 ${pid:-?} 已不存在"
        fi
        rm -f "$pidfile"
    elif [[ -n "$pattern" ]] && pkill -f "$pattern" 2>/dev/null; then
        echo "已停止 $name"
    else
        echo "$name 未在运行"
    fi
}

# 等 HTTP 就绪：wait_health <url> [最多试几次=30] [响应落盘路径] [每次间隔秒=1]
# 就绪返回 0；超时返回 1（调用方自己决定报错还是继续——重排服务超时只是降级项）。
# 第三个参数用于把 /health 的响应体留成文件（start_all.sh 要用它打印组件表格）；
# 间隔单独可调，因为重排服务冷启动要 1–3 分钟，探测太密没意义（start_all 用 3 秒一次）。
wait_health() {
    local url="$1" tries="${2:-30}" out="${3-}" interval="${4:-1}" i
    for ((i = 1; i <= tries; i++)); do
        if [[ -n "$out" ]]; then
            curl -sf -m 5 -o "$out" "$url" >/dev/null 2>&1 && return 0
        else
            curl -sf -m 5 "$url" >/dev/null 2>&1 && return 0
        fi
        sleep "$interval"
    done
    return 1
}
