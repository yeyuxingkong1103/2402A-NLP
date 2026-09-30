#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../02-研发/核心代码" && pwd)"

PASS=0
WARN=0
FAIL=0

pass() { printf '[PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
warn() { printf '[WARN] %s\n' "$1"; WARN=$((WARN + 1)); }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=$((FAIL + 1)); }

printf 'RAG项目 Linux 环境检查\n'
printf '项目目录: %s\n\n' "$PROJECT_ROOT"

if [[ "$(uname -s)" == "Linux" ]]; then
  pass "操作系统为 Linux"
else
  fail "仅支持 Linux，当前为 $(uname -s)"
fi

ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" ]]; then
  pass "CPU 架构为 x86_64"
else
  warn "当前架构为 ${ARCH}，安装脚本仅自动支持 x86_64"
fi

CPU_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '0')"
if (( CPU_CORES >= 4 )); then
  pass "CPU 核数 ${CPU_CORES}"
else
  warn "CPU 核数 ${CPU_CORES}，建议至少 4 核"
fi

MEM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || printf '0')"
MEM_GB=$((MEM_KB / 1024 / 1024))
if (( MEM_GB >= 8 )); then
  pass "内存约 ${MEM_GB} GiB"
else
  warn "内存约 ${MEM_GB} GiB，建议至少 8 GiB"
fi

DISK_KB="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
DISK_GB=$((DISK_KB / 1024 / 1024))
if (( DISK_GB >= 20 )); then
  pass "项目磁盘可用空间约 ${DISK_GB} GiB"
else
  warn "项目磁盘可用空间约 ${DISK_GB} GiB，建议至少 20 GiB"
fi

for command_name in curl git; do
  if command -v "$command_name" >/dev/null 2>&1; then
    pass "已安装 ${command_name}"
  else
    fail "缺少 ${command_name}"
  fi
done

if command -v conda >/dev/null 2>&1; then
  pass "已安装 Conda: $(conda --version 2>/dev/null)"
else
  warn "未安装 Conda；install.sh 可安装 Miniconda"
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    pass "Docker 可用"
  else
    warn "Docker 已安装但当前用户无法访问 daemon"
  fi
else
  warn "未安装 Docker；如基础服务独立部署可忽略"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
  pass "检测到 NVIDIA GPU: ${GPU_NAME:-unknown}"
else
  warn "未检测到 nvidia-smi；只能使用 CPU 或外部模型服务模式"
fi

if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  pass "找到 requirements.txt"
else
  fail "未找到 requirements.txt"
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  pass "找到 .env"
else
  warn "未找到 .env；安装脚本会从 .env.example 创建"
fi

for port in 8000 19530 3306 6379; do
  if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
    warn "端口 ${port} 已有监听进程"
  fi
done

printf '\n检查汇总: PASS=%d WARN=%d FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
if (( FAIL > 0 )); then
  exit 1
fi
