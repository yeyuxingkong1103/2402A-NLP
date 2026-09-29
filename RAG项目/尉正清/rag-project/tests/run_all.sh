#!/usr/bin/env bash
# 一键执行三类测试，结果统一落到 tests/reports/
#
#   ./tests/run_all.sh             跑全部
#   ./tests/run_all.sh pytest      只跑接口测试
#   ./tests/run_all.sh newman      只跑 Postman 集合
#   ./tests/run_all.sh jmeter      只跑压测
#
# 前置：服务已启动（./deploy/run.sh -d）
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

REPORT_DIR="$PROJECT_DIR/tests/reports"
mkdir -p "$REPORT_DIR"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

info() { printf '\033[32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }
step() { printf '\n\033[36m=== %s ===\033[0m\n' "$*"; }

WHAT="${1:-all}"

# ---------- 前置检查 ----------
if ! curl -s -m 5 "$BASE_URL/api/health" >/dev/null 2>&1; then
  warn "服务不可用：$BASE_URL"
  echo "请先启动：./deploy/run.sh -d"
  exit 1
fi
info "服务正常：$BASE_URL"

# ---------- 1. pytest 接口测试 ----------
if [ "$WHAT" = "all" ] || [ "$WHAT" = "pytest" ]; then
  step "pytest 接口测试"
  .venv/bin/python -m pytest tests/ -v \
    --junitxml="$REPORT_DIR/pytest.xml" \
    2>&1 | tee "$REPORT_DIR/pytest.log" | tail -25
  info "报告：$REPORT_DIR/pytest.xml"
fi

# ---------- 2. Postman 集合（newman） ----------
if [ "$WHAT" = "all" ] || [ "$WHAT" = "newman" ]; then
  step "Postman 集合（newman）"
  if ! command -v newman >/dev/null 2>&1; then
    warn "未安装 newman，跳过。安装：npm i -g newman"
  else
    # 集合若不存在则先生成
    [ -f tests/postman_collection.json ] || .venv/bin/python tests/build_postman.py
    newman run tests/postman_collection.json \
      --env-var "baseUrl=$BASE_URL" \
      --reporters cli,json \
      --reporter-json-export "$REPORT_DIR/newman.json" \
      --timeout-request 300000 \
      2>&1 | tee "$REPORT_DIR/newman.log" | tail -30
    info "报告：$REPORT_DIR/newman.json"
  fi
fi

# ---------- 3. JMeter 压测 ----------
if [ "$WHAT" = "all" ] || [ "$WHAT" = "jmeter" ]; then
  step "JMeter 压测"
  if ! command -v jmeter >/dev/null 2>&1; then
    warn "未找到 jmeter，跳过。请把 JMeter 的 bin 目录加入 PATH"
  else
    # 读接口：并发 20，每线程 20 次
    info "读接口压测（20 并发 × 20 次）"
    jmeter -n -t jmeter/read_api.jmx \
      -Jhost=127.0.0.1 -Jport=8000 \
      -Jthreads=20 -Jrampup=10 -Jloops=20 \
      -l "$REPORT_DIR/jmeter_read.jtl" \
      -e -o "$REPORT_DIR/jmeter_read_report" \
      > "$REPORT_DIR/jmeter_read.log" 2>&1
    info "读接口报告：$REPORT_DIR/jmeter_read_report/index.html"

    # 问答接口：并发 5，每线程 2 次（含大模型，低并发长响应）
    info "问答接口压测（5 并发 × 2 次）"
    jmeter -n -t jmeter/chat_api.jmx \
      -Jhost=127.0.0.1 -Jport=8000 \
      -Jthreads=5 -Jrampup=10 -Jloops=2 \
      -Jcsvfile="$PROJECT_DIR/jmeter/questions.csv" \
      -l "$REPORT_DIR/jmeter_chat.jtl" \
      -e -o "$REPORT_DIR/jmeter_chat_report" \
      > "$REPORT_DIR/jmeter_chat.log" 2>&1
    info "问答报告：$REPORT_DIR/jmeter_chat_report/index.html"
  fi
fi

step "完成"
ls -la "$REPORT_DIR" 2>/dev/null | tail -12
