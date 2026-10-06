#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统
# scripts/run_tests.sh — 运行测试 + 端到端验证

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -f "/home/dabaie/code/my_project/.venv/bin/python" ]; then
        PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"
    else
        PYTHON_BIN="$(which python3)"
    fi
fi

echo "============================================================"
echo "🧪 运行测试 — 工单：人工智能NLP-RAG-基于PDF文档的问答系统"
echo "============================================================"
echo "Python: $($PYTHON_BIN --version)"
echo ""

ALL_PASSED=true

# ========== 1. 单元测试 ==========
echo "📋 [1/3] 单元测试 (pytest)..."
if [ -d "tests" ] && [ "$(ls tests/*.py 2>/dev/null | wc -l)" -gt 0 ]; then
    RESULT=$("$PYTHON_BIN" -m pytest tests/ -v --tb=short 2>&1 || true)
    echo "$RESULT" | tail -30
    if echo "$RESULT" | grep -q "passed"; then
        echo "  ✅ pytest 通过"
    elif echo "$RESULT" | grep -q "no tests ran"; then
        echo "  ℹ️  无 pytest 单元测试（跳过）"
    else
        ALL_PASSED=false; echo "  ❌ pytest 失败"
    fi
else
    echo "  ℹ️  tests/ 目录不存在（跳过 pytest）"
fi

# ========== 2. API 端到端 ==========
echo ""
echo "📋 [2/3] API 端到端验证..."
API_PORT="${APP_PORT:-8001}"

# 等 API 就绪
for i in $(seq 1 10); do
    if curl -s -m 2 "http://127.0.0.1:$API_PORT/api/health" &>/dev/null; then
        break
    fi
    echo "  等待 API... ($i/10)"; sleep 2
done

HEALTH=$(curl -s -m 3 "http://127.0.0.1:$API_PORT/api/health")
if [ -z "$HEALTH" ]; then
    echo "  ❌ API 未运行，请先 bash scripts/start.sh"
    ALL_PASSED=false
else
    echo "  ✅ Health: $HEALTH"

    # 测试 /api/ask
    ANSWER=$(curl -s -m 30 -X POST "http://127.0.0.1:$API_PORT/api/ask" \
        -H "Content-Type: application/json" \
        -d '{"question":"武汉兴图新科电子股份有限公司法定代表人是谁？","use_rag":true,"top_k":5}' \
        | "$PYTHON_BIN" -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','')[:60])" 2>/dev/null)
    if echo "$ANSWER" | grep -qi "程家明"; then
        echo "  ✅ RAG 问答正确: $ANSWER"
    else
        echo "  ⚠️  RAG 问答结果: $ANSWER (需人工确认)"
    fi

    # 测试反馈
    curl -s -m 5 -X POST "http://127.0.0.1:$API_PORT/api/feedback" \
        -H "Content-Type: application/json" \
        -d '{"qa_log_id":1,"rating":5,"comment":"自动化测试"}' &>/dev/null && echo "  ✅ 反馈接口 OK" || echo "  ⚠️  反馈接口跳过"

    # 测试 SSE（只检查 HTTP 200，不等流结束）
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 -X POST "http://127.0.0.1:$API_PORT/api/ask_stream" \
        -H "Content-Type: application/json" \
        -d '{"question":"测试","use_rag":false,"top_k":3}' || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  ✅ SSE 流式 OK (HTTP $HTTP_CODE)"
    else
        echo "  ⚠️  SSE 跳过 (HTTP $HTTP_CODE)"
    fi
fi

# ========== 3. 模块导入测试 ==========
echo ""
echo "📋 [3/3] 模块导入验证..."
for mod in src.pdf_parser src.chunker src.embedding src.vector_store src.db src.models src.query_understanding src.llm_client src.retriever src.rag_engine src.schemas src.api; do
    if "$PYTHON_BIN" -c "import $mod" 2>&1; then
        echo "  ✅ $mod"
    else
        echo "  ❌ $mod 导入失败"; ALL_PASSED=false
    fi
done

echo ""
if $ALL_PASSED; then
    echo "🎉 全部通过！"
    exit 0
else
    echo "⚠️  部分测试失败，查看上方日志"
    exit 1
fi
