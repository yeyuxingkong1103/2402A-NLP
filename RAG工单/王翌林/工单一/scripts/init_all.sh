#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统
# scripts/init_all.sh — 一键初始化：MySQL + Milvus + PDF 入库

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ========== Python 解释器 ==========
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -f "/home/dabaie/code/my_project/.venv/bin/python" ]; then
        PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"
    else
        PYTHON_BIN="$(which python3)"
    fi
fi

echo "============================================================"
echo "🔧 一键初始化 — 工单：人工智能NLP-RAG-基于PDF文档的问答系统"
echo "   Python: $PYTHON_BIN"
echo "============================================================"

# ========== 0. 环境变量 ==========
if [ -f ".env" ]; then
    echo "[0/4] 加载 .env..."
    set -a; source ".env"; set +a
fi

# ========== 1. MySQL ==========
echo ""
echo "[1/4] 初始化 MySQL..."
"$PYTHON_BIN" "$SCRIPT_DIR/init_mysql.py" 2>&1 | tee /tmp/init_mysql.log || {
    echo "  ⚠️  MySQL 初始化失败，查看 /tmp/init_mysql.log"
}

# ========== 2. Milvus ==========
echo ""
echo "[2/4] 初始化 Milvus..."
"$PYTHON_BIN" "$SCRIPT_DIR/init_milvus.py" 2>&1 | tee /tmp/init_milvus.log || {
    echo "  ⚠️  Milvus 初始化失败，查看 /tmp/init_milvus.log"
}

# ========== 3. PDF 入库 ==========
echo ""
echo "[3/4] PDF 入库..."
PDF_DIR="${PDF_DIR:-$PROJECT_ROOT/附件}"
if [ -d "$PDF_DIR" ]; then
    pdf_count=$(find "$PDF_DIR" -name '*.pdf' | wc -l)
    echo "  发现 $pdf_count 个 PDF 文件在 $PDF_DIR"
    if [ "$pdf_count" -gt 0 ]; then
        # 默认入库第一个（通常是招股说明书1.pdf）
        first_pdf=$(find "$PDF_DIR" -name '*.pdf' | head -1)
        echo "  入库: $first_pdf"
        "$PYTHON_BIN" "$SCRIPT_DIR/ingest_pdf.py" --pdf "$first_pdf" 2>&1 | tee /tmp/init_ingest.log || {
            echo "  ⚠️  入库失败，查看 /tmp/init_ingest.log"
        }
    fi
else
    echo "  ⚠️  PDF 目录不存在: $PDF_DIR"
fi

# ========== 4. 验证 ==========
echo ""
echo "[4/4] 验证..."
echo "  MySQL tables:"
"$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from src.db import get_session
from src.models import Document, Chunk, QALog
with get_session() as sess:
    for m in [Document, Chunk, QALog]:
        cnt = sess.query(m).count()
        print(f'    {m.__tablename__}: {cnt} rows')
" 2>/dev/null || echo "  ⚠️  MySQL 验证跳过"

echo ""
echo "🎉 初始化完成！"
echo ""
echo "   bash scripts/start.sh    # 启动服务"
echo "   bash scripts/stop.sh     # 停止服务"
echo "   bash scripts/run_tests.sh # 运行测试"
