#!/bin/bash
# ================================================================
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/stop_v4.sh —— 停止 FastAPI(8004) + Streamlit(8504)
# 用法：bash scripts/stop_v4.sh
# ================================================================

echo "================================================"
echo "工单四：停止服务"
echo "工单编号：人工智能NLP-RAG-图像内容解析及检索优化"
echo "================================================"

# ================= 1. 停止 FastAPI (8004) =================
echo "[STEP 1] 停止 FastAPI (端口 8004)..."
PIDS=$(lsof -t -i :8004 2>/dev/null)
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        kill "$PID" 2>/dev/null
        echo "[OK] 已停止 PID $PID (8004)"
    done
else
    echo "[INFO] 端口 8004 无服务运行"
fi

# ================= 2. 停止 Streamlit (8504) =================
echo ""
echo "[STEP 2] 停止 Streamlit (端口 8504)..."
PIDS=$(lsof -t -i :8504 2>/dev/null)
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        kill "$PID" 2>/dev/null
        echo "[OK] 已停止 PID $PID (8504)"
    done
else
    echo "[INFO] 端口 8504 无服务运行"
fi

echo ""
echo "================================================"
echo "[DONE] 服务已停止"
echo "================================================"
