#!/bin/bash
# ================================================================
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# scripts/stop_v3.sh —— 停止 FastAPI(8003) + Streamlit(8503)
# 用法：bash scripts/stop_v3.sh
# ================================================================

echo "================================================"
echo "工单三：停止服务"
echo "工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化"
echo "================================================"

# ================= 1. 停止 FastAPI (8003) =================
echo "[STEP 1] 停止 FastAPI (端口 8003)..."
PIDS=$(lsof -t -i :8003 2>/dev/null)
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        kill "$PID" 2>/dev/null
        echo "[OK] 已停止 PID $PID (8003)"
    done
else
    echo "[INFO] 端口 8003 无服务运行"
fi

# ================= 2. 停止 Streamlit (8503) =================
echo ""
echo "[STEP 2] 停止 Streamlit (端口 8503)..."
PIDS=$(lsof -t -i :8503 2>/dev/null)
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        kill "$PID" 2>/dev/null
        echo "[OK] 已停止 PID $PID (8503)"
    done
else
    echo "[INFO] 端口 8503 无服务运行"
fi

# ================= 3. 清理临时文件 =================
# echo ""
# echo "[STEP 3] 清理临时文件..."
# rm -f /tmp/api_v3.log /tmp/streamlit_v3.log 2>/dev/null

echo ""
echo "================================================"
echo "[DONE] 服务已停止"
echo "================================================"
