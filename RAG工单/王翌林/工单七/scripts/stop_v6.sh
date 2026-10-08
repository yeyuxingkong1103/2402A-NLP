#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# scripts/stop_v6.sh —— 工单六 停止 FastAPI v6(8006) + Streamlit v6(8506)
cd "$(dirname "$0")/.."

for name in api_v6 streamlit_v6; do
    pidf="data/pids/${name}.pid"
    if [ -f "$pidf" ]; then
        pid=$(cat "$pidf")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "[v6] 已停止 $name (pid=$pid)"
        fi
        rm -f "$pidf"
    fi
done

# 兜底：按端口清理
pkill -f "uvicorn src.api_v6:app" 2>/dev/null && echo "[v6] 已清理 uvicorn v6" || true
pkill -f "streamlit run app/streamlit_app_v6.py" 2>/dev/null && echo "[v6] 已清理 streamlit v6" || true
echo "[v6] done"
