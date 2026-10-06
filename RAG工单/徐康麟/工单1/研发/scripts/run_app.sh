#!/usr/bin/env bash
# =============================================================================
# 启动 Streamlit 网页界面
# =============================================================================
# 目录结构（按交付阶段分类）：
#     工单1/
#     ├─ 设计/     文档与规格说明
#     ├─ 研发/     源码 app/ + 脚本 scripts/   <- 本脚本所在
#     ├─ 测试/     测试用例
#     ├─ 优化/     评估结果
#     └─ 部署/     环境配置与启动入口
#
# 用法：
#     bash 研发/scripts/run_app.sh                # 默认 8501 端口
#     PORT=8600 bash 研发/scripts/run_app.sh
#
# 前置条件（缺一不可）：
#     1) conda 环境 gao6gongdan 已就绪（见 部署/环境配置/requirements.txt）
#     2) 已建好索引：python 研发/scripts/build_index.py
#     3) 若要用 LLM 生成（否则自动降级为抽取式回答）：
#        bash 研发/scripts/run_vllm.sh   # 另开一个终端
# =============================================================================
set -euo pipefail

PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"

# 本脚本位于 <root>/研发/scripts/：向上一级是「研发」（源码根），再上一级是项目根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SOURCE_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 让 `import app.xxx` 生效（不依赖 pip install -e）
export PYTHONPATH="${SOURCE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8

echo "=============================================="
echo " 启动 RAG 问答系统 Web 界面"
echo "   项目根目录: ${PROJECT_ROOT}"
echo "   源码目录  : ${SOURCE_ROOT}"
echo "   监听地址  : http://${ADDRESS}:${PORT}"
echo "=============================================="
echo

# 索引自检：缺失时给出明确提示，而不是让界面起来后再报错
if [ ! -f "data/index/rag.sqlite3" ] && [ ! -f "data/index/bm25_index.pkl" ]; then
    echo "[提示] 未检测到索引文件，请先运行："
    echo "       python 研发/scripts/build_index.py"
    echo
fi

exec streamlit run "${SOURCE_ROOT}/app/ui/streamlit_app.py" \
    --server.port "${PORT}" \
    --server.address "${ADDRESS}" \
    --server.headless true \
    --browser.gatherUsageStats false
