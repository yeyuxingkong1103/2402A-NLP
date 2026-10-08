#!/bin/bash
set -e
ENV_NAME="rag_wo15"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/研发/src"
LOG_DIR="${SCRIPT_DIR}/results"
mkdir -p "${LOG_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "===== 跨模态检索优化问答 ====="
(cd "${SRC_DIR}" && python crossmodal_rag.py) | tee "${LOG_DIR}/qa_run.log"
echo "===== 生成测试图表与报告 ====="
(cd "${PROJECT_DIR}/测试" && python plot_results.py)
echo "===== 生成截图 ====="
(cd "${PROJECT_DIR}/测试" && python take_screenshots.py) || echo "截图需Playwright"
echo "===== 完成 ====="
echo "结果: ${PROJECT_DIR}/研发/qa_results.json"
echo "报告: ${PROJECT_DIR}/测试/跨模态检索测试报告.html"
