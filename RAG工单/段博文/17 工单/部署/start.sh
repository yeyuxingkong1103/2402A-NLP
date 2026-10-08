#!/bin/bash
set -e
ENV_NAME="rag_wo17"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/研发/src"
LOG_DIR="${SCRIPT_DIR}/results"
mkdir -p "${LOG_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "===== 压测（基线 vs 优化后）====="
(cd "${SRC_DIR}" && python benchmark.py) | tee "${LOG_DIR}/benchmark_run.log"
echo "===== 生成图表与报告 ====="
(cd "${PROJECT_DIR}/设计" && python gen_all.py)
(cd "${PROJECT_DIR}/测试" && python batch_shots.py) || echo "截图需Playwright"
echo "===== 完成 ====="
