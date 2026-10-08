#!/bin/bash
set -e
ENV_NAME="rag_wo16"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/研发/src"
LOG_DIR="${SCRIPT_DIR}/results"
mkdir -p "${LOG_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "===== 1. 数据转换 ====="
(cd "${SRC_DIR}" && python convert_data.py) | tee "${LOG_DIR}/data_convert.log"
echo "===== 2. 训练（CPU模拟/GPU实跑）====="
# GPU环境: llamafactory-cli train lora_qwen_vl_industrial.yaml
(cd "${SRC_DIR}" && python mock_train.py) | tee "${LOG_DIR}/train_run.log"
echo "===== 3. 评估 ====="
(cd "${SRC_DIR}" && python evaluate.py) | tee "${LOG_DIR}/eval_run.log"
echo "===== 4. 生成图表与报告 ====="
(cd "${PROJECT_DIR}/设计" && python gen_all.py)
(cd "${PROJECT_DIR}/测试" && python batch_shots.py) || echo "截图需Playwright"
echo "===== 完成 ====="
