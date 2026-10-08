#!/bin/bash
# 工单14：启动低质量PDF解析与抽取式问答流水线（Linux）
set -e

ENV_NAME="rag_wo14"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/研发/src"
LOG_DIR="${SCRIPT_DIR}/results"
PID_FILE="${SCRIPT_DIR}/results/pipeline.pid"

mkdir -p "${LOG_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "===== 1. （可选）生成模拟图片型工业 PDF ====="
if [ ! -f "${PROJECT_DIR}/研发/CN100342976C.pdf" ]; then
    (cd "${PROJECT_DIR}/研发" && python gen_simulated_pdf.py)
fi

echo "===== 2. DeepDoc 风格解析（渲染→OCR→版面分析→段落重建） ====="
echo "  如需启用真实 easyocr，请先下载模型后设置 USE_OCR=1"
(cd "${SRC_DIR}" && USE_OCR="${USE_OCR:-0}" python deepdoc_ocr_parser.py \
    "${PROJECT_DIR}/研发/CN100342976C.pdf") | tee "${LOG_DIR}/parser_run.log"

echo "===== 3. 本地抽取式问答（bge-m3 检索 + 规则抽取，零API费用） ====="
(cd "${SRC_DIR}" && python rag_qa.py) | tee "${LOG_DIR}/qa_run.log"

echo "===== 4. 生成测试图表与HTML报告 ====="
(cd "${PROJECT_DIR}/测试" && python plot_results.py)

echo "===== 全部完成 ====="
echo "结果文件: ${PROJECT_DIR}/研发/qa_results.json"
echo "HTML报告: ${PROJECT_DIR}/测试/解析与问答测试报告.html"
