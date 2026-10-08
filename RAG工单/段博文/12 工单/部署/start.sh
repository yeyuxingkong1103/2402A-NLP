#!/usr/bin/env bash
# ============================================================
# 工单编号：人工智能NLP-RAG-LightRAG优化
# 启动脚本：全流程执行
#   1. PDF 解析  2. 知识图谱构建(增量演示)
#   3. 图谱统计/导出  4. RAG vs LightRAG 对比 + RAGAS 评估
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../研发/src"
cd "$SRC_DIR"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate rag_lightrag

echo "========== [1/4] PDF 解析 =========="
python pdf_ingest.py

echo "========== [2/4] LightRAG 知识图谱构建（增量更新演示） =========="
python lightrag_build.py

echo "========== [3/4] 知识图谱统计与可视化导出 =========="
python export_graph.py

echo "========== [4/4] RAG vs LightRAG 对比评估（16 题 + RAGAS） =========="
python compare_eval.py

echo "全部完成。结果见 研发/src/results 与 测试 目录。"
