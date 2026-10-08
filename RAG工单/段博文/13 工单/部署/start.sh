#!/bin/bash
# 工单13：RAG性能瓶颈识别与优化 - 运行脚本

echo "========================================"
echo "RAG性能瓶颈识别与优化 - 基准测试"
echo "========================================"

cd "$(dirname "$0")/../研发/src"

echo "前提：Milvus 已启动且 rag_legal 集合有数据（persona_rag 项目已入库）"
echo ""

echo "运行基线 vs 优化 基准测试（含 cProfile）..."
python benchmark.py --profile --repeat 2

echo ""
echo "分阶段耗时汇总..."
python stage_summary.py

echo ""
echo "生成图表与报告..."
cd ../../测试
python plot_results.py

echo ""
echo "完成！查看 研发/src/results/ 与 测试/ 目录"
