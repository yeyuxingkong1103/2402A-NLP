#!/usr/bin/env bash
# ============================================================
# 工单编号：人工智能NLP-RAG-LightRAG优化
# 结束脚本：停止 LightRAG 相关后台进程 / 清理运行缓存
# ============================================================

echo "停止可能的后台 LightRAG/评估进程..."
pkill -f "lightrag_build.py" 2>/dev/null || true
pkill -f "compare_eval.py" 2>/dev/null || true
pkill -f "main.py" 2>/dev/null || true

echo "可选清理（如需重建图谱，请取消注释）："
# rm -rf "../研发/src/lightrag_data"      # 图谱存储
# rm -rf "../研发/src/lightrag_trial"     # 小试目录
# rm -f  "../研发/src/results/baseline_chunk_vectors.npy"   # 向量缓存

echo "已结束。"
