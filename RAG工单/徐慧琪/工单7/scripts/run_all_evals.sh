#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# 串行跑完四套评估（必须串行：GPU 显存 8G + Qdrant 本地模式单进程独占）
set -x
cd "$(dirname "$0")/.."
python evaluate_retrieval.py --strategies hybrid,vector,fulltext --top-k 5 > logs/eval_retrieval_full.log 2>&1
echo "RETRIEVAL_EXIT=$?" >> logs/eval_chain.status
python evaluate_compare.py > logs/eval_compare.log 2>&1
echo "COMPARE_EXIT=$?" >> logs/eval_chain.status
python evaluate_ragas.py --limit 16 > logs/eval_ragas.log 2>&1
echo "RAGAS_EXIT=$?" >> logs/eval_chain.status
python benchmark_load.py --levels 10,50,100 --per-level 20 > logs/benchmark.log 2>&1
echo "BENCH_EXIT=$?" >> logs/eval_chain.status
echo "ALL_DONE" >> logs/eval_chain.status
