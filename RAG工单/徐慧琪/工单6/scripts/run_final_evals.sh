#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# 终轮：基于最终代码重跑 RAGAS（16题，重采答案）与高并发压测
cd "$(dirname "$0")/.."
python evaluate_ragas.py --limit 16 > logs/eval_ragas.log 2>&1
echo "RAGAS_EXIT=$?" >> logs/eval_chain_final.status
python benchmark_load.py --levels 10,50,100 --per-level 20 > logs/benchmark.log 2>&1
echo "BENCH_EXIT=$?" >> logs/eval_chain_final.status
echo "ALL_DONE" >> logs/eval_chain_final.status
