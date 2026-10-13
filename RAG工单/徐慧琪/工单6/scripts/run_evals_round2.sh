#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# 第二轮：应用 ARM_QUOTA（每路保底）、证据口径修正、书名号修复后的复评
cd "$(dirname "$0")/.."
python evaluate_retrieval.py --strategies hybrid,vector,fulltext --top-k 5 > logs/eval_retrieval_full.log 2>&1
echo "RETRIEVAL_EXIT=$?" >> logs/eval_chain2.status
python evaluate_compare.py > logs/eval_compare.log 2>&1
echo "COMPARE_EXIT=$?" >> logs/eval_chain2.status
echo "ALL_DONE" >> logs/eval_chain2.status
