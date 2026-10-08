#!/bin/bash
# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
# ============================================================
# 一键执行完整微调流程（数据生成 → 训练 → 前后评估）
# ============================================================
set -e

cd "$(dirname "$0")/../研发/src"

# 激活 conda 环境（如存在）
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rag_embed_ft 2>/dev/null || true
fi

# DeepSeek API（请提前 export 或在此填入）
export deepseek_api_key1="${deepseek_api_key1:-请填入你的KEY}"
export deepseek_base_url1="${deepseek_base_url1:-https://api.deepseek.com}"

echo "===== 步骤 1/4：段落采样 ====="
[ -f dataset/passages.jsonl ] || python gen_passages.py

echo "===== 步骤 2/4：生成 QA 对 ====="
[ -f dataset/qa_pairs.jsonl ] || python gen_qa_pairs.py

echo "===== 步骤 3/4：数据划分 ====="
python build_dataset.py

echo "===== 步骤 4/4：微调 + 前后评估 ====="
python finetune.py

echo "全部完成！微调模型位于 研发/src/models/bge-base-finance-ft"
