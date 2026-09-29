# Embedding 模型微调指南（预留机制）

本机无 NVIDIA GPU，微调需在算力云/带 GPU 的机器上进行。系统已预留模型切换机制：
**换模型只改 `.env` 一行，代码零改动**。

## 一、微调步骤（GPU 环境）

### 1. 准备训练数据

```jsonl
// data/finetune/train.jsonl —— 每条：query / positive / negatives
{"query": "高血压怎么诊断", "positive": "收缩压≥140mmHg可诊断为高血压", "negative": "糖尿病诊断标准"}
{"query": "每天吃多少盐", "positive": "每日食盐摄入量不超过5克", "negative": "高血压诊断标准"}
```

### 2. 用 llamafactory 微调 BGE-m3

```bash
pip install llamafactory
llamafactory-cli train \
  --model_name_or_path BAAI/bge-m3 \
  --stage sft \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --dataset 你的数据集名 \
  --template default \
  --output_dir models/bge-m3-finetuned \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --bf16
```

> 导出合并权重：`llamafactory-cli export --model_name_or_path BAAI/bge-m3 --adapter_name_or_path models/bge-m3-finetuned --export_dir models/bge-m3-merged`

### 3. 切换系统使用微调模型

```ini
# .env
EMBEDDING_MODEL_PATH=models/bge-m3-merged   # 原为 models/bge-m3
```

重启服务即生效（LazyModelProxy 首次检索时加载新路径）。重排模型同理：`RERANK_MODEL_PATH`。

## 二、验证切换生效

```bash
python scripts/verify_rag.py          # 走一遍上传+检索+对话
python scripts/run_eval.py            # 与基线对比 RAGAS 分数
```

基线锚点：context_recall 1.000 / context_precision 0.968 / faithfulness 0.951（微调后应对比这些数字）。

## 三、注意事项

- 微调后向量维度仍是 1024（BGE-m3 结构不变），Milvus Collection 无需重建
- 若更换为不同维度的模型，需重建 Collection（删除 `role_kb_v2_*` 后重传文档）
- 训练数据建议 ≥ 1 万条同领域问答对，数据量不足时微调可能劣化检索
