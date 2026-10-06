#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单
# LLaMA-Factory 微调启动命令

# 1. 准备数据
python data_convert.py

# 2. 注册数据集（在 LLaMA-Factory 目录下编辑 data/dataset_info.json，
#    新增 imdr_vlm 指向 data/train.jsonl 的 image/question/answer 三字段）

# 3. 启动 LoRA 微调
llamafactory-cli train train_config.yaml

# 4. 导出合并后的模型（可选）
llamafactory-cli export --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --adapter_name_or_path ./output/vlm_lora --template qwen_vl \
  --finetuning_type lora --export_dir ./output/vlm_merged
