# -*- coding: utf-8 -*-
# vlm_lora_sft.yaml 工单16 VLM LoRA微调配置 LLaMA‑Factory
model_name_or_path: "/mnt/d/models/Qwen‑VL‑Chat"
model_type: qwen_vl
visual_inputs: true

# 数据集
dataset: llama_factory_train
dataset_dir: ./dataset
template: qwen_vl

# LoRA参数
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all
lora_dropout: 0.05

# 训练超参
stage: sft
do_train: true
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e‑4
lr_scheduler_type: cosine
warmup_ratio: 0.05

# 图像预处理（必须和基座VLM保持一致）
image_resolution: 768

# 输出保存
output_dir: ./vlm_lora_output
save_steps: 100
logging_steps: 10
overwrite_output_dir: true

# 显存优化
gradient_checkpointing: true
fp16: true
