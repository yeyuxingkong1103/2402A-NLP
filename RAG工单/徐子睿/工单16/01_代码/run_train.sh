#!/usr/bin/env bash
# 工单 16 · 专用 VLM LoRA 微调启动命令（GPU 实例：AutoDL / Linux）
# 前置：pip install llamafactory  (或 git clone LLaMA-Factory && pip install -e .)
set -e

export HF_ENDPOINT=https://hf-mirror.com          # 国内实例走镜像
export CUDA_VISIBLE_DEVICES=0

# 1) 生成微调数据（在 CPU 或 GPU 上均可）
python convert_to_vlm_dataset.py \
  --documents /root/autodl-tmp/original_problems/documents \
  --questions /root/autodl-tmp/original_problems/questions.jsonl \
  --out /root/autodl-tmp/qwen25vl_imdr/vlm_dataset.jsonl \
  --limit 5000

# 2) 在 LLaMA-Factory/data/dataset_info.json 注册数据集 imdr_vlm：
#    {
#      "imdr_vlm": {
#        "file_name": "vlm_dataset.jsonl",
#        "formatting": "sharegpt",
#        "columns": {"messages": "messages", "images": "images"}
#      }
#    }

# 3) 启动 LoRA 微调
llamafactory-cli train lora_config.yaml

# 4) 合并 LoRA 权重（导出独立模型）
llamafactory-cli export lora_export.yaml

# 5) 评估（基线 vs 微调；见 eval_industrial.py）
python eval_industrial.py \
  --pred /root/autodl-tmp/qwen25vl_imdr/eval_pred_ft.jsonl \
  --ref  /root/autodl-tmp/qwen25vl_imdr/eval_ref.jsonl
