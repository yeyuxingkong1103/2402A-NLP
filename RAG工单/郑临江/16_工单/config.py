# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单
配置：基础 VLM、LoRA 微调参数、图像预处理、评估用工业术语。
"""
import os

ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单1\14-17附件"
DATA_ZIP = os.path.join(ATTACH_DIR, "original_problems.zip")
QUESTIONS_IN_ZIP = "original_problems/questions.jsonl"

# 基础视觉语言模型
BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# LoRA 微调参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
MAX_LENGTH = 2048

# 图像预处理（Qwen-VL 建议最大像素数）
IMAGE_MAX_PIXELS = 1474560   # 约 768*768*2.5

# 数据输出目录
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_JSONL = os.path.join(OUT_DIR, "train.jsonl")

# 工业领域术语（用于专业术语准确性评估）
INDUSTRIAL_TERMS = ["淬火", "公差配合", "散料", "除尘", "静电", "配气",
                    "铸造", "锻造", "热处理", "圆锥", "外壳直径", "带孔盘"]
