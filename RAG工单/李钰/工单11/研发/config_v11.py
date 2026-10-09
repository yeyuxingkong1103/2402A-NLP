# -*- coding: utf-8 -*-
"""
训练配置 V11 - Embedding 微调
工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 基础模型 ============
BASE_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# 可选: "BAAI/bge-base-en-v1.5", "BAAI/bge-small-zh"

# ============ 训练参数 ============
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("TRAIN_LR", "2e-5"))
MAX_LENGTH = int(os.environ.get("TRAIN_MAX_LEN", "256"))
WARMUP_RATIO = 0.1

# ============ 损失函数 ============
LOSS_TYPE = os.environ.get("LOSS_TYPE", "triplet")
# triplet / contrastive / cosine / matryoshka

# ============ 数据 ============
TRAIN_DATA = os.environ.get("TRAIN_DATA",
                            os.path.join(BASE_DIR, "preset_training_data.json"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            os.path.join(BASE_DIR, "fine_tuned_model"))

# ============ 数据生成参数 ============
NUM_QUERY_PAIRS = 200     # 生成多少问答对
NUM_TRIPLETS = 500        # 生成多少三元组
NEG_PER_POS = 4           # 每个正例配多少负例

# ============ 评估 ============
TEST_DATA = os.path.join(BASE_DIR, "preset_training_data.json")
TOP_K_EVAL = [1, 3, 5, 10]

# ============ 设备 ============
DEVICE = os.environ.get("DEVICE", "cpu")  # cpu / cuda
USE_AMP = True  # 自动混合精度 (GPU 时)

os.makedirs(OUTPUT_DIR, exist_ok=True)
