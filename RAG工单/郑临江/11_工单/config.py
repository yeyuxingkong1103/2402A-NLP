# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
配置：数据路径、基础模型、损失函数、训练参数。
"""
import os

ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单\附件"
PDF1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")
PDF2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")

# 基础模型：BAAI 高性能文本嵌入模型
BASE_MODEL = "BAAI/bge-base-en-v1.5"

# 损失函数可选：triplet / contrastive / cosine / matryoshka
LOSS_TYPE = "triplet"

# 训练参数
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EVAL_STEPS = 200

# 数据集与模型输出
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "output_model")

# 微调前/后评估的检索问题
EVAL_QUESTIONS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉力源信息技术股份有限公司本次发行股数是多少？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
]
