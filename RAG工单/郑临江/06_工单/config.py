# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
配置：数据路径、问题列表、混合检索参数。
"""
import os

ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单\附件"
PDF1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")
PDF2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")

QUESTIONS = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# 向量召回与重排参数
RECALL_TOP_K = 10      # 向量召回候选数
RERANK_TOP_K = 3       # 重排后返回数

# 混合检索权重（向量 : 全文）
VECTOR_WEIGHT = 0.6
FULLTEXT_WEIGHT = 0.4

# 可选语义嵌入模型（bge / m3e）
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
