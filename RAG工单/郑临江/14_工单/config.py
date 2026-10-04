# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
配置：数据集路径、解析方法、分块策略、测试问题。
"""
import os

# 附件（IMDR 数据集 zip + 目标文档目录）
ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单1\14-17附件"
DATA_ZIP = os.path.join(ATTACH_DIR, "original_problems.zip")
DOC_DIR = os.path.join(ATTACH_DIR, "original_problems", "documents")

# 工单二测试的专利文档
TEST_PDF = "CN100342976C.pdf"

# 页面类型判定阈值：单页字符数低于该值判定为“扫描页”（需 OCR）
PAGE_CHAR_THRESHOLD = 100
# 扫描页占比超过该值，整个文件标记为“扫描型”
SCANNED_RATIO = 0.7

# 解析方法（parser_id）对应的分块策略
PARSER_METHODS = ["paper", "table", "one", "knowledge_graph"]

# 6 个测试问题（question + 标准答案，用于评测精度）
TEST_QUESTIONS = [
    {"question": "根据文本信息，该静电除尘器的发明人是：",
     "answer": "A. P·吉特勒"},
    {"question": "根据文本信息，以下哪个描述符合该静电除尘器的特征？",
     "answer": "管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式。"},
    {"question": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？",
     "answer": "部件4位于部件5的左侧"},
    {"question": "在文件中第7页的图片中，尺寸X1，X2，X3分别代表什么部件的间隔距离？",
     "answer": "配气带孔盘6，6'，6\"之间的间隔距离"},
    {"question": "根据文件中第7页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？",
     "answer": "先经过部件6\"，再经过部件6'"},
    {"question": "根据文件中第7页图示，如果已知外壳直径D，那么h1和h2的尺寸可以用来计算什么？",
     "answer": "确定配气带孔盘6，6'，6\"的位置"},
]

# 检索参数（可调，用于优化精度）
VECTOR_WEIGHT = 0.7       # 向量相似度权重
TEXT_WEIGHT = 0.3         # 关键词权重
TOP_K = 6                 # 召回数
RERANK_ENABLE = True      # 是否启用 ReRank
