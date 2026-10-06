# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
配置：数据集路径、6 个图文混合测试问题、融合策略参数。
"""
import os

ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单1\14-17附件"
DATA_ZIP = os.path.join(ATTACH_DIR, "original_problems.zip")
DOC_DIR = os.path.join(ATTACH_DIR, "original_problems", "documents")

TEST_PDF = "CN100347506C.pdf"

# 6 个测试问题（前 2 个纯文本，后 4 个图文强关联）
TEST_QUESTIONS = [
    {"question": "根据专利文本，本发明主要涉及哪种物料的分配装置？",
     "answer": "块状散料", "kind": "text"},
    {"question": "根据专利文本，本发明的分散装置包含以下哪个组件？",
     "answer": "链条", "kind": "text"},
    {"question": "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？",
     "answer": "位于编号12的部件之内", "kind": "image"},
    {"question": "在文件中第11页图3中，编号14的部件位于整个装置的哪个位置？",
     "answer": "顶部", "kind": "image"},
    {"question": "根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？",
     "answer": "部件13", "kind": "image"},
    {"question": "在文件中第11页图3的装置中，如果需要调整链条的位置，需要操作哪个部件?",
     "answer": "部件11", "kind": "image"},
]

# 视觉引用关键词（用于查询理解：识别“图3”“第11页图示”等）
VISUAL_REF_HINTS = ["图3", "图 3", "图示", "第11页", "部件", "编号"]

# 混合检索融合策略：weighted（加权）/ rrf（倒数排名融合）
FUSION_METHOD = "rrf"
IMAGE_QUERY_WEIGHT = 0.6   # 图像增强查询在加权融合中的权重
TOP_K = 6
RERANK_ENABLE = True       # 跨模态重排开关
