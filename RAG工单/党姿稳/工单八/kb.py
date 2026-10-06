# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
知识库装配：加载（或构建）ccf_competition 语料块与向量缓存。
"""
import os
import json
import numpy as np

from config import CHUNKS_FILE, VECTORS_FILE
from ccf_parser import build_chunks


def load_chunks():
    """加载语料块，不存在则解析 PDF 构建"""
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_chunks()


def count_chunks():
    return len(load_chunks())
