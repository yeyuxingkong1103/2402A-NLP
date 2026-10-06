# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_image_store.py —— 工单四 rag_images 存储单测（Milvus Lite 临时库，不依赖模型）
"""
from pathlib import Path

import pytest

from src.image_parser.image_store import ImageStore  # 工单四：被测模块

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"


@pytest.fixture()
def store(tmp_path):
    """工单四：Milvus Lite 临时实例（force_lite 隔离远程，独立于工单三 collection）"""
    s = ImageStore(lite_path=str(tmp_path / "test_images.db"), force_lite=True)
    s.ensure_collection()
    return s


def _rec(doc_id="招股说明书2", image_id="img_008", page=39):
    """工单四：构造样例记录（1024 维假向量）"""
    return {"doc_id": doc_id, "image_id": image_id, "page": page,
            "path": f"data/images/{doc_id}/page_{page:03d}_draw_1.png",
            "caption": "组织结构图：总经理下设销售部、研发部等部门",
            "ocr_text": "销售部 大客户销售部 珠海销售处 深圳销售处",
            "vqa_text": "销售部由4个部门构成；大客户销售部下设6个销售处",
            "embedding": [0.01] * 1024, "clip_embedding": [0.02] * 512,
            "metadata": {"work_order": WORK_ORDER, "extract_source": "L2_vector"}}


def test_insert_and_count(store):
    """工单四：插入与统计"""
    n = store.insert([_rec(), _rec(image_id="img_011", page=72)])
    assert n == 2
    assert store.count() == 2


def test_search_text(store):
    """工单四：向量检索返回字段完整（path/caption/ocr/vqa）"""
    store.insert([_rec(), _rec(image_id="img_011", page=72)])
    hits = store.search_text([0.01] * 1024, top_k=2)
    assert len(hits) == 2
    h = hits[0]
    for k in ("doc_id", "image_id", "page", "path", "caption",
              "ocr_text", "vqa_text", "score"):
        assert k in h
    assert h["image_id"] in ("img_008", "img_011")


def test_search_with_doc_filter(store):
    """工单四：doc_id 过滤只命中指定文档"""
    store.insert([_rec(), _rec(doc_id="招股说明书1", image_id="img_001", page=1)])
    hits = store.search_text([0.01] * 1024, top_k=5, doc_ids=["招股说明书1"])
    assert len(hits) == 1 and hits[0]["doc_id"] == "招股说明书1"


def test_delete_by_doc_id(store):
    """工单四：按 doc_id 删除后计数归零（另一文档保留）"""
    store.insert([_rec(), _rec(doc_id="招股说明书1", image_id="img_001", page=1)])
    store.delete_by_doc_id("招股说明书2")
    hits = store.search_text([0.01] * 1024, top_k=5)
    assert len(hits) == 1 and hits[0]["doc_id"] == "招股说明书1"
