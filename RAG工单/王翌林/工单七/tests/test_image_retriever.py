# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_image_retriever.py —— 工单四图像路由与检索器单测（Mock 存储/嵌入，不依赖模型）
"""
from src.image_parser.image_query_router import IMAGE_KEYWORDS, route_query
from src.image_parser.image_retriever import ImageRetriever  # 工单四：被测模块

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"


class MockStore:
    """工单四：Mock Milvus 存储（固定命中 org chart 一条）"""

    def search_text(self, qvec, top_k=5, doc_ids=None, use_clip=False):
        return [{
            "doc_id": "招股说明书2", "image_id": "img_008", "page": 39,
            "path": "data/images/招股说明书2/page_039_draw_1.png",
            "caption": "武汉力源信息技术股份有限公司组织结构图",
            "ocr_text": "总经理 销售部 大客户销售部 珠海销售处 深圳销售处",
            "vqa_text": "销售部由4个部门构成；大客户销售部下设6个销售处",
            "metadata": {"work_order": WORK_ORDER},
            "score": 0.62,
        }][:top_k]


class MockEmbedder:
    def embed_text(self, text):
        return [0.0] * 1024


# ---------------- 工单四：图像感知路由 ----------------
def test_router_image_first_strong():
    """工单四：id5/id6 类问题命中强信号 → image_first"""
    r1 = route_query("武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？")
    assert r1["mode"] == "image_first" and "组织结构" in r1["hits"]
    r2 = route_query("从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？")
    assert r2["mode"] == "image_first" and "增长率" in r2["hits"]


def test_router_hybrid_fallback():
    """工单四：纯文本/表格问题不触发图像优先"""
    assert route_query("武汉兴图新科2018年营业收入是多少？")["mode"] == "hybrid"
    assert route_query("发行人本次发行股数是多少？")["mode"] == "hybrid"


def test_router_keywords_coverage():
    """工单四：关键词表覆盖工单要求（图/组织结构/增长图/图表/示意图/销售部/增长率）"""
    for kw in ("图", "组织结构", "增长图", "图表", "示意图", "销售部", "增长率"):
        assert kw in IMAGE_KEYWORDS


# ---------------- 工单四：图像检索器 ----------------
def test_retriever_flow_and_boost():
    """工单四：检索流程与关键词加权重排（命中 ocr/vqa 的记录分更高）"""
    r = ImageRetriever(store=MockStore(), embedder=MockEmbedder())
    hits = r.retrieve("组织结构图 销售部", top_k=5)
    assert hits and hits[0]["image_id"] == "img_008"
    h = hits[0]
    # 工单四：返回字段齐全
    for k in ("path", "caption", "ocr_text", "vqa_text", "image_id",
              "page", "doc_id", "final_score"):
        assert k in h
    # 工单四：关键词命中 → final_score = score + boost*kw_hits > 原始 score
    assert h["kw_hits"] >= 1 and h["final_score"] > h["score"]


def test_retriever_doc_filter_passthrough():
    """工单四：doc_ids 过滤参数透传至 store 层"""

    class FilterStore(MockStore):
        def __init__(self):
            self.seen = None

        def search_text(self, qvec, top_k=5, doc_ids=None, use_clip=False):
            self.seen = doc_ids
            return super().search_text(qvec, top_k, doc_ids, use_clip)

    fs = FilterStore()
    r = ImageRetriever(store=fs, embedder=MockEmbedder())
    r.retrieve("组织结构", top_k=3, doc_ids=["招股说明书2"])
    assert fs.seen == ["招股说明书2"]
