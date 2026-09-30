"""集成测试（integration）：真实加载 BGE-M3 / BGE-Reranker-v2-M3。

模型加载一次（module 级 fixture 复用），验证维度、归一化、语义区分与重排合理性。
"""
import math

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    from src.rag.embedder import get_embedder
    return get_embedder()


@pytest.fixture(scope="module")
def reranker():
    from src.rag.reranker import get_reranker
    return get_reranker()


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def test_bge_m3_dimension_1024(embedder):
    vec = embedder.encode_query("我最近总是控制不住地担忧未来的事情")
    assert len(vec) == 1024


def test_bge_m3_normalized(embedder):
    vec = embedder.encode_query("测试归一化")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3       # 归一化向量，配合 COSINE 检索


def test_bge_m3_batch_encode(embedder):
    vectors = embedder.encode(["第一句", "第二句", "第三句"])
    assert len(vectors) == 3 and all(len(v) == 1024 for v in vectors)


def test_bge_m3_semantic_separation(embedder):
    """心理话题之间相似度应显著高于与无关财经话题的相似度。"""
    anx = embedder.encode_query("我感到很焦虑，晚上翻来覆去睡不着")
    ins = embedder.encode_query("失眠的时候可以做腹式呼吸放松训练")
    stock = embedder.encode_query("今天上证指数收盘上涨百分之二")
    assert _cosine(anx, ins) > _cosine(anx, stock) + 0.05


def test_reranker_ranks_relevant_first(reranker):
    candidates = [
        {"doc_id": 1, "text": "企业记账应遵循权责发生制，在收入实现时确认。"},
        {"doc_id": 2, "text": "缓解考前焦虑可以先做 4-7-8 呼吸练习，让身体先平静下来。"},
        {"doc_id": 3, "text": "本周气温骤降，请注意添衣保暖。"},
    ]
    ranked = reranker.rerank("考试前特别紧张怎么办？", candidates, top_n=3)
    assert ranked[0]["doc_id"] == 2     # 相关片段排第一
    assert len(ranked) == 3


def test_reranker_score_in_unit_interval(reranker):
    ranked = reranker.rerank("如何缓解焦虑", [
        {"text": "深呼吸与正念冥想有助于缓解焦虑情绪。"},
        {"text": "番茄炒蛋的做法很简单。"},
    ])
    assert all(0.0 < h["rerank_score"] < 1.0 for h in ranked)
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]
