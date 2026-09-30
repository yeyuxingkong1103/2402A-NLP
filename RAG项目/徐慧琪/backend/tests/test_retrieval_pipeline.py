# 编排测试分两层：合并规则是纯函数，可离线断言；
# 端到端检索要 MySQL + Milvus + 两个模型同时在线，离线时跳过。
import pytest

from app.db.milvus import get_client
from app.retrieval.pipeline import (
    RECALL_TOPK, RetrievalResult, merge_blocks, retrieve,
)


def _services_available() -> bool:
    try:
        get_client().list_collections()
        from app.db.mysql import connect
        connect().close()
        return True
    except Exception:
        return False


def test_recall_topk_matches_spec():
    # 技术方案 5.2 的起始值：两路各 top50
    assert RECALL_TOPK == 50


def test_merge_puts_exact_blocks_first_in_question_order():
    exact = [{"article_no": 584, "source": "exact", "text": "x"},
             {"article_no": 577, "source": "exact", "text": "y"}]
    ranked = [{"article_no": 1, "source": "vector", "text": "z"}]
    merged = merge_blocks(exact, ranked)
    assert [b["article_no"] for b in merged] == [584, 577, 1]


def test_merge_dedupes_by_article_no_keeping_exact():
    # 精确块与向量块撞同一条时留精确块：它有 MySQL 原文与效力状态，
    # 校验层第②关直接用它，不必再查一次
    exact = [{"article_no": 584, "source": "exact", "text": "原文"}]
    ranked = [{"article_no": 584, "source": "vector", "text": "召回"}]
    merged = merge_blocks(exact, ranked)
    assert len(merged) == 1
    assert merged[0]["source"] == "exact"


def test_merge_without_exact_returns_ranked_untouched():
    ranked = [{"article_no": 1, "source": "vector", "text": "z"}]
    assert merge_blocks([], ranked) == ranked


def test_merge_respects_top_k_cut():
    ranked = [{"article_no": i, "source": "vector", "text": str(i)} for i in range(10)]
    assert len(merge_blocks([], ranked, top_k=5)) == 5


@pytest.mark.skipif(not _services_available(), reason="MySQL/Milvus 不在线")
def test_retrieve_by_article_number_puts_exact_block_first():
    """问"第五百八十四条"时该条必须排第一——这是 AC-2 在编排层的体现。"""
    result = retrieve("第五百八十四条规定了什么", **_deps())
    assert isinstance(result, RetrievalResult)
    assert result.blocks[0]["article_no"] == 584
    assert result.blocks[0]["source"] == "exact"
    assert 584 in result.exact_nos


@pytest.mark.skipif(not _services_available(), reason="MySQL/Milvus 不在线")
def test_retrieve_semantic_question_returns_parents_with_text():
    """语义问题走向量路：结果里必须有父块且带原文，否则生成层无米下锅。"""
    result = retrieve("租房押金不退怎么办", **_deps())
    assert result.blocks, "语义问句应召回至少一个父块"
    for block in result.blocks:
        assert block["chunk_type"] == "father"
        assert block["text"].strip()
    # 子块一并带出，供校验层数款/项号（设计文档 4.3 第③关）
    assert result.chunks
    # 两个字段不能对调：对调后 recalled_blocks 变成原始子块、chunks 变成父块，
    # 而 Task 8 引用校验第③关要靠 chunks 的 paragraph_no / item_no 判"第三款
    # 是否真的存在"——拿到父块会把真引用判成幻觉；Task 12 又会拿
    # recalled_blocks 算 Recall@k，混进子块口径就错。all() 对空列表恒真，
    # 故先钉非空，避免 pipeline 哪天漏填 recalled_blocks 这条断言白过
    assert result.recalled_blocks, "回填后应有未精排的父块全集"
    assert all(b["chunk_type"] == "father" for b in result.recalled_blocks)
    assert all(b["chunk_type"] in ("paragraph", "item") for b in result.chunks)


@pytest.mark.skipif(not _services_available(), reason="MySQL/Milvus 不在线")
def test_retrieve_out_of_range_article_falls_back_to_vector_path():
    """第9999条不存在，不该置顶也不该抛错，应正常退回向量路。"""
    result = retrieve("第9999条规定了什么", **_deps())
    assert result.exact_nos == []
    assert all(b["source"] == "vector" for b in result.blocks)


# 端到端依赖只造一次：bge-m3 加载要几十秒，三条集成测各加载一次会让
# 这个文件跑成分钟级，久了就没人愿意跑它
_DEPS: dict = {}


def _deps() -> dict:
    """构造端到端依赖：真模型、真连接。只在集成测里用。"""
    if not _DEPS:
        from app.db.mysql import connect
        from app.ingest.embed import load_model
        from app.retrieval.rerank import load_reranker
        _DEPS.update({"conn": connect(), "client": get_client(),
                      "encoder": load_model(), "reranker": load_reranker()})
    return _DEPS
