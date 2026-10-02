"""离线测试：检索（``app/core/retriever.py``）。

测试目标（工单 9.1 / 5.3）：
1. 索引装载后向量库与 BM25 都有数据；
2. 混合检索返回按分数降序、带来源标记的候选片段；
3. **对工单 10 道固定问题，检查 top_k 候选中是否包含答案所在页**
   （答案页码取自 ``data/eval/golden_qa.jsonl`` 的 ``evidence_pages``）；
4. 页码过滤、空查询、相关性阈值等分支行为正确。

索引缺失时整个模块跳过：
``索引未就绪：请先运行 python scripts/build_index.py``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.retriever import Retriever
from app.models.schemas import GoldenQA, QueryAnalysis, RetrievedChunk

# 目录结构：<root>/测试/tests/<子目录>/xxx.py
#   parents[0]=子目录 parents[1]=tests parents[2]=测试 parents[3]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "研发"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_qa.jsonl"

# --------------------------------------------------------------------------
# 已知召回缺口：在"无语义模型（哈希降级向量）"环境下曾出现证据页召不回的问题。
# 当前索引已换成真实语义模型（bge-small-zh-v1.5），10 题全部召回，故此处为空；
# 若后续环境退化，把题号与原因登记到这里即可让对应用例标记为 xfail。
# --------------------------------------------------------------------------
KNOWN_RECALL_GAPS: dict[int, str] = {}


def _load_golden_items() -> list[GoldenQA]:
    """读取标准答案文件（文件缺失时返回空列表，由 fixture 负责跳过）。"""
    if not GOLDEN_PATH.exists():
        return []
    return [
        GoldenQA(**json.loads(line))
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pytest_generate_tests(metafunc) -> None:
    """按标准答案文件动态参数化工单 10 题，并给已知缺口打上 xfail 标记。"""
    if "golden_item" not in metafunc.fixturenames:
        return
    params = []
    for item in _load_golden_items():
        marks = ()
        if item.id in KNOWN_RECALL_GAPS:
            marks = pytest.mark.xfail(reason=f"题号 {item.id}：{KNOWN_RECALL_GAPS[item.id]}", strict=False)
        params.append(pytest.param(item, id=f"q{item.id}", marks=marks))
    metafunc.parametrize("golden_item", params)


# --------------------------------------------------------------------------
# 索引状态
# --------------------------------------------------------------------------
def test_index_is_loaded(engine, requires_index) -> None:
    """引擎装载索引后，分块、BM25 与向量库三者数量必须一致。"""
    stats = engine.stats()
    assert stats["index_ready"] is True, "引擎未装载到任何分块，索引装载失败"
    assert stats["chunks"] > 0, "引擎中的分块数为 0"
    assert stats["bm25_documents"] == stats["chunks"], (
        f"BM25 文档数({stats['bm25_documents']})与分块数({stats['chunks']})不一致"
    )
    assert stats["vector_count"] == stats["chunks"], (
        f"向量库条数({stats['vector_count']})与分块数({stats['chunks']})不一致，检索会漏召回"
    )

    health = engine.retriever.health()
    assert health["chunks_registered"] == stats["chunks"], "检索器注册的分块数与引擎不一致"
    assert isinstance(health["embedder"], dict) and health["embedder"]["dimension"] > 0, (
        "检索器健康信息中缺少有效的嵌入维度"
    )


# --------------------------------------------------------------------------
# 检索结果形态
# --------------------------------------------------------------------------
def test_retrieve_returns_ranked_scored_chunks(engine, requires_index) -> None:
    """一次正常检索必须返回非空、按分数降序、页码合法的候选片段。"""
    results = engine.retriever.retrieve("注册资本是多少", top_k=5)
    assert results, "检索『注册资本』返回空结果，索引或检索链路异常"
    assert len(results) <= 5, f"top_k=5 时返回了 {len(results)} 条结果"

    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True), f"检索结果未按分数降序排列：{scores}"
    assert scores[0] > 0, "最高分必须为正数"

    for item in results:
        assert isinstance(item, RetrievedChunk), "检索结果元素必须是 RetrievedChunk"
        assert 1 <= item.chunk.page <= 548, f"命中片段页码 {item.chunk.page} 超出 PDF 页码范围"
        assert item.source in {"vector", "bm25", "hybrid", "table"}, f"未知的召回来源：{item.source}"
        assert item.chunk.content.strip(), "命中片段正文为空"


def test_both_vector_and_bm25_contribute(engine, requires_index) -> None:
    """混合检索应有向量与 BM25 两路召回的证据（否则不是混合检索）。"""
    results = engine.retriever.retrieve("军用领域收入占主营业务收入的比重", top_k=10)
    assert results, "混合检索返回空结果"
    assert any(item.vector_score > 0 for item in results), "没有任何片段来自向量召回"
    assert any(item.bm25_score > 0 for item in results), "没有任何片段来自 BM25 召回"
    assert any(item.source == "hybrid" for item in results), "没有任何片段被向量与 BM25 同时命中（融合未生效）"


def test_empty_query_returns_empty(engine, requires_index) -> None:
    """空查询必须返回空列表，不得把整库当成结果。"""
    assert engine.retriever.retrieve("") == [], "空查询应返回空结果"
    assert engine.retriever.retrieve("   ") == [], "纯空白查询应返回空结果"


def test_page_filter_restricts_results(engine, requires_index) -> None:
    """显式页码过滤必须生效：命中的片段都应落在指定页。"""
    analysis = QueryAnalysis(original="注册资本", rewritten="注册资本", page_filter=[22])
    results = engine.retriever.retrieve("注册资本 法定代表人", analysis=analysis, top_k=5)
    assert results, "对第 22 页做页码过滤后没有命中任何片段（该页含公司基本情况表）"
    assert {item.chunk.page for item in results} == {22}, (
        f"页码过滤未生效，返回的页码为 {sorted({item.chunk.page for item in results})}"
    )


# --------------------------------------------------------------------------
# 相关性阈值（判定"该不该回复不清楚"的核心逻辑）
# --------------------------------------------------------------------------
def test_relevance_threshold_on_related_and_unrelated_queries(engine, requires_index) -> None:
    """相关性判定必须基于**原始余弦相似度**：相关问题可信、无关问题不可信。

    说明：融合分数在 ``retrieve`` 内被归一化为 1.0，不能用来做阈值判断；
    实现改用 ``last_top_cosine``（归一化前的最高余弦），本用例正是防止该逻辑退化。
    """
    retriever = engine.retriever
    threshold = retriever.settings.retrieval.min_confidence_cosine
    assert 0.0 < threshold < 1.0, f"原始余弦阈值应在 (0,1) 之间，实际 {threshold}"

    related = retriever.retrieve("武汉兴图新科电子股份有限公司注册资本是多少？", top_k=5)
    assert related, "相关问题没有召回任何片段"
    assert retriever.last_top_cosine >= threshold, (
        f"相关问题的最高原始余弦 {retriever.last_top_cosine:.4f} 低于阈值 {threshold}"
    )
    assert retriever.is_confident(related) is True, "相关问题应判为可信"

    unrelated = retriever.retrieve("红烧肉怎么做才好吃？", top_k=5)
    assert retriever.last_top_cosine < threshold, (
        f"无关问题的最高原始余弦 {retriever.last_top_cosine:.4f} 竟达到阈值 {threshold}，兜底会失效"
    )
    assert retriever.is_confident(unrelated) is False, "无关问题应判为不可信（触发『不清楚』）"


def test_relevance_threshold_empty_results() -> None:
    """空检索结果必须判为不可信（纯单元检查，不需要索引）。"""
    assert Retriever().is_confident([]) is False, "空检索结果必须判为不可信"


# --------------------------------------------------------------------------
# 工单 10 题：证据页召回
# --------------------------------------------------------------------------
def test_golden_question_recalls_evidence_page(engine, golden_item, requires_index) -> None:
    """逐题检查：在候选深度（fusion_top_k）内是否召回标准答案的证据页。

    这是工单 9.1 对 ``test_retriever.py`` 的明确要求：
    "对 10 个问题检索，检查 top_k 是否包含答案所在页"。
    """
    depth = engine.settings.retrieval.fusion_top_k
    analysis = engine.understander.analyze(golden_item.question, [])
    results = engine.retriever.retrieve(
        engine.understander.search_query(analysis), analysis=analysis, top_k=depth
    )
    pages = sorted({result.chunk.page for result in results})
    assert results, f"题号 {golden_item.id} 的检索结果为空"
    assert set(pages) & set(golden_item.evidence_pages), (
        f"题号 {golden_item.id} 的证据页 {golden_item.evidence_pages} 未被召回；"
        f"候选深度 {depth} 内命中的页码为 {pages}"
    )


def test_golden_question_recall_rate_baseline(engine, golden_qa, requires_index) -> None:
    """召回率基线：10 题中至少 9 题必须能在候选深度内召回证据页。

    基线是在"无语义模型（哈希降级向量）"环境下实测得到的，
    用于防止后续改动导致召回能力进一步退化。
    """
    depth = engine.settings.retrieval.fusion_top_k
    hits = 0
    missed: list[int] = []
    for item in golden_qa:
        analysis = engine.understander.analyze(item.question, [])
        results = engine.retriever.retrieve(
            engine.understander.search_query(analysis), analysis=analysis, top_k=depth
        )
        if {result.chunk.page for result in results} & set(item.evidence_pages):
            hits += 1
        else:
            missed.append(item.id)

    assert hits >= 9, (
        f"证据页召回率下降：仅 {hits}/{len(golden_qa)} 题命中（要求 >= 9），未命中题号 {missed}"
    )


def test_page_filter_of_analysis_is_applied(engine, requires_index) -> None:
    """带"第 N 页"的问题应被解析成页码过滤条件并真正生效。"""
    question = "第 22 页中，公司的注册资本是多少？"
    analysis = engine.understander.analyze(question, [])
    assert analysis.page_filter == [22], f"页码过滤解析失败：{analysis.page_filter}"

    results = engine.retriever.retrieve(engine.understander.search_query(analysis), analysis=analysis, top_k=5)
    assert results, "对第 22 页的过滤检索没有命中任何片段"
    assert all(item.chunk.page == 22 for item in results), (
        f"页码过滤未生效，返回页码 {sorted({item.chunk.page for item in results})}"
    )
