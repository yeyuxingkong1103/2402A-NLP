# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
"""
冒烟测试：不依赖 Milvus，覆盖解析 / 分块 / 去重 / 规则命中 / 文本规范化 /
工单02 的同事实聚合与邻块扩展 / 工单03 的文档配置、表格解析与实体路由。

运行：pytest tests/ -v
Milvus 相关用例在不可达时自动跳过（skip），保证「没起向量库也能跑测试」。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.core.chunker import chunk_pages  # noqa: E402
from app.core.dedup import content_hash, hamming, simhash  # noqa: E402
from app.core.evaluator import retrieval_metrics, rule_hit  # noqa: E402
from app.core.pdf_parser import normalize_text, page_label, parse_pdf_full  # noqa: E402
from app.core.profiles import MIN_CONTEXT_CHARS, PROFILES, get_profile  # noqa: E402
from app.core.retriever import (  # noqa: E402
    _authority_key, _sentences_same_fact, best_window, build_context,
    cluster_facts, expand_with_neighbors, fact_key_sentences, _tokens,
    abstract_query,
)
from app.core.vectorstore import SearchHit, lexical_sparse  # noqa: E402

PDF = settings.data_path / "raw" / "招股说明书1.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason="招股说明书1.pdf 不存在")

# 工单02 的两道目标问题（就是评测集里的 id=95 / id=795）
Q95 = "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"
Q795 = "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"


def _hit(c) -> SearchHit:
    """把 Chunk 转成 SearchHit，供不依赖 Milvus 的纯函数用例使用。"""
    return SearchHit(
        chunk_id=c.chunk_index, score=1.0, content=c.content,
        page_no=c.page_no, page_label=c.page_label, chunk_type=c.chunk_type,
        section_path=c.section_path, doc_name="招股说明书1.pdf",
        chunk_index=c.chunk_index,
    )


@pytest.fixture(scope="module")
def chunks():
    """整篇解析一次、模块内共用（parse 一次约十几秒，别每个用例都跑）。

    只取前 200 页就够：目标片段都在 1-1-26 ~ 1-1-181 之间。
    """
    if not PDF.exists():
        pytest.skip("招股说明书1.pdf 不存在")
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=200)
    return [c for c in chunk_pages(pages) if c.chunk_type == "text"]


# ----------------------------------------------------------------------
# 文本规范化
# ----------------------------------------------------------------------
def test_normalize_removes_cjk_spacing():
    # 实测 PDF 抽出来的样子
    assert normalize_text("注册资本5,520.00 万元") == "注册资本5,520.00万元"
    assert normalize_text("2019 年12 月16 日") == "2019年12月16日"
    assert normalize_text("3.55 元/ 股") == "3.55元/股"


def test_normalize_keeps_english_spacing():
    # 纯英文词组不能被压掉空格
    s = normalize_text("Wuhan Xingtu Xinke Electronics Co.,Ltd.")
    assert s == "Wuhan Xingtu Xinke Electronics Co.,Ltd."


def test_page_label_offset_zero():
    # 实测偏移为 0：index N ↔ 页脚 1-1-N
    assert page_label(22, 0) == "1-1-22"
    assert page_label(129, 0) == "1-1-129"


# ----------------------------------------------------------------------
# 规则化数值命中
# ----------------------------------------------------------------------
def test_rule_hit_all():
    ans = "分别为6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。"
    ok, detail = rule_hit(ans, ["6,464.51", "14,414.16", "18,780.67", "4,627.14"], "all")
    assert ok, detail


def test_rule_hit_tolerates_comma_style():
    # 模型把 15,000.00 写成 15000.00 或 15000 也应算命中
    ok, _ = rule_hit("公司拟投入15000万元用于补充流动资金", ["15,000.00"], "any")
    assert ok


def test_rule_hit_missing():
    ok, detail = rule_hit("资料中未提及相关信息。", ["程家明"], "any")
    assert not ok and "缺失" in detail


def test_rule_hit_all_partial_is_fail():
    ok, _ = rule_hit("收入为6,464.51万元", ["6,464.51", "14,414.16"], "all")
    assert not ok, "all 模式下缺一个就应判未命中"


# ----------------------------------------------------------------------
# 去重
# ----------------------------------------------------------------------
def test_content_hash_stable():
    assert content_hash("abc  def") == content_hash("abcdef")


def test_simhash_similar_texts_are_close():
    a = "公司目前已经成为军队视频指挥领域的重要供应商，参与制定了全军第一个视频指挥系统技术标准。"
    b = "公司目前已经成为军队视频指挥领域的核心供应商，参与制定了全军第一个视频指挥系统技术标准。"
    c = "报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元。"
    assert hamming(simhash(a), simhash(b)) < hamming(simhash(a), simhash(c))


# ----------------------------------------------------------------------
# 稀疏向量
# ----------------------------------------------------------------------
def test_lexical_sparse_keeps_numbers():
    """数字必须整体保留 —— 招股书的核心答案大量是数字。"""
    v = lexical_sparse("注册资本5,520.00万元，页码1-1-42")
    assert len(v) > 0
    # 索引是哈希后的整数，只验证非空与取值范围
    assert all(isinstance(k, int) and 0 <= k < 2**31 for k in v)


# ----------------------------------------------------------------------
# 端到端：解析 + 分块（需要 PDF）
# ----------------------------------------------------------------------
@needs_pdf
def test_parse_removes_header_footer_exactly_once_per_page():
    pages, stats = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=40)
    # n_pages 是文档总页数（用于报告），本次实际解析的是 limit 页
    assert stats.n_pages == 548
    assert len(pages) == 40
    # 实测每页恰好一个页眉、一个页脚
    assert stats.n_header_removed == 40
    assert stats.n_footer_removed == 40


@needs_pdf
def test_key_page_21_has_registered_capital_and_legal_rep():
    """p21 是「法定代表人」「注册资本」两题的答案页 —— 解析错则两题必挂。"""
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=25)
    pc = pages[21]
    blob = "\n".join(t.text for t in pc.texts)
    blob += "\n" + "\n".join(tb.to_markdown() for tb in pc.tables)
    assert "程家明" in blob
    assert "5,520.00" in blob or "5,520" in blob


@needs_pdf
def test_key_page_29_has_supplementary_working_capital():
    """p29 的募集资金表是 id=207 的答案页。"""
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=35)
    pc = pages[29]
    blob = "\n".join(tb.to_markdown() for tb in pc.tables)
    assert "补充流动资金" in blob
    assert "15,000.00" in blob


@needs_pdf
def test_golden_sentence_survives_chunking():
    """
    p128 那句 137 字同时装着 id=260 与 id=33 两个答案，
    固定字数切块会把它切断 —— 本用例锁死这个回归。
    """
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=135)
    chunks = [c for c in chunk_pages(pages) if c.chunk_type == "text"]
    key = ("公司来自军用领域的收入分别为6,464.51万元、14,414.16万元、"
           "18,780.67万元和4,627.14万元，占主营业务收入比重分别为"
           "82.10%、97.31%、94.84%和94.34%")
    assert any(key in c.content for c in chunks), "金句被切断了"


@needs_pdf
def test_chunk_sizes_within_bounds():
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=80)
    texts = [c for c in chunk_pages(pages) if c.chunk_type == "text"]
    assert texts
    # 上限 600 字，允许个别无标点长段落到 800（chunker 有硬切兜底）
    assert max(c.n_chars for c in texts) <= 800


@needs_pdf
def test_table_chunk_is_markdown_with_title():
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=35)
    tables = [c for c in chunk_pages(pages) if c.chunk_type == "table"]
    assert tables
    md = tables[0].content
    assert "|" in md and "---" in md


# ----------------------------------------------------------------------
# 工单02：剖面
# ----------------------------------------------------------------------
def test_profile_context_window_floor_is_enforced():
    """非 baseline 的剖面不许把上下文窗口降到 600 以下。

    这是工单01 踩过的坑：窗口 300 会静默切掉 id=793 的「军队、政府机关、能源」，
    接口照常返回、不报错，只是答案悄悄缺了两项。用断言钉死，别只写注释。
    """
    assert MIN_CONTEXT_CHARS == 600
    assert PROFILES["baseline"].context_chunk_chars < MIN_CONTEXT_CHARS  # 唯一豁免
    with pytest.raises(ValueError):
        PROFILES["delivered"].derived(context_chunk_chars=300)


def test_profile_delivered_matches_workorder01_defaults():
    """delivered 必须与工单01 的常量完全一致 —— 它是重构的回归闸门。"""
    p = get_profile("delivered")
    assert (p.dual_view, p.pool_size, p.lexical_rerank, p.diverse) == (True, 30, True, True)
    assert (p.w_dense, p.dup_jaccard) == (0.5, 0.85)
    assert (p.top_k, p.context_chunk_chars) == (3, 600)
    # baseline 才是刻意的"朴素地板"
    b = get_profile("baseline")
    assert (b.dual_view, b.lexical_rerank, b.diverse) == (False, False, False)
    assert get_profile("不存在的名字").name == "delivered"   # 未知名回退，不抛异常


# ----------------------------------------------------------------------
# 工单02：同事实聚合（用真语料，不用编造的短句）
# ----------------------------------------------------------------------
def _chunk_on(chunks, label: str, anchor: str):
    for c in chunks:
        if c.page_label == label and anchor in c.content:
            return c
    return None


def test_fact_clustering_merges_conflicting_restatements(chunks):
    """id=95 的 top-3 是同一句话的三种互相矛盾的说法，必须聚成一组。

    实测这三块（1-1-158 / 1-1-179 / 1-1-181）的整块 Jaccard 只有
    0.055 / 0.071 / 0.306 —— 远低于 select_diverse 的 0.85，所以字面去重
    抓不到它们。这正是同事实聚合存在的理由。
    """
    a = _chunk_on(chunks, "1-1-158", "全军第一个视频指挥系统技术标准")   # 全军…《某视频指挥系统技术规范（1.0版）》
    b = _chunk_on(chunks, "1-1-179", "国防用户第一个视频指挥系统技术标准")
    c = _chunk_on(chunks, "1-1-181", "全军第一个视频指挥系统技术标准")
    assert a and b and c, "定位不到三块矛盾表述，语料或分块逻辑变了"

    hits = [_hit(x) for x in (a, b, c)]
    qt = _tokens(abstract_query(Q95))
    ka = fact_key_sentences(hits[0], qt)
    kb = fact_key_sentences(hits[1], qt)
    kc = fact_key_sentences(hits[2], qt)

    # a 与 b 的措辞真的分叉（“全军” vs “国防用户”）—— 句子级也判为不同事实
    assert not _sentences_same_fact(ka, kb, 6, 0.65), "预期这两块措辞分叉"
    # 但它们各自都与 c 相同 → 只有传递闭包能把三块并成一组
    assert _sentences_same_fact(ka, kc, 6, 0.65)
    assert _sentences_same_fact(kb, kc, 6, 0.65)

    groups = cluster_facts(hits, Q95, get_profile("optimized"))
    assert len(groups) == 1, f"应聚成 1 组，实际 {[len(g) for g in groups]}"


def test_fact_clustering_does_not_merge_unrelated(chunks):
    """不相干的块不能被并进来 —— 错聚类会直接丢答案，比不聚类严重得多。"""
    a = _chunk_on(chunks, "1-1-158", "全军第一个视频指挥系统技术标准")
    # 讲注册资本/法定代表人的表格题片段，与“技术标准”无关
    other = next(c for c in chunks if "5,520.00" in c.content)
    assert a is not None
    groups = cluster_facts([_hit(a), _hit(other)], Q95, get_profile("optimized"))
    assert len(groups) == 2


def test_authority_prefers_overview_section(chunks):
    """组内代表块优先取「第二节 概览」的权威表述。

    id=795 的组里，1-1-26（第二节概览）那句同时含完整的
    「某情报、指挥、控制与通信网络一体化工程（即相当于美军的C4ISR系统）」。
    """
    c26 = next(c for c in chunks
               if c.page_label == "1-1-26" and "全军第一个" in c.content)
    c181 = _chunk_on(chunks, "1-1-181", "全军第一个视频指挥系统技术标准")
    c158 = _chunk_on(chunks, "1-1-158", "全军第一个视频指挥系统技术标准")
    assert c26.section_path.startswith("第二节")
    rep = min([_hit(c26), _hit(c181), _hit(c158)], key=_authority_key)
    assert rep.page_label == "1-1-26"


# ----------------------------------------------------------------------
# 工单02：邻块扩展
# ----------------------------------------------------------------------
def test_build_context_marks_neighbors_and_keeps_numbering():
    """邻块必须渲染成「邻接补充」，不能占用 [片段N] 编号。

    否则模型会把它当独立证据去引用，而它只是给某条命中块做补充的上下文。
    """
    hits = [
        SearchHit(1, 0.9, "命中的正文一", 10, "1-1-10", "text", "", "a.pdf", 0),
        SearchHit(2, 0.8, "命中的正文二", 11, "1-1-11", "text", "", "a.pdf", 1),
        SearchHit(3, 0.0, "补进来的邻块", 11, "1-1-11", "text", "", "a.pdf", 2,
                  is_neighbor=True, serves=1),
    ]
    ctx = build_context(hits, query="正文", profile=get_profile("delivered"))
    assert "[片段1]" in ctx and "[片段2]" in ctx
    assert "[片段3]" not in ctx                 # 邻块不占编号
    assert "[邻接补充]" in ctx and "片段1 的相邻内容" in ctx


def test_expand_neighbors_requires_complementary_tokens():
    """邻块必须带来原块没有的查询词才收 —— 否则白涨 prompt。

    【口径的前提】互补性是按「查询词集合」算的：邻块里出现了原块没有的、
    且**出现在问题里**的词，才算带来新信息。所以这里的问题与正文都刻意
    不含泛词 —— 实测泛词（如「情况」）确实会被算成新信息，这是该过滤器的
    已知松处；真实场景下由 neighbor_total_chars 的硬上限兜底，不会失控。
    """
    query = "某情报 一体化工程 荣获 国家科技进步一等奖"
    owner = SearchHit(1, 0.9, "公司获得了该项奖励。", 10, "1-1-10", "text",
                      "", "a.pdf", 10, doc_id="d1")
    useful = SearchHit(2, 0.0, "某情报一体化工程荣获国家科技进步一等奖。",
                       10, "1-1-10", "text", "", "a.pdf", 11, doc_id="d1")
    useless = SearchHit(3, 0.0, "公司获得了该项奖励，良好。", 10, "1-1-10",
                        "text", "", "a.pdf", 12, doc_id="d1")

    p = get_profile("optimized")
    added, total = expand_with_neighbors([owner], query, p,
                                         lambda idx, doc: [useful, useless])
    assert [h.chunk_index for h in added] == [11], "无关邻块应被互补性过滤挡掉"
    assert added[0].is_neighbor and added[0].serves == 10
    assert total == len(added[0].content)


def test_expand_neighbors_accepts_table_even_without_new_query_words():
    """表格邻块不受互补性过滤约束（工单03）。

    实测 id=2：page 21 的引子句「五、募集资金用途本次募集资金拟投资以下项目：」和
    它下面那张项目表被切成了相邻两块。引子句与问法高度相似 → 它被稠密检索召回，
    而表格连候选池都进不去。偏偏表格里全是项目名和金额、**不含任何新的查询词**，
    于是被互补性过滤判成"没带来新信息"丢掉 —— 可答案就是它。
    查询词集合这套判据天然测不出"邻块里有答案"，所以表格必须无条件收。
    """
    query = "本次募集资金拟投资哪些项目？"
    lead = SearchHit(1, 0.9, "五、募集资金用途本次募集资金拟投资以下项目：",
                     21, "21", "text", "", "招股说明书2.pdf", 42, doc_id="d2")
    table = SearchHit(2, 0.0, "**五、募集资金用途**\n\n| 序号 | 项目名称 | 计划总投资(万元) |\n"
                              "|---|---|---|\n| 1 | 仓储及物流中心 | 3,393.40 |",
                      21, "21", "table", "", "招股说明书2.pdf", 41, doc_id="d2")

    added, _ = expand_with_neighbors([lead], query, get_profile("optimized"),
                                     lambda idx, doc: [table])
    assert [h.chunk_index for h in added] == [41], "表格邻块必须被收进来"
    assert added[0].is_neighbor and added[0].serves == 42
    # 整表不可切分：不能按查询词密度截窗口，否则表会被切在行中间
    assert added[0].content == table.content


def test_expand_neighbors_respects_total_cap():
    query = "某情报 一体化工程 荣获 国家科技进步一等奖"
    owner = SearchHit(1, 0.9, "公司获得了该项奖励。", 10, "1-1-10", "text",
                      "", "a.pdf", 10, doc_id="d1")
    nbr = SearchHit(2, 0.0, "某情报一体化工程荣获国家科技进步一等奖。" * 6,
                    10, "1-1-10", "text", "", "a.pdf", 11, doc_id="d1")
    p = get_profile("optimized").derived(neighbor_total_chars=40)
    added, total = expand_with_neighbors([owner], query, p, lambda idx, doc: [nbr])
    assert total <= 40, f"邻块合计 {total} 字超过了上限 40"


def test_expand_neighbors_survives_fetch_failure():
    """邻块是锦上添花：取不到就当没有，绝不能让问答整体挂掉。"""
    owner = SearchHit(1, 0.9, "正文", 10, "1-1-10", "text", "", "a.pdf", 10, doc_id="d1")

    def boom(idx, doc):
        raise RuntimeError("Milvus 挂了")

    added, total = expand_with_neighbors([owner], "问题", get_profile("optimized"), boom)
    assert added == [] and total == 0


# ----------------------------------------------------------------------
# 工单02：检索侧指标
# ----------------------------------------------------------------------
def test_retrieval_metrics_excludes_neighbors_from_precision():
    """邻块不进精确率分母，否则新机制会机械压低自己的成绩。"""
    hits = [
        SearchHit(1, 0.9, "含答案的正文", 128, "1-1-128", "text", "", "a.pdf", 0),
        SearchHit(2, 0.0, "邻块", 999, "1-1-999", "text", "", "a.pdf", 1,
                  is_neighbor=True),
    ]
    m = retrieval_metrics(hits, ["答案"], "1-1-128")
    assert m["retrieved_pages"] == ["1-1-128"]        # 邻块的页不在召回页里
    assert m["page_precision"] == 1.0
    assert m["page_recall"] == 1.0
    assert m["context_keyword_coverage"] == 1.0


def test_retrieval_metrics_separates_retrieval_from_generation():
    """CKC 能识破「检索对了但生成没答出来」—— 这正是 id=95/795 的真实情况。"""
    hits = [SearchHit(1, 0.9, "参与制定了全军第一个视频指挥系统技术标准。",
                      158, "1-1-158", "text", "", "a.pdf", 0)]
    m = retrieval_metrics(hits, ["视频指挥系统技术标准"], "1-1-158")
    assert m["context_keyword_coverage"] == 1.0, "关键词明明在上下文里"
    # 而端到端的 rule_hit 取决于模型输出，可能仍是 False
    ok, _ = rule_hit("参与制定了《某视频技术规范1.0》。", ["视频指挥系统技术标准"], "any")
    assert not ok


def test_context_keyword_coverage_normalizes_numbers():
    """CKC 必须与 rule_hit 用同一套匹配规则（数字按数值比，不按字面）。

    实测踩过：CKC 最初写成字面子串匹配，`15000` 这种缺千分位的写法被判成"缺失"，
    id=207/543 的 CKC 被压到 0.33（规则判法是 3/3、2/3）。两套口径不一致会让
    对比表整体偏低，且系统性偏向惩罚数字题。

    【已知限度，别夸大成"全等"】带单位后缀的关键词（`15,000万`）**不做**数值归一化：
    `_keyword_is_numeric` 认为去掉数字后还剩「万」字就不是纯数值，只能字面匹配。
    这是工单01 原有的保守行为（旧评估里 543 报「缺失 5,520万」就是它），
    本用例只保证 **CKC 与 rule_hit 口径一致**，不假装两者都能认 万 后缀。
    """
    hits = [SearchHit(1, 0.9, "| 补充流动资金 | 15,000.00 |", 30, "1-1-30",
                      "table", "", "a.pdf", 0)]
    kws = ["15,000.00", "15,000万", "15000"]
    m = retrieval_metrics(hits, kws, "1-1-30")
    # 15,000.00 字面命中、15000 数值命中、15,000万 不命中 → 2/3
    assert m["context_keyword_coverage"] == pytest.approx(2 / 3, abs=1e-4)
    # 关键不变量：CKC 的判定与 rule_hit 同源
    from app.core.evaluator import keyword_hits
    assert keyword_hits("".join(h.content for h in hits), kws) == ["15,000.00", "15000"]


def test_keyword_hits_matches_rule_hit():
    """两个函数必须同源，不能各写一套。"""
    from app.core.evaluator import keyword_hits
    text = "拟投入15000万元用于补充流动资金"
    kws = ["15,000.00", "不存在的词"]
    ok, _ = rule_hit(text, kws, "any")
    assert ok == bool(keyword_hits(text, kws))
    assert keyword_hits(text, kws) == ["15,000.00"]


# ----------------------------------------------------------------------
# 工单02：上下文窗口回归（锁死工单01 踩过的坑）
# ----------------------------------------------------------------------
def test_context_window_600_keeps_answer_but_300_truncates(chunks):
    """p151 那块 408 字的片段：答案句在 336 字处。

    窗口 300 会把它切掉，模型只看到前文图注里的「军队企事业单位」，
    于是答成「军队、企事业单位」，漏了政府机关与能源 ——
    **答案不是检索错了，是被截断毁了**，而且不报错。
    """
    c = next(x for x in chunks if "军队、政府机关、能源" in x.content)
    qt = _tokens(abstract_query(
        "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"))
    wide = best_window(c.content, qt, 600)
    narrow = best_window(c.content, qt, 300)
    assert "军队、政府机关、能源" in wide, "600 字窗口必须保住答案"
    assert "军队、政府机关、能源" not in narrow, (
        "300 字窗口预期会截断答案 —— 若这条不再成立，说明窗口策略变了，"
        "请同步更新工单02 的优化方案文档")


# ----------------------------------------------------------------------
# 工单03：文档配置 / 页码格式 / 表格解析
# ----------------------------------------------------------------------
def test_doc_profiles_cover_both_pdfs_and_reject_unknown():
    from app.core.doc_profiles import DOC_PROFILES, profile_for_path
    assert set(DOC_PROFILES) == {"xingtu", "liyuan"}
    assert profile_for_path("招股说明书1.pdf").key == "xingtu"
    assert profile_for_path("/任意/路径/招股说明书2.pdf").key == "liyuan"
    # 没配过的文档必须**当场报错**，不能静默套用别的文档的页眉页脚规则
    with pytest.raises(ValueError, match="没有为"):
        profile_for_path("第三家公司的招股书.pdf")


def test_page_label_format_is_per_document():
    """页码格式按文档走：书1 是 `1-1-N`，书2 是裸数字 `N`。

    统一成 `1-1-N` 会有两个后果：引用与 PDF 里印的页码对不上（验收没法核对），
    且两份文档会产生同名页码，页级指标跨文档歧义。
    """
    from app.core.doc_profiles import DOC_PROFILES
    assert page_label(22, 0, DOC_PROFILES["xingtu"]) == "1-1-22"
    assert page_label(22, 0, DOC_PROFILES["liyuan"]) == "22"
    assert page_label(207, 0, DOC_PROFILES["liyuan"]) == "207"
    # 不传 doc 时保持工单01/02 的既有行为
    assert page_label(22, 0) == "1-1-22"


def _table(rows, title=""):
    from app.core.pdf_parser import TableBlock
    return TableBlock(page_no=0, page_label="x", rows=rows, bbox=(0, 0, 1, 1),
                      title=title)


def test_table_drops_all_empty_columns():
    """整列全空才删；有值的列一个都不动。"""
    tb = _table([
        ["", "序号", "", "项目名称", "", "金额"],
        ["1", "", "", "甲", "", "10"],
        ["2", "", "", "乙", "", "20"],
    ])
    md = tb.to_markdown()
    head = [l for l in md.split("\n") if l.startswith("|")][0]
    assert head.count("|") - 1 == 3, f"应剩 3 列，实际：{head}"
    assert "序号" in md and "项目名称" in md and "金额" in md
    assert "甲" in md and "20" in md


def test_table_column_drop_keeps_every_value():
    """砍列不能丢任何非空单元格 —— 这是"宁可留噪声也不丢数据"的底线。"""
    rows = [["A", "", "B", "", "C"], ["1", "", "2", "", "3"], ["", "x", "", "", "y"]]
    md = _table(rows).to_markdown()
    for v in ("A", "B", "C", "1", "2", "3", "x", "y"):
        assert v in md, f"值 {v!r} 被砍列弄丢了"


def test_table_realigns_merged_header():
    """find_tables 的合并表头会错位一列，必须按出现顺序重新对齐。

    实测第 21 页募投表：表头值在列 1/4/7，数据值在列 0/3/6。
    不修就是「序号」压在「—」上；修完才能得到
    `| 序号 | 项目名称 | 计划总投资(万元) |`。
    """
    tb = _table([
        ["", "序号", "", "", "项目名称", "", "", "计划总投资(万元)", ""],
        ["1", None, None, "仓储及物流中心", None, None, "3,393.40", None, None],
    ])
    md = tb.to_markdown()
    lines = [l for l in md.split("\n") if l.startswith("|")]
    assert lines[0] == "| 序号 | 项目名称 | 计划总投资(万元) |", lines[0]
    assert lines[2] == "| 1 | 仓储及物流中心 | 3,393.40 |", lines[2]


def test_table_escapes_literal_pipe():
    """单元格里的字面竖线必须转义，否则整张表结构崩掉。

    实测招股书1 第 1-1-20 页术语表里有一处 `参数扰动|摄动`。
    """
    md = _table([["术语", "释义"], ["鲁棒性", "参数扰动|摄动"]], title="").to_markdown()
    body = [l for l in md.split("\n") if l.startswith("|")]
    # 按 Markdown 规则切列：分隔符是不带反斜杠的 `|`
    ncols = len(re.split(r"(?<!\\)\|", body[2])) - 2
    assert ncols == 2, f"转义后应仍是 2 列，实际 {ncols} 列：{body[2]}"
    assert r"参数扰动\|摄动" in body[2]
    # 整张表每行列数必须一致（否则 Markdown 渲染崩掉）
    counts = {len(re.split(r"(?<!\\)\|", l)) - 2
              for l in md.split("\n") if l.startswith("|")}
    assert counts == {2}, f"各行列数不一致：{counts}"


@needs_pdf
def test_pdf2_header_footer_fully_removed():
    """招股书2 的页眉页脚必须**全部**清掉。

    它们的模板与书1 完全不同（书2 页脚是裸数字），工单03 之前实测各清 0 条。
    350 页里 342 页是常规页、8 页是旋转 90° 的横向表（页眉页脚不在常规分带内），
    总共 350 条 —— 这个数是逐页扫出来的，不是估的。
    """
    pdf2 = settings.data_path / "raw" / "招股说明书2.pdf"
    if not pdf2.exists():
        pytest.skip("招股说明书2.pdf 不存在")
    pages, stats = parse_pdf_full(pdf2, offset=settings.page_label_offset)
    assert stats.n_pages == 350
    assert stats.n_header_removed == 350, f"页眉只清掉 {stats.n_header_removed} 条"
    assert stats.n_footer_removed == 350, f"页脚只清掉 {stats.n_footer_removed} 条"
    # 页码格式是裸数字
    assert pages[21].page_label == "21"


@needs_pdf
def test_pdf2_table_titles_not_polluted_by_header():
    """页眉没清掉时会被"表格补标题"启发式抓走，变成表的标题（实测过）。

    正常正文里出现「招股意向书」字样的标题是合法的（如
    「截至本招股意向书签署日…」），所以只断言**标题不等于页眉那一行**。
    """
    pdf2 = settings.data_path / "raw" / "招股说明书2.pdf"
    if not pdf2.exists():
        pytest.skip("招股说明书2.pdf 不存在")
    pages, _ = parse_pdf_full(pdf2, offset=settings.page_label_offset, limit=25)
    header = "武汉力源信息技术股份有限公司招股意向书"
    bad = [tb.title for pc in pages for tb in pc.tables
           if tb.title.replace(" ", "") == header]
    assert not bad, f"有表的标题被页眉污染：{bad}"
    # 第 21 页的募投表标题应是正文里的小节名
    p21 = pages[21]
    titles = [tb.title for tb in p21.tables]
    assert "五、募集资金用途" in titles, titles


# ----------------------------------------------------------------------
# 工单03：实体路由（把问题里的公司专名变成 doc_name 硬过滤）
# ----------------------------------------------------------------------
def test_route_picks_the_company_named_in_the_question():
    """两份招股书章节结构雷同、关键词两边都有，只能靠公司专名确定地分流。"""
    from app.core.router import route

    r2 = route("武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？")
    assert r2.doc_key == "liyuan" and r2.doc_name == "招股说明书2.pdf"
    assert r2.routed and r2.expr == 'doc_name == "招股说明书2.pdf"'
    # 最长匹配优先：matched 要记全称。「武汉力源」也能命中同一份文档，
    # 但记成短名的话，排查时看不出到底匹配到了什么。
    assert r2.matched == "武汉力源信息技术股份有限公司"

    r1 = route("武汉兴图新科电子股份有限公司法定代表人是谁？")
    assert r1.doc_key == "xingtu" and r1.expr == 'doc_name == "招股说明书1.pdf"'

    # 没点名 → 不过滤（expr 空），而不是随便挑一份
    r0 = route("招股说明书里募集资金用途是什么？")
    assert not r0.routed and r0.expr == ""


def test_route_multi_detects_questions_about_two_companies():
    """一句话里点了两家公司时不能过滤 —— 限定任何一份都会漏掉另一半。"""
    from app.core.router import route_multi
    assert route_multi(
        "武汉兴图新科电子股份有限公司和武汉力源信息技术股份有限公司的注册资本各是多少？"
    ) == ["liyuan", "xingtu"]


def test_entity_names_cover_both_documents_longest_first():
    """查询抽象掉的专名必须两份文档都在。

    只列兴图那三个名字的话，力源的题会原样重演工单01 的召回退化
    （公司全称主导查询向量）；乱序的话会先命中短名。
    """
    from app.core.retriever import _entity_names
    names = _entity_names()
    assert "武汉力源信息技术股份有限公司" in names
    assert "武汉兴图新科电子股份有限公司" in names
    assert names == sorted(names, key=len, reverse=True)


class _CannedStore:
    """按过滤条件返回预置命中，并记下每次 search 实际下推的 expr。"""

    def __init__(self, by_expr):
        self.by_expr = by_expr
        self.exprs: list[str] = []

    def search(self, vector, top_k, *, expr=None):
        self.exprs.append(expr or "")
        return list(self.by_expr.get(expr or "", []))

    def fetch_by_chunk_index(self, idxs, doc_id):
        return []


class _CannedEmbedder:
    async def embed_one(self, text):
        return [0.0] * 4


def _canned_hit(cid, text, doc_name):
    return SearchHit(cid, 0.9, text, 21, "21", "text", "", doc_name, cid,
                     doc_id=doc_name)


def _canned(*by_expr):
    from app.core.retriever import Retriever
    store = _CannedStore(dict(by_expr))
    return store, Retriever(store=store, embedder=_CannedEmbedder(),
                            profile=get_profile("delivered"))


def test_retrieve_hard_filters_to_the_routed_document():
    """路由必须真的落到 search 的 filter 上 —— 只写个 route() 不接线就是死代码。"""
    liyuan = 'doc_name == "招股说明书2.pdf"'
    store, ret = _canned((liyuan, [_canned_hit(1, "本次发行股数为 3,000 万股。",
                                       "招股说明书2.pdf")]))
    r = ret.retrieve_sync("武汉力源信息技术股份有限公司本次发行股数是多少？")
    # delivered 剖面是双路召回，所以同一份过滤条件会下推两次
    assert set(store.exprs) == {liyuan}, f"实际下推的过滤条件：{store.exprs}"
    assert r.routed_doc == "招股说明书2.pdf"
    assert not r.route_fallback


def test_retrieve_falls_back_to_full_search_when_routed_search_is_empty():
    """路由后零召回要回退全库，并把这件事记下来。

    问题里点了名的公司，和答案所在的公司**不总是同一家**；空上下文会让模型
    只能凭空编。而 route_fallback 不留痕的话，事后分不清"路由生效了"和
    "路由白过滤了一次"。
    """
    store, ret = _canned(("", [_canned_hit(9, "答案其实在另一份文档里。",
                                   "招股说明书1.pdf")]))
    r = ret.retrieve_sync("武汉力源信息技术股份有限公司本次发行股数是多少？")
    assert store.exprs[0] == 'doc_name == "招股说明书2.pdf"'
    assert store.exprs[-1] == "", "过滤后零召回必须回退全库"
    assert r.route_fallback and r.routed_doc == "招股说明书2.pdf"
    assert [h.doc_name for h in r.hits] == ["招股说明书1.pdf"]


def test_system_prompt_names_the_routed_document():
    """system prompt 必须按文档走，否则会造成**系统性拒答**。

    原先写死「基于《武汉兴图新科电子股份有限公司招股意向书》片段回答问题」
    和「（见 1-1-XX 页）」。加第二份文档后，模型读到"这是兴图新科的文件"，
    于是对力源的问题判定"问题与文档不符"，直接输出"资料中未提及相关信息" ——
    而答案就在片段里（实测 id=1：1,670 万股 / 25.04% 就在上下文里）。
    它还会按 1-1-XX 编造书2 的页码（书2 页脚是裸数字）。
    """
    from app.core.doc_profiles import DOC_PROFILES
    from app.core.generator import SYSTEM_PROMPT, system_prompt_for

    xingtu = system_prompt_for(DOC_PROFILES["xingtu"])
    liyuan = system_prompt_for(DOC_PROFILES["liyuan"])
    generic = system_prompt_for(None)

    assert "武汉兴图新科电子股份有限公司" in xingtu
    assert "1-1-XX" in xingtu
    assert "武汉力源信息技术股份有限公司" in liyuan
    assert "1-1-XX" not in liyuan, "书2 不能出现书1 的页码格式"
    assert "（见 XX 页）" in liyuan
    # 各文档特有的作答提醒（书1 的「某」脱敏 / 书2 的 [◆] 占位）
    assert "某" in xingtu and "[ ◆ ]" in liyuan

    # 不点名文档时不能硬套某一份 —— 跨文档提问时点名任何一家都是错的
    assert "兴图" not in generic and "力源" not in generic
    assert "1-1-XX" not in generic
    assert SYSTEM_PROMPT == generic
    # 页码占位符必须像个页码。填字面「页码」的话规则会读作「用『（见 页码 页）』的形式」，
    # 模型照抄模板，输出「（见 页码 12 页）」这种畸形引用（实测英文提问时必现）
    assert "（见 XX 页）" in generic
    assert "（见 页码 页）" not in generic
    assert "页码 页" not in generic


def test_page_re_follows_the_routed_document():
    """证据页抽取必须按文档走格式，否则书2 的页级指标整列静默失效。

    书2 的 evidence_page 是裸数字 `"21"`，拿书1 的 `1-1-\\d+` 去抽是空集 →
    gold 为空 → page_recall 变成 None，不报错、只是指标没了。
    """
    from app.core.evaluator import page_re_for, retrieval_metrics
    liyuan = [_canned_hit(1, "正文", "招股说明书2.pdf")]
    xingtu = [_canned_hit(1, "正文", "招股说明书1.pdf")]

    assert page_re_for(liyuan, "招股说明书2.pdf").findall("21 23") == ["21", "23"]
    assert page_re_for(xingtu, "招股说明书1.pdf").findall("1-1-128") == ["1-1-128"]
    # 没路由时退回命中块所属的文档；命中跨文档则用书1 格式（保持旧行为）
    assert page_re_for(liyuan, "").findall("21") == ["21"]
    assert page_re_for(liyuan + xingtu, "").findall("21") == []

    # 端到端：书2 的证据页能被抽出来，page_recall 才算得出来
    rm = retrieval_metrics(liyuan, [], "21", page_re=page_re_for(liyuan, ""))
    assert rm["evidence_pages"] == ["21"]
    # 命中块的 page_label 是 "21"（见 _canned_hit）→ 精确召回
    assert rm["page_recall"] == 1.0


def test_retrieve_does_not_filter_when_two_companies_are_named():
    store, ret = _canned(("", [_canned_hit(1, "两家公司的注册资本…", "招股说明书1.pdf"),
                               _canned_hit(2, "两家公司的注册资本…", "招股说明书2.pdf")]))
    r = ret.retrieve_sync("兴图新科和力源信息的注册资本各是多少？")
    assert set(store.exprs) == {""}, f"跨文档不该下推过滤：{store.exprs}"
    # 没过滤就别在 routed_doc 里写一份 —— 否则"没过滤"看起来像"过滤了"
    assert r.routed_doc == "" and not r.route_fallback


# ----------------------------------------------------------------------
# Milvus（不可达则跳过）
# ----------------------------------------------------------------------
def test_milvus_roundtrip():
    from app.core.vectorstore import VectorStore, VectorStoreError
    store = VectorStore()
    ok, msg = store.health()
    if not ok:
        pytest.skip(f"Milvus 不可达：{msg[:80]}")
    assert store.collection
