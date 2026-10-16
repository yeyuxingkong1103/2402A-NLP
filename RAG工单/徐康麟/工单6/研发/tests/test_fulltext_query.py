# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-混合检索任务（工单6）

工单编号：人工智能NLP-RAG-混合检索任务

``app/core/fulltext_query.py`` 的**离线单元测试**（纯内存构造假 bundle + 手工倒排索引，
不联网、不读 2676 块的真实索引、不调 Ollama）：

1. 查询 DSL 解析（布尔 / 短语 / 模糊 / 字段限定 / 括号优先级）与 ``describe()`` 输出；
2. **查询侧分词回归**：中文自然语言长问句必须被切成多个词项（整串查表会恒 0 命中）；
3. ``标题:"军用"`` 必须是**单个** ``Phrase``（字段限定不得被静默丢掉）；
4. 纯否定查询守卫（空结果 + notes 说明）与双重否定 / 混合否定；
5. OOV 词项（命中分词停用词表）必须如实上报；
6. 有界编辑距离与 ``FulltextIndex.fuzzy_expand`` 的距离剪枝；
7. ``field_text_of`` 四个字段各自的提取口径。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_fulltext_query.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.fulltext_query import (DEFAULT_FIELDS, And, FulltextIndex, Node,  # noqa: E402
                                     Not, Or, Phrase, Term, _bounded_edit_distance,
                                     evaluate_query, field_text_of, parse_fulltext_query,
                                     search_fulltext)
from app.core.text_utils import STOPWORDS, tokenize  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

#: 假语料（4 块，纯内存）：m1/m2 含「军用」，m4 含发行股数，m3 只含「民用」
CORPUS: dict[str, dict[str, Any]] = {
    "m1": {"content": "公司来自军用领域的收入为 100 万元",
           "section": "军用领域", "keywords": ["军用", "收入"]},
    "m2": {"content": "军用产品毛利率为 40%",
           "section": "毛利率", "keywords": ["军用"]},
    "m3": {"content": "民用市场收入为 200 万元",
           "section": "民用市场", "keywords": ["民用"]},
    "m4": {"content": "本次发行股数为 5,000 万股，占发行后总股本的 25%",
           "section": "发行概况", "keywords": ["发行", "股数"]},
}
ORDER: tuple[str, ...] = ("m1", "m2", "m3", "m4")

#: 假的「词项 → 行 → 分」表（模拟 BM25 打分，避免依赖真实索引）
TERM_SCORES: dict[str, dict[str, float]] = {
    "军用": {"m1": 3.0, "m2": 2.0},
    "领域": {"m1": 2.0},
    "收入": {"m1": 1.0, "m3": 1.0},
    "民用": {"m3": 2.0},
    "发行": {"m4": 4.0},
    "股数": {"m4": 3.0},
    "股本": {"m4": 2.0},
}

#: 模块 docstring 里点名的坑：中文长问句（无空格、无运算符）
LONG_QUESTION = "本次发行股数是多少，占发行后总股本的比例是多少？"


class _FakeInnerBM25:
    """``bundle.bm25.bm25`` 的假实现：只要 ``idf`` 与 ``get_scores``。"""

    idf: dict[str, float] = {"军用": 2.0, "领域": 1.5, "收入": 1.2, "发行": 2.2, "股数": 2.2}

    def get_scores(self, terms: list[str]) -> list[float]:
        scores = [0.0] * len(ORDER)
        for term in terms:
            for cid, value in TERM_SCORES.get(str(term), {}).items():
                scores[ORDER.index(cid)] += value
        return scores


@pytest.fixture()
def bundle() -> SimpleNamespace:
    """假 bundle：``order`` / ``chunks`` / ``bm25.doc_tokens`` / ``bm25.bm25`` 齐备即可。"""
    return SimpleNamespace(
        index_dir="",
        order=list(ORDER),
        chunks=CORPUS,
        bm25=SimpleNamespace(
            doc_tokens=[tokenize(CORPUS[cid]["content"]) for cid in ORDER],
            bm25=_FakeInnerBM25(),
        ),
    )


@pytest.fixture()
def index(bundle: SimpleNamespace) -> FulltextIndex:
    return FulltextIndex.build(bundle)


def _walk_terms(node: Node | None) -> list[Term]:
    """把 AST 里所有 ``Term`` 摊平（测试内的小工具，不依赖被测模块的私有实现）。"""
    if node is None:
        return []
    if isinstance(node, Term):
        return [node]
    if isinstance(node, Not):
        return _walk_terms(node.child)
    children = getattr(node, "children", [])
    out: list[Term] = []
    for child in children:
        out.extend(_walk_terms(child))
    return out


def _handmade_index() -> FulltextIndex:
    """手工构造的倒排索引（词表里有「军用」的两个近词，用于模糊匹配用例）。"""
    idx = FulltextIndex(index_dir="", chunk_ids=["r0", "r1", "r2"])
    idx.postings["content"] = {"军用": [0], "军民": [0, 1], "军队": [1], "完全无关的另一个词": [2]}
    idx.vocab["content"] = set(idx.postings["content"])
    idx.field_tokens["content"] = [["军用"], ["军民"], ["完全无关的另一个词"]]
    return idx


# ---------------------------------------------------------------------------
# ① DSL 解析正确性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected_type,expected_describe", [
    ("甲 AND 乙", And, "(甲 AND 乙)"),
    ("甲 && 乙", And, "(甲 AND 乙)"),
    ("甲 且 乙", And, "(甲 AND 乙)"),
    ("甲 OR 乙", Or, "(甲 OR 乙)"),
    ("甲 || 乙", Or, "(甲 OR 乙)"),
    ("NOT 甲", Not, "NOT 甲"),
    ("-甲", Not, "NOT 甲"),
    ("非 甲", Not, "NOT 甲"),
    ('"短语"', Phrase, '"短语"'),
    ("词~", Term, "词~1"),
    ("词~2", Term, "词~2"),
    ("标题:词", Term, "title:词"),
    ("(甲 OR 乙) AND 丙", And, "((甲 OR 乙) AND 丙)"),
], ids=["AND", "AND-&&", "AND-且", "OR", "OR-||", "NOT", "NOT-减号", "NOT-非",
        "短语", "模糊-默认1", "模糊-2", "字段-title", "括号分组"])
def test_dsl_parse_structure_and_describe(query: str, expected_type: type,
                                          expected_describe: str) -> None:
    node = parse_fulltext_query(query)
    assert isinstance(node, expected_type), f"{query!r} 解析成了 {type(node).__name__}"
    assert node.describe() == expected_describe


def test_dsl_parentheses_force_and_over_or() -> None:
    """``(甲 OR 乙) AND 丙``：顶层是 And，左子节点是 Or（优先级 NOT > AND > OR）。"""
    node = parse_fulltext_query("(甲 OR 乙) AND 丙")
    assert isinstance(node, And) and len(node.children) == 2
    assert isinstance(node.children[0], Or)
    assert [c.describe() for c in node.children[0].children] == ["甲", "乙"]
    assert node.children[1].describe() == "丙"
    assert node.positive_terms() == ["甲", "乙", "丙"]


def test_dsl_not_contributes_no_positive_terms() -> None:
    """被 NOT 包住的词不得计入正向词项（否则被排除的词反而会给分）。"""
    assert parse_fulltext_query("NOT 甲").positive_terms() == []
    assert parse_fulltext_query("军用 AND NOT 领域").positive_terms() == ["军用"]


def test_dsl_fuzzy_and_field_details() -> None:
    node = parse_fulltext_query("词~2")
    assert isinstance(node, Term) and node.fuzzy_distance == 2 and node.field == "content"
    fields = parse_fulltext_query("标题:发行股数")
    assert fields.describe() == "(title:发行 OR title:股数)"
    assert {t.field for t in _walk_terms(fields)} == {"title"}


def test_dsl_analyze_false_keeps_exact_term() -> None:
    """``analyze=False`` 用于精确词项匹配（调试/测试），此时不再分词。"""
    node = parse_fulltext_query("发行股数", analyze=False)
    assert isinstance(node, Term) and node.text == "发行股数"


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"], ids=["None", "空串", "空白", "制表符"])
def test_dsl_empty_query_returns_none(raw: object) -> None:
    assert parse_fulltext_query(raw) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ② 查询侧分词（踩坑回归：整串查表恒 0 命中）
# ---------------------------------------------------------------------------
def test_query_side_tokenization_splits_chinese_question() -> None:
    node = parse_fulltext_query(LONG_QUESTION)
    terms = _walk_terms(node)
    texts = [t.text for t in terms]
    assert len(terms) >= 3, f"中文长问句被当成了单个词项：{texts}"
    assert LONG_QUESTION not in texts
    assert "发行" in texts and "股数" in texts


def test_long_question_hits_index_but_raw_string_does_not(bundle: SimpleNamespace,
                                                          index: FulltextIndex) -> None:
    """回归核心：整串作为词项在倒排词表里不存在（恒 0 命中），
    但经查询侧分词后必须能召回 —— 这正是本模块「文档侧/查询侧同口径分词」的护栏。"""
    assert index.rows_with_term(LONG_QUESTION, "content") == set()
    hits = search_fulltext(bundle, LONG_QUESTION, index=index)
    assert hits, "中文长问句经分词后仍 0 命中，查询侧分词回归失败"
    assert "m4" in {cid for cid, _ in hits}


# ---------------------------------------------------------------------------
# ③ 字段 + 短语：必须合成单个 Phrase
# ---------------------------------------------------------------------------
def test_field_phrase_is_single_phrase_node() -> None:
    """``标题:"军用"`` 必须是一个 ``Phrase(field='title')``，
    而不是被切成 ``标题:`` + ``"军用"`` 两个 token（后者会静默丢掉字段限定）。"""
    node = parse_fulltext_query('标题:"军用"')
    assert isinstance(node, Phrase), f"解析成了 {type(node).__name__}：{node!r}"
    assert node.field == "title"
    assert node.text == "军用"
    assert node.describe() == 'title:"军用"'
    assert node.positive_terms() == ["军用"]


def test_field_phrase_actually_scopes_to_title_field(index: FulltextIndex) -> None:
    """功能验证：字段限定确实生效（不是只体现在 describe 上）。

    ``军用`` 在 content 字段命中 m1/m2；``标题:军用`` 只命中标题字段含该词的 m1。
    """
    plain = evaluate_query(index, parse_fulltext_query("军用"))
    scoped = evaluate_query(index, parse_fulltext_query("标题:军用"))
    assert {index.chunk_ids[r] for r in plain.rows} == {"m1", "m2"}
    assert {index.chunk_ids[r] for r in scoped.rows} == {"m1"}
    phrases = [p for p in evaluate_query(index, parse_fulltext_query('标题:"军用"')).phrases]
    assert phrases == [("军用", "title")]


def test_phrase_requires_adjacent_tokens(index: FulltextIndex) -> None:
    """短语匹配要求分词后位置相邻：``"收入军用"``（顺序反了）不得命中 m1。"""
    node = parse_fulltext_query('"收入军用"')
    assert isinstance(node, Phrase)
    assert evaluate_query(index, node).rows == set()
    assert evaluate_query(index, parse_fulltext_query('"军用领域"')).rows == {0}


# ---------------------------------------------------------------------------
# ④ 纯否定查询守卫
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", ["NOT 军用", "-军用", "非 军用"],
                         ids=["NOT", "减号", "非"])
def test_pure_negative_query_returns_empty_with_note(bundle: SimpleNamespace,
                                                     index: FulltextIndex, query: str) -> None:
    diag: dict[str, Any] = {}
    hits = search_fulltext(bundle, query, index=index, diag=diag)
    assert hits == [], f"{query!r} 是纯否定查询，必须返回空结果而不是几乎全库"
    assert diag["positive_terms"] == []
    assert diag["boolean_hits"] == 0
    assert any("纯否定查询" in note for note in diag["notes"]), diag["notes"]


@pytest.mark.parametrize("query,expected", [
    ("NOT NOT 军用", {"m1", "m2"}),
    ("军用 AND NOT 领域", {"m2"}),
], ids=["双重否定", "AND-NOT"])
def test_negation_keeping_a_positive_term_is_not_empty(bundle: SimpleNamespace,
                                                       index: FulltextIndex,
                                                       query: str, expected: set[str]) -> None:
    hits = search_fulltext(bundle, query, index=index)
    assert {cid for cid, _ in hits} == expected, f"{query!r} 有正向词项，不应被纯否定守卫拦掉"
    direct = evaluate_query(index, parse_fulltext_query(query))
    assert {index.chunk_ids[r] for r in direct.rows} == expected
    assert direct.positive_terms == ["军用"]


# ---------------------------------------------------------------------------
# ⑤ OOV 词项上报
# ---------------------------------------------------------------------------
def test_stopword_term_is_reported_as_oov(bundle: SimpleNamespace, index: FulltextIndex) -> None:
    """「项目」在 ``text_utils.STOPWORDS`` 里 ⇒ 分词时被丢掉 ⇒ 索引里本就没有。"""
    assert "项目" in STOPWORDS, "前提失效：停用词表已不含「项目」"
    diag: dict[str, Any] = {}
    hits = search_fulltext(bundle, "项目 AND 军用", index=index, diag=diag)
    assert hits == [], "AND 上一个词表外的词，必然 0 命中"
    assert "项目" in diag["oov_terms"]
    assert "军用" not in diag["oov_terms"], "词表内的词不得被误报为 OOV"


def test_oov_reported_even_when_query_alone_returns_nothing(bundle: SimpleNamespace,
                                                           index: FulltextIndex) -> None:
    diag: dict[str, Any] = {}
    assert search_fulltext(bundle, "项目", index=index, diag=diag) == []
    assert "项目" in diag["oov_terms"]
    assert diag["positive_terms"] == ["项目"], "OOV 词仍应作为正向词项上报，便于界面解释"


def test_evaluate_query_collects_oov_terms(index: FulltextIndex) -> None:
    evaluation = evaluate_query(index, parse_fulltext_query("项目 OR 军用"))
    assert evaluation.oov_terms == ["项目"]
    assert {index.chunk_ids[r] for r in evaluation.rows} == {"m1", "m2"}


# ---------------------------------------------------------------------------
# ⑥ 模糊匹配：有界编辑距离与词表内扩展
# ---------------------------------------------------------------------------
def test_bounded_edit_distance_within_limit() -> None:
    assert _bounded_edit_distance("军用", "军民", 1) == 1
    assert _bounded_edit_distance("军用", "军用", 1) == 0


def test_bounded_edit_distance_over_limit_returns_none() -> None:
    assert _bounded_edit_distance("军用", "完全无关的另一个词", 1) is None
    assert _bounded_edit_distance("军", "军民军", 1) is None, "长度差超限应被剪枝"


def test_fuzzy_expand_keeps_only_within_distance() -> None:
    expanded = _handmade_index().fuzzy_expand("军用", field_name="content", max_distance=1)
    assert set(expanded) == {"军民", "军队"}
    assert "完全无关的另一个词" not in expanded
    assert "军用" not in expanded, "自身不算模糊扩展"
    assert expanded[0] == "军民", "同距离时应按词频降序"


def test_fuzzy_expand_returns_empty_when_disabled_or_unknown_field() -> None:
    index = _handmade_index()
    assert index.fuzzy_expand("军用", field_name="content", max_distance=0) == []
    assert index.fuzzy_expand("军用", field_name="不存在的字段", max_distance=1) == []


def test_fuzzy_expand_respects_limit() -> None:
    index = _handmade_index()
    assert len(index.fuzzy_expand("军用", field_name="content", max_distance=1, limit=1)) == 1
    assert len(index.fuzzy_expand("军用", field_name="content", max_distance=1, limit=8)) == 2


def test_fuzzy_query_expands_and_records() -> None:
    """``军用~1`` 在手工索引上应扩展到「军民 / 军队」，命中 r0/r1 并记录扩展结果。"""
    index = _handmade_index()
    evaluation = evaluate_query(index, parse_fulltext_query("军用~1"))
    assert evaluation.rows == {0, 1}
    assert evaluation.fuzzy_expansions == {"军用": ["军民", "军队"]}
    assert evaluation.positive_terms == ["军用", "军民", "军队"]


def test_fuzzy_query_disabled_keeps_exact_term_only() -> None:
    index = _handmade_index()
    evaluation = evaluate_query(index, parse_fulltext_query("军用~1"), enable_fuzzy=False)
    assert evaluation.rows == {0}
    assert evaluation.fuzzy_expansions == {}


# ---------------------------------------------------------------------------
# ⑦ field_text_of 四个字段
# ---------------------------------------------------------------------------
def test_field_text_of_title_prefers_table_title_line() -> None:
    chunk = {"content": "[表标题] 军用领域收入构成\n序号 项目 金额", "section": "第三节 财务"}
    assert field_text_of(chunk, "title") == "军用领域收入构成"


def test_field_text_of_title_falls_back_to_section_then_first_line() -> None:
    assert field_text_of({"content": "普通正文\n第二行", "section": "第二节 管理层讨论"}, "title") \
        == "第二节 管理层讨论"
    assert field_text_of({"content": "abcdefg\n第二行"}, "title") == "abcdefg"


def test_field_text_of_keywords_joins_keywords_and_numbers() -> None:
    chunk = {"content": "2020 年收入 1,234.56 万元，占比 30%", "keywords": ["军用", "收入"]}
    assert field_text_of(chunk, "keywords") == "军用 收入 2020 1,234.56 30%"
    assert field_text_of({"content": "没有数字也没有关键词"}, "keywords") == ""


def test_field_text_of_summary_is_first_220_chars() -> None:
    content = "甲" * 300
    summary = field_text_of({"content": content}, "summary")
    assert len(summary) == 220
    assert summary == content[:220]


def test_field_text_of_content_is_verbatim() -> None:
    assert field_text_of({"content": "原样正文\n第二行"}, "content") == "原样正文\n第二行"
    assert field_text_of({}, "content") == ""
    assert field_text_of({"content": None}, "content") == ""


# ---------------------------------------------------------------------------
# ⑧ FulltextIndex 构建与检索入口
# ---------------------------------------------------------------------------
def test_fulltext_index_build_basic_contract(index: FulltextIndex) -> None:
    assert index.size() == len(ORDER)
    assert index.chunk_ids == list(ORDER)
    assert index.rows_with_term("军用", "content") == {0, 1}
    assert index.rows_with_term("军用", "title") == {0}
    assert index.rows_with_term("不存在的词", "content") == set()
    assert index.rows_with_term("军用", "不存在的字段") == set()
    assert all(index.vocab[field] for field in ("content", "title", "keywords", "summary"))


def test_fulltext_index_to_dict_contract(index: FulltextIndex) -> None:
    payload = index.to_dict()
    assert payload["chunks"] == len(ORDER)
    assert set(payload["fields"]) == {"content", "title", "keywords", "summary"}
    assert payload["fields"]["content"]["terms"] == len(index.vocab["content"]) > 0
    assert payload["built_ms"] >= 0.0


def test_fulltext_index_content_field_reuses_bm25_doc_tokens(bundle: SimpleNamespace) -> None:
    """content 字段直接复用 BM25 的 ``doc_tokens``（零额外分词开销）—— 口径必须一致。"""
    partial = FulltextIndex.build(bundle, fields=("content",))
    assert partial.field_tokens["content"] == [list(tokens) for tokens in bundle.bm25.doc_tokens]
    assert partial.postings["content"]["军用"] == [0, 1]
    assert set(partial.postings) == {"content"}


def test_search_fulltext_bm25_order_and_threshold(bundle: SimpleNamespace,
                                                  index: FulltextIndex) -> None:
    hits = search_fulltext(bundle, "军用", index=index)
    assert [cid for cid, _ in hits] == ["m1", "m2"]
    assert dict(hits) == {"m1": 3.0, "m2": 2.0}
    assert [cid for cid, _ in search_fulltext(bundle, "军用", index=index, score_threshold=2.5)] \
        == ["m1"]
    assert len(search_fulltext(bundle, "军用", index=index, top_k=1)) == 1


def test_search_fulltext_field_boost_uses_max_boost(bundle: SimpleNamespace,
                                                    index: FulltextIndex) -> None:
    """多字段取**命中字段的最大加成**（标题 1.35 > 关键词 1.20），不是连乘。

    m1：标题（section=军用领域）与关键词（["军用","收入"]）都命中 ⇒ 取 1.35，不是 1.35×1.20；
    m2：只有关键词命中 ⇒ 1.20。
    """
    plain = dict(search_fulltext(bundle, "军用", index=index, fields=("content",)))
    all_fields = dict(search_fulltext(bundle, "军用", index=index,
                                      fields=("content", "title", "keywords", "summary")))
    assert all_fields["m1"] == pytest.approx(plain["m1"] * 1.35)
    assert all_fields["m1"] != pytest.approx(plain["m1"] * 1.35 * 1.20)
    assert all_fields["m2"] == pytest.approx(plain["m2"] * 1.20)


def test_search_fulltext_phrase_gets_extra_weight(bundle: SimpleNamespace,
                                                  index: FulltextIndex) -> None:
    conjunction = dict(search_fulltext(bundle, "军用 领域", index=index))
    phrase = dict(search_fulltext(bundle, '"军用领域"', index=index))
    assert set(phrase) == {"m1"}, "短语要求位置相邻，只有 m1 满足"
    assert phrase["m1"] == pytest.approx(conjunction["m1"] * 1.1)


def test_search_fulltext_allowed_ids_and_default_op(bundle: SimpleNamespace,
                                                    index: FulltextIndex) -> None:
    assert [cid for cid, _ in search_fulltext(bundle, "军用", index=index, allowed_ids={"m2"})] \
        == ["m2"]
    assert search_fulltext(bundle, "军用", index=index, allowed_ids={"m3"}) == []
    assert [cid for cid, _ in search_fulltext(bundle, "军用 领域", index=index, default_op="AND")] \
        == ["m1"]


def test_search_fulltext_unknown_field_raises(bundle: SimpleNamespace,
                                              index: FulltextIndex) -> None:
    with pytest.raises(ValueError):
        search_fulltext(bundle, "军用", index=index, fields=("body",))


def test_search_fulltext_empty_query_returns_empty_with_note(bundle: SimpleNamespace,
                                                             index: FulltextIndex) -> None:
    diag: dict[str, Any] = {}
    assert search_fulltext(bundle, "", index=index, diag=diag) == []
    assert diag["query_ast"] is None
    assert "空查询" in diag["notes"]


def test_evaluate_query_empty_node_and_default_fields() -> None:
    evaluation = evaluate_query(_handmade_index(), None)
    assert evaluation.rows == set()
    assert evaluation.notes == ["空查询"]
    assert DEFAULT_FIELDS == ("content",), "工单最低需求口径：默认只检索正文"
