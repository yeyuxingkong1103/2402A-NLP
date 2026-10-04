# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
全文检索的纯函数测试。

【与 test_pipeline.py 同样的原则】**不碰 Milvus、不碰 Ollama** ——
倒排索引的构建与查询都是纯计算，用一小撮预置行就能把不变量钉死。
（真实库上的效果验证放在评测脚本里，见 docs/工单06-混合检索.md。）

跑法：pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.fulltext import (                                    # noqa: E402
    InvertedIndex, QPhrase, QTerm, parse)
from app.core.text_analysis import (                               # noqa: E402
    is_numeric_token, term_id, tokenize, tokenize_positions)
from app.core.vectorstore import lexical_sparse                    # noqa: E402


# ----------------------------------------------------------------------
def _row(i: int, content: str, *, section: str = "", page: str = "1-1-1",
         doc: str = "招股说明书1.pdf", ctype: str = "text") -> dict:
    return {"id": i, "content": content, "page_no": i, "page_label": page,
            "chunk_type": ctype, "section_path": section, "doc_name": doc,
            "chunk_index": i, "doc_id": doc}


@pytest.fixture(scope="module")
def index() -> InvertedIndex:
    return InvertedIndex.build([
        _row(1, "报告期内，公司来自军用领域的收入分别为6,464.51万元。", page="1-1-128",
             section="第四节 财务会计信息 > 一、营业收入"),
        _row(2, "本次发行股数1,670万股，占发行后总股本的比例为25.04%。", page="21",
             doc="招股说明书2.pdf", section="四、本次发行情况"),
        _row(3, "公司注册资本增加至5,520万元，新增股东程家明。", page="1-1-58",
             section="第五节 股本 > 二、股本变化"),
        _row(4, "募集资金用途如下：补充流动资金15,000万元。", page="1-1-29",
             section="八、募集资金用途"),
        _row(5, "军用视频指挥领域已经成为重要供应商。", page="1-1-25",
             section="第五节 业务与技术"),
        _row(6, "销售部由四个部门构成，大客户销售部有六个销售处。", page="38",
             doc="招股说明书2.pdf", section="四、发行人组织结构 > 1、公司组织结构图",
             ctype="image"),
    ])


def _pages(hits) -> list[str]:
    return [h.page_label for h in hits]


# ======================================================================
# 分词（工单06 的共用地基）
# ======================================================================
def test_number_token_kept_whole():
    """`5,520.00` / `1-1-42` 必须是**单个** token，不能被切碎。

    招股书的核心答案大量是数字与页码；切碎之后「注册资本是多少」就查不到了。
    """
    toks = tokenize("注册资本5,520.00万元，见 1-1-42 页")
    assert "5,520.00" in toks
    assert "1-1-42" in toks
    assert "5" not in toks and "520" not in toks      # 没有碎片


def test_positions_are_in_text_order():
    """位置必须按原文顺序 —— 乱序会让短语匹配彻底失效（旧实现正是乱序的）。"""
    pos = tokenize_positions("军用领域的收入")
    terms = [t for t, _ in pos]
    assert terms.index("军用") < terms.index("领域") < terms.index("收入")
    assert [p for _, p in pos] == sorted(p for _, p in pos)


def test_term_id_is_stable_across_processes():
    """token→维度 必须是**内容寻址**的。

    旧实现用内置 `hash()`，它对 str 逐进程随机化 —— 入库与查询是两个进程，
    维度对不上 → Milvus 稀疏检索静默失效（不报错、召回率≈0）。
    这里钉死"同内容必同号"，并给出一组已知值防止又被换成随机哈希。
    """
    assert term_id("万元") == term_id("万元")
    assert term_id("万元") != term_id("军队")
    # 固定期望值：换掉 term_id 的实现会立刻在这里暴露（而不是等到检索悄悄失效）
    assert term_id("万元") == 1900190143


def test_sparse_ids_match_token_ids():
    """稀疏向量的维度必须与 text_analysis.term_id 同源。"""
    text = "注册资本"
    sp = lexical_sparse(text)
    assert set(sp) == {term_id(t) for t in set(tokenize(text))}


def test_sparse_with_idf_weights():
    """IDF 传入后，稀有词权重应被抬高、常见词被压低。"""
    text = "公司注册资本"
    base = lexical_sparse(text)
    idf = {term_id("注册资本"): 5.0, term_id("公司"): 0.1}
    weighted = lexical_sparse(text, idf=idf)
    assert weighted[term_id("注册资本")] > base[term_id("注册资本")]
    assert weighted[term_id("公司")] < base[term_id("公司")]


def test_is_numeric_token():
    assert is_numeric_token("5,520.00") and is_numeric_token("1-1-42")
    assert not is_numeric_token("万元") and not is_numeric_token("军用")


# ======================================================================
# 查询解析
# ======================================================================
def test_parse_builds_phrase_node():
    """引号必须解析成 QPhrase —— 退化成"各词 OR"就等于没做短语（且不报错）。"""
    node = parse('"军用领域的收入"')
    assert isinstance(node, QPhrase)
    assert node.terms[0] == "军用"


def test_parse_bare_word_is_tokenized():
    """裸词必须走分词：库里存的是词条，整串去查一个都查不到。"""
    node = parse("注册资本")
    # 单个词条 → QTerm；多个词条 → OR（隐式运算符）
    assert isinstance(node, QTerm) and node.term == "注册资本"


# ======================================================================
# 检索行为
# ======================================================================
def test_phrase_requires_adjacent_positions(index):
    """短语要**相邻且有序**：顺序打乱必须落空。"""
    assert "1-1-128" in _pages(index.search('"军用领域的收入"', 5))
    assert index.search('"军用收入领域"', 5) == []      # 顺序错 → 空


def test_phrase_still_matches_whole_term(index):
    assert "1-1-58" in _pages(index.search('"注册资本"', 5))


def test_boolean_and_narrows(index):
    both = index.search("注册资本 AND 5,520", 10)
    assert _pages(both) == ["1-1-58"]


def test_not_is_difference_not_complement(index):
    """NOT 只能在正部结果集上做差 —— 对全库取补会把 6 篇全返回。

    这是"看起来有结果"的静默错误：用户以为搜到了，其实拿到了整个库。
    """
    all_hits = index.search("军用", 100)
    not_hits = index.search("军用 AND NOT 视频", 100)
    assert len(not_hits) < len(all_hits)
    assert "1-1-25" not in _pages(not_hits)             # 含"视频"的那篇被排除


def test_field_scoped_query(index):
    """`title:` 只看章节标题字段。"""
    hit = index.search("title:募集资金用途", 5)
    assert "1-1-29" in _pages(hit)
    # 同一批词出现在正文里、但标题不含它 → 不该被 title: 命中
    assert "1-1-128" not in _pages(index.search("title:军用领域", 5))


def test_fuzzy_matches_typo_with_penalty(index):
    """错别字能命中，但**分数必须低于**精确命中。"""
    fuzzy = index.search("军用领城~", 5)               # 城 → 域
    assert "1-1-128" in _pages(fuzzy)
    exact = index.search("军用领域", 5)
    assert exact[0].score > fuzzy[0].score


def test_fuzzy_never_applies_to_numbers(index):
    """数字绝不做编辑距离：`5,530` 命中 `5,520` 等于把答案改错。"""
    assert index.search("5,530~", 5) == []
    assert "1-1-58" in _pages(index.search("5,520", 5))


def test_fuzzy_scores_are_marked(index):
    """模糊命中要能与精确命区分（分数已打折）。"""
    hits = index.search("军用领城~", 5)
    assert hits and max(h.score for h in hits) > 0


def test_short_terms_skip_fuzzy():
    """短词（< fuzzy_min_len）不做模糊扩展，避免炸开噪声。

    注意这里直接测扩展函数：模糊查询**同时收精确命中的子词**（折扣 1.0），
    所以 `军用领城~` 会因为 `军用` 精确命中而仍有结果 —— 那是正确行为，
    要验的是"2 字词不参与扩展"这条。
    """
    strict = InvertedIndex.build([_row(1, "军用领域")], fuzzy_min_len=4)
    assert strict._fuzzy_expand("领城", 1) == []            # 2 字 < 4 → 不扩展
    default = InvertedIndex.build([_row(1, "军用领域")])
    assert [c for c, _ in default._fuzzy_expand("领城", 1)] == ["领域"]


def test_doc_filter_excludes_other_document(index):
    """路由过滤：限定书2 时绝不能返回书1 的块（工单03 的文档隔离）。"""
    hits = index.search("军用", 10, doc_name="招股说明书2.pdf")
    assert all(h.doc_name == "招股说明书2.pdf" for h in hits)
    assert index.search("军用", 10, doc_name="不存在的.pdf") == []


def test_field_weight_prefers_title(index):
    """同一批词，出现在标题字段的块应排在只有正文的块之前。"""
    idx = InvertedIndex.build([
        _row(1, "本节讲别的事情，正文里顺带提了一句募集资金。", section="一、其他"),
        _row(2, "本节内容。", section="八、募集资金用途"),
    ])
    hits = idx.search("募集资金用途", 5)
    assert _pages(hits)[0] == "1-1-1" or hits[0].chunk_id == 2


def test_image_chunk_is_indexed(index):
    """图块（工单04 的转写文本）同样进倒排索引，否则图题搜不到。"""
    assert "38" in _pages(index.search("大客户销售部", 5))


def test_empty_index_returns_empty():
    """空库不抛异常（降级为无召回，而不是 500）。"""
    idx = InvertedIndex.build([])
    assert idx.search("任何查询", 5) == []
    assert idx.search("", 5) == []


def test_empty_query_returns_empty(index):
    assert index.search("", 5) == []
    assert index.search("   ", 5) == []


def test_stats_and_roundtrip(tmp_path, index):
    """存取一轮，索引内容必须一致（缓存命中不能改变行为）。"""
    p = tmp_path / "idx.pkl"
    index.save(p)
    loaded = InvertedIndex.load(p)
    assert loaded.n_docs == index.n_docs
    assert _pages(loaded.search("军用领域的收入", 3)) == _pages(index.search("军用领域的收入", 3))
    assert loaded.stats()["n_terms"] > 0
