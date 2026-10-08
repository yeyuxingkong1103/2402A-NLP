"""自检用的断言与构造语料。**`selfcheck.py` 跑的就是这里列出的 10 条。**

与 `selfcheck.py` 的分界是「跑什么」与「怎么跑」：本模块是前者（断言本身 +
它们依赖的构造语料），那边是后者（逐条执行、收集失败、渲染结果）。

---

## 为什么用构造语料而不是真实数据

依赖真实数据的话，数据一变这些断言就跟着失去意义 —— 而它们要验证的是**公式**。
`SEED` 是四条手工写死的语料，全文可见、可推演、不随库内容变化。

## 为什么这些断言值得存在（本项目没有 pytest）

research R11：pytest 未安装，且引入测试框架是独立决策，不该夹带在检索特性里。
而 BM25 打分与 RRF 融合是本特性唯一"算错了也不会报错"的部分 —— 排名只是变得
不那么好，没有任何异常。它们的断言因此必须有一个可执行的落点。
"""

from __future__ import annotations

from . import (
    DEFAULT_ADMIT_RANK,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_RRF_K,
)
from . import fuse, lexical
from .models import Candidate, CorpusChunk
from .selftest_data import seed_corpus, seed_map

__all__ = ["CHECKS"]




def _sc_idf_positive() -> None:
    """IDF 必须恒正。

    `df > N/2` 时变负的实现会让"文档包含查询词"反而**降低**分数 —— 那类实现
    在语料小、查询词常见时（正是本项目当前的状态：65 chunks）会给出反直觉的
    排序，且不报错。
    """

    for df, total in ((1, 10), (5, 10), (9, 10), (10, 10)):
        value = lexical._idf(df, total)
        assert value > 0.0, "df=%d n=%d 时 IDF 非正：%r" % (df, total, value)


def _sc_tf_monotone() -> None:
    """同一文档内词频越高，BM25 分数越高（`k1` 饱和但不反转）。

    构造两份只差一个词频的语料来对比 —— 在同一次查询里比较两个文档是测不出
    词频项的（它们还差着文档长度与 df）。
    """

    def corpus_with(repeats: int) -> list[CorpusChunk]:
        return [
            CorpusChunk(chunk_id="t:0000", text="高血压 " * repeats + "其他内容",
                        file_name="t.pdf", page_start=1, page_end=1, block_type="text"),
            CorpusChunk(chunk_id="t:0001", text="无关的另一些文字",
                        file_name="t.pdf", page_start=1, page_end=1, block_type="text"),
        ]

    def score_of(repeats: int) -> float:
        index = lexical.build_index(corpus_with(repeats))
        return dict(lexical.search(index, "高血压", 10)).get("t:0000", 0.0)

    low, high = score_of(1), score_of(3)
    assert high > low, "词频 1 → 3 时分数未上升：%.6f → %.6f" % (low, high)

    hits = lexical.search(lexical.build_index(seed_corpus()), "高血压 糖尿病", 10)
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1)), (
        "BM25 结果未按分数降序：%r" % hits
    )


def _sc_empty_corpus() -> None:
    """空语料是正常状态，不是错误（spec 的 Edge Case）。"""

    index = lexical.build_index([])
    assert index.n_docs == 0
    assert index.avgdl == 1.0, "空语料 avgdl 应为 1.0（除零保护），实际 %r" % index.avgdl
    assert lexical.search(index, "任何问题", 10) == [], "空语料应返回空列表"


def _sc_rrf_both_beats_single() -> None:
    """RRF 的核心效果：两路都靠前的片段胜过只在一路靠前的。"""

    merged = fuse.merge(
        [("t:0001", 0.9), ("t:0000", 0.8)],
        [("t:0002", 9.0), ("t:0000", 5.0)],
        seed_map(),
        DEFAULT_RRF_K,
    )
    both = next(c for c in merged if c.chunk.chunk_id == "t:0000")
    singles = [c for c in merged if c.chunk.chunk_id != "t:0000"]
    assert singles, "缺少单路候选"
    assert both.rrf_score > max(c.rrf_score for c in singles), (
        "两路命中的候选未排在单路命中之前：%.5f vs %s"
        % (both.rrf_score, [round(c.rrf_score, 5) for c in singles])
    )


def _sc_rrf_k_smooths() -> None:
    """k 越大，头部名次的差距被压得越平 —— 这正是 RRF 稳健性的来源。"""

    semantic = [("t:0000", 0.9), ("t:0001", 0.8)]
    lexical_hits = [("t:0002", 9.0), ("t:0000", 5.0)]

    def gap(k: int) -> float:
        merged = {c.chunk.chunk_id: c
                  for c in fuse.merge(semantic, lexical_hits, seed_map(), k)}
        return merged["t:0000"].rrf_score - merged["t:0002"].rrf_score

    assert gap(100) < gap(1), "k 增大应让头部差距缩小：%r vs %r" % (gap(100), gap(1))


def _sc_stable_sort() -> None:
    """排序键的逐级裁决：RRF → 余弦 → chunk_id，且同一输入必得同一输出。

    ⚠️ 写这条检查时踩过一个坑，值得记下来：最初的版本假设"两条候选可以拿到
    相同 RRF 分"，于是断言"同分时按 chunk_id 升序"。**那个前提是错的** ——
    名次不同则 RRF 必然不同，所以同一路内的两条候选永远分不出平局。

    真正能打平的情形只有一种：**两路各自的第 1 名**（都是 `1/(k+1)`）。
    因此这里分三层测：先测 RRF 打平时靠余弦决胜，再单独测排序键的末位，
    最后测可复现性。
    """

    # (a) RRF 打平（语义第 1 名 vs 关键词第 1 名）→ 靠余弦决胜
    merged = fuse.merge([("t:0000", 0.5)], [("t:0001", 9.0)], seed_map(), DEFAULT_RRF_K)
    assert merged[0].rrf_score == merged[1].rrf_score, (
        "构造失败：两条的 RRF 应当打平，实际 %r vs %r"
        % (merged[0].rrf_score, merged[1].rrf_score)
    )
    assert merged[0].chunk.chunk_id == "t:0000", (
        "RRF 打平时未按余弦降序：%s" % [c.chunk.chunk_id for c in merged]
    )

    # (b) RRF 与余弦都相同时按 chunk_id 升序 —— 直接测排序键，
    #     因为这种平局在真实的两路名次下构造不出来（见上面的说明）。
    chunks = seed_map()
    later = Candidate(chunk=chunks["t:0002"], cosine=0.5, semantic_rank=1, rrf_score=0.5)
    earlier = Candidate(chunk=chunks["t:0001"], cosine=0.5, semantic_rank=1, rrf_score=0.5)
    assert fuse.sort_key(earlier) < fuse.sort_key(later), (
        "RRF 与余弦都相同时未按 chunk_id 升序：%r vs %r"
        % (fuse.sort_key(earlier), fuse.sort_key(later))
    )

    # (c) 可复现：完全相同的输入必得完全相同的输出（SC-006）
    semantic = [("t:0001", 0.7), ("t:0000", 0.5)]
    lexical_hits = [("t:0003", 1.0), ("t:0002", 0.9)]
    first = [(c.chunk.chunk_id, c.rrf_score)
             for c in fuse.merge(semantic, lexical_hits, seed_map(), DEFAULT_RRF_K)]
    again = [(c.chunk.chunk_id, c.rrf_score)
             for c in fuse.merge(list(semantic), list(lexical_hits), seed_map(), DEFAULT_RRF_K)]
    assert first == again, "相同输入得到不同输出：%r vs %r" % (first, again)


def _sc_dedup() -> None:
    """同一 chunk 两路命中 → 只保留一条，且两个名次都留着。"""

    merged = fuse.merge(
        [("t:0000", 0.9), ("t:0001", 0.7)],
        [("t:0000", 5.0)],
        seed_map(),
        DEFAULT_RRF_K,
    )
    ids = [c.chunk.chunk_id for c in merged]
    assert len(ids) == len(set(ids)), "融合结果出现重复 chunk_id：%s" % ids

    both = next(c for c in merged if c.chunk.chunk_id == "t:0000")
    assert both.semantic_rank == 1 and both.lexical_rank == 1, (
        "去重后未合并两路名次：语义 %r 关键词 %r"
        % (both.semantic_rank, both.lexical_rank)
    )


def _sc_quadrants() -> None:
    """`is_empty` / `below_threshold` / `top_score` 的四象限（data-model.md §1.2）。

    ⚠️ 第二、三档是关键：二者 `is_empty` 都是 `True`，**必须靠
    `below_threshold` 区分**（FR-013）。实现期这里踩过一次坑 —— 当时用的是
    `below_threshold = (not is_empty) and (...)`，让两档拿到了同一个值。
    """
    from .service import _build_result  # noqa: PLC0415 —— 避免顶层循环导入

    # ① 一条候选都没有 → 完全没命中
    nothing = _build_result([], [], 0.6)
    assert nothing.is_empty and nothing.top_score is None, "无候选象限不符：%r" % nothing
    assert not nothing.below_threshold, "无候选时不应判为 below_threshold：%r" % nothing

    # ② 有候选，但全部低于阈值 → 命中但不够像  ← 与 ① 必须可区分
    all_low = fuse.merge([("t:0000", 0.3), ("t:0001", 0.2)], [], seed_map(), DEFAULT_RRF_K)
    below = _build_result([], all_low, 0.6)
    assert below.is_empty and below.top_score is None, "全低象限不符：%r" % below
    assert below.below_threshold, (
        "有候选但全不达标时必须 below_threshold=True —— 否则与「完全没命中」"
        "不可区分（FR-013）：%r" % below
    )

    # ③ 至少一条余弦 >= 阈值 → 正常命中
    high = fuse.merge([("t:0000", 0.9)], [], seed_map(), DEFAULT_RRF_K)
    ok = _build_result(high, high, 0.6)
    assert not ok.is_empty and not ok.below_threshold and ok.top_score == 0.9, (
        "高余弦象限不符：%r" % ok
    )

    # ④ 有结果但全部靠关键词纳入 → 非空 + below_threshold
    only_lex = fuse.merge([], [("t:0000", 9.0)], seed_map(), DEFAULT_RRF_K)
    lexical_only = _build_result(only_lex, only_lex, 0.6)
    assert not lexical_only.is_empty and lexical_only.below_threshold, (
        "仅关键词象限不符（应为非空 + below_threshold）：%r" % lexical_only
    )
    assert lexical_only.passages[0].score == 0.0, (
        "关键词路独有命中的 score 应为 0.0（不是伪造的余弦），实际 %r"
        % lexical_only.passages[0].score
    )


def _sc_coverage_gate() -> None:
    """关键词准入必须同时满足名次与覆盖率（2026-09-28 裁决）。

    **这条断言修的是一个真实缺陷**：只有名次条件时，无关提问会被放行 ——
    BM25 对任何查询都必然返回前 N 名。实测「今天晚饭吃什么好」的 top-3
    覆盖率仅 0.20，而「氢氯噻嗪」这类术语照抄是 1.00。

    这里用构造的 Candidate 直接测 `_admitted`，因为覆盖率是 `service` 在融合后
    回填的，不经过 `fuse.merge`。
    """

    from .service import _admitted  # noqa: PLC0415 —— 避免顶层循环导入

    chunks = seed_map()

    def cand(cosine, lex_rank, ratio) -> Candidate:
        return Candidate(
            chunk=chunks["t:0000"],
            cosine=cosine,
            semantic_rank=1 if cosine is not None else None,
            bm25=1.0 if lex_rank else None,
            lexical_rank=lex_rank,
            matched_ratio=ratio,
        )

    # 语义达标 → 无条件纳入，与覆盖率无关（覆盖率只管关键词准入）
    assert _admitted(cand(0.9, None, 0.0), 0.6, 3, 0.5), "余弦达标却未纳入"

    # 术语照抄：名次靠前 + 覆盖率 1.00 → 纳入
    assert _admitted(cand(0.3, 1, 1.0), 0.6, 3, 0.5), "术语照抄（覆盖率 1.0）被误拦"

    # 无关提问：名次很靠前但覆盖率 0.20 → 拦下（这是修的那个缺陷）
    assert not _admitted(cand(0.5, 1, 0.2), 0.6, 3, 0.5), (
        "覆盖率 0.20 的候选仍被关键词路放行 —— 无关提问会绕过拒答"
    )

    # 名次太靠后 → 拦下，与覆盖率无关
    assert not _admitted(cand(0.3, 9, 1.0), 0.6, 3, 0.5), "名次超线却仍被纳入"

    # 覆盖率函数本身
    index = lexical.build_index(seed_corpus())
    terms = set(lexical.tokenize("高血压 糖尿病"))
    assert lexical.coverage(index, "t:0000", terms) == 0.5, "覆盖率计算不符"
    assert lexical.coverage(index, "t:0000", set()) == 0.0, "空查询词应返回 0.0"


def _sc_tokenize() -> None:
    """分词必须幂等，且索引期与查询期走同一函数（research R2）。"""

    text = "盐酸氨氯地平 5mg ACEI"
    first = lexical.tokenize(text)
    assert first == lexical.tokenize(text), "tokenize 不幂等"
    assert all(t for t in first), "tokenize 产出了空 token"
    assert all(t == t.lower() for t in first), "tokenize 未小写化"
    assert lexical.tokenize("？？？。，、") == [], "纯标点应被过滤掉"
    assert DEFAULT_BM25_K1 > 0 and 0 <= DEFAULT_BM25_B <= 1, "BM25 参数越界"
    assert DEFAULT_ADMIT_RANK >= 1, "准入线应 >= 1"


# 自检清单。顺序即输出顺序，也是从"最底层的公式"到"最上层的结果语义"的顺序 ——
# 前面的错了，后面的结论没有意义。
CHECKS: tuple[tuple[str, object], ...] = (
    ("SC-1  BM25 IDF 恒正", _sc_idf_positive),
    ("SC-2  BM25 分数对词频单调不减", _sc_tf_monotone),
    ("SC-3  空语料不抛异常", _sc_empty_corpus),
    ("SC-4  RRF：两路命中 > 单路命中", _sc_rrf_both_beats_single),
    ("SC-5  RRF：k 增大时头部差距缩小", _sc_rrf_k_smooths),
    ("SC-6  同分排序稳定（带 chunk_id 末位）", _sc_stable_sort),
    ("SC-7  同一 chunk 两路命中只保留一条并合并名次", _sc_dedup),
    ("SC-8  below_threshold 四象限", _sc_quadrants),
    ("SC-9  tokenize 幂等且过滤纯标点", _sc_tokenize),
    ("SC-10 关键词准入受覆盖率约束（无关提问被拦下）", _sc_coverage_gate),
)
