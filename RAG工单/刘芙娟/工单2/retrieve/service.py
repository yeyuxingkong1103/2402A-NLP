"""检索编排。**本包唯一对外的检索入口。**

    service.search(question, query_vector, top_k, threshold) -> RetrievalResult

这是 `docs/05` §4.2 的 I-05：服务端（`backend/api/stream.py`）与 CLI
（`backend/retrieve_search.py`）**共用同一个函数**（FR-023）。

索引的构建与生命周期在 `bundle.py`；本模块只管"给一条问题怎么检索"。

---

## 为什么 `search()` 不接收 config 对象

签名必须与 `docs/05` §4.2 逐字一致（只有 `question` / `query_vector` /
`top_k` / `threshold`）。`RETRIEVAL_CANDIDATES` / `RRF_K` / `LEXICAL_ADMIT_RANK`
这些**索引侧的构建参数**因此存在 `IndexBundle` 里，由 `build_index()` 在启动期
一次性固定 —— 这也符合它们的性质：它们在进程生命周期内不会变。

## 为什么只有一条编排实现

`search()` 与 `search_traced()` 走的是**同一个函数体**，差别只在后者多返回一份
内部细节。若让两条路径各自编排，CLI 验证的就不再是服务端跑的那条链路 ——
而 CLI 存在的全部意义就是验证它。
"""

from __future__ import annotations

import logging
import time

from . import (
    STATE_BELOW_THRESHOLD,
    STATE_LEXICAL_ONLY,
    STATE_NO_CANDIDATES,
    STATE_OK,
)
from . import fuse, lexical, store
from .bundle import IndexBundle, SearchTrace, get_index
from .models import Candidate, RetrievalResult, RetrievedPassage

logger = logging.getLogger(__name__)

__all__ = ["search", "search_traced"]


def search(
    question: str,
    query_vector: list[float],
    top_k: int,
    threshold: float,
) -> RetrievalResult:
    """`docs/05` §4.2 的 I-05。**签名 MUST 与契约逐字一致。**

    失败时抛 `RetrievalError`（Milvus 不可达、维度不符）；
    **检索为空是正常返回值**，不抛异常（`docs/05` §4.2 约束 2）。
    """

    return search_traced(question, query_vector, top_k, threshold).result


def search_traced(
    question: str,
    query_vector: list[float],
    top_k: int,
    threshold: float,
    *,
    index: IndexBundle | None = None,
    skip_semantic: bool = False,
    skip_lexical: bool = False,
) -> SearchTrace:
    """与 `search()` 同一条实现，额外返回两路名次与统计。**CLI 用这个。**

    ⚠️ `skip_semantic` / `skip_lexical` **仅供 CLI 标定**（contracts/cli.md §4）。
    它们让"混合比单路好多少"可测量。**服务端 MUST NOT 使用** ——
    单向降级正是 spec 的 Edge Case 禁止的静默失效，那种情况必须走
    `RetrievalError` 的显式路径，而不是在这里悄悄跳过。
    """

    started = time.monotonic()
    bundle = index or get_index()

    # ---- 两路检索 ----
    #
    # 两路各自按**自己的**相关性降序取 candidates 条。不要在取候选时就融合 ——
    # 那会把 RRF 降级成"比大小"，而两路的分数量纲本来就不可比（见 fuse.py）。
    semantic_hits: list[tuple[str, float]] = []
    if not skip_semantic:
        semantic_hits = store.search_semantic(
            bundle.client, bundle.collection, query_vector, bundle.candidates
        )

    lexical_hits: list[tuple[str, float]] = []
    if not skip_lexical:
        lexical_hits = lexical.search(bundle.lexical, question, bundle.candidates)

    # ---- RRF 融合（按 chunk_id 去重在这一步内部完成）----
    fused = fuse.merge(semantic_hits, lexical_hits, bundle.chunks_by_id, bundle.rrf_k)

    # ---- 回填查询词覆盖率 ----
    #
    # 融合是纯函数、不认识查询，因此覆盖率在这里补上（`fuse.merge` 的 docstring
    # 说明了为何不放在它里面）。只对候选算，不对整个语料算。
    terms = set(lexical.tokenize(question))
    for cand in fused:
        cand.matched_ratio = lexical.coverage(bundle.lexical, cand.chunk.chunk_id, terms)

    # ---- 阈值裁定（Q4 裁决 / research R7 + 2026-09-28 的覆盖率补充）----
    #
    # 先去重融合、再过滤：[1:top_k] 必须发生在过滤**之后**，否则被过滤掉的
    # 候选会先占掉名额，实际返回条数少于 top_k。
    selected = [
        cand
        for cand in fused
        if _admitted(cand, threshold, bundle.admit_rank, bundle.min_coverage)
    ][:top_k]

    result = _build_result(selected, fused, threshold)

    trace = SearchTrace(
        result=result,
        fused=fused,
        semantic_n=len(semantic_hits),
        lexical_n=len(lexical_hits),
        selected=selected,
        state=_classify(fused, selected, result),
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )

    _log_trace(question, trace, threshold)
    return trace


def _admitted(
    cand: Candidate, threshold: float, admit_rank: int, min_coverage: float
) -> bool:
    """候选是否纳入最终结果。

        纳入 ⟺ (余弦 ≥ 阈值)
               或 (BM25 名次 ≤ admit_rank 且 查询词覆盖率 ≥ min_coverage)

    ⚠️ 这是本特性对 `docs/05` §4.2 的**契约扩展**（Q4 裁决 + 2026-09-28 的
    覆盖率补充），必须回写该文档。

    **为什么要扩展**：原文只设想单路检索，阈值只作用于余弦。混合检索下若照
    字面执行，用户照抄罕见术语时 BM25 的精确命中会被判为不相关并走拒答 ——
    而"照抄术语能命中"（US1）恰恰是本特性存在的理由。

    **为什么准入还需要覆盖率**：只有名次条件时，无关提问会被放行 ——
    BM25 对任何查询都必然返回前 N 名。实测「今天晚饭吃什么好」的 top-3
    覆盖率仅 0.20，语义最高余弦 0.5198 < 阈值 —— 两个信号同时判它不相关，
    正是我们要的。而「氢氯噻嗪」（术语照抄）覆盖率 1.00，照常准入。
    见 `__init__.py` 的 DEFAULT_LEXICAL_MIN_COVERAGE（那里也记了被否掉的
    另外两个判据：IDF 门槛与余弦下限，都被实测证明分不开）。

    取 `admit_rank = 3` 而非 1：允许关键词路的前几名都过 —— 只放第 1 名会让
    "原文在两段里各说一半"这种情况只召回其中一半。
    """

    if cand.cosine is not None and cand.cosine >= threshold:
        return True
    return (
        cand.lexical_rank is not None
        and cand.lexical_rank <= admit_rank
        and cand.matched_ratio >= min_coverage
    )


def _build_result(
    selected: list[Candidate], fused: list[Candidate], threshold: float
) -> RetrievalResult:
    """组装契约模型。三字段语义见 contracts/retrieval.md §2。

    ---

    ⚠️ **`below_threshold` 的定义在实现期被修正过一次**，值得记下来：

    最初的实现照抄了一个看起来更简洁的公式
    `below_threshold = (not is_empty) and (top_score < threshold)`。
    它在"有候选但一条都没过"这一档上恒为 `False` —— 于是
    **`is_empty=True` 的两种情形（完全没命中 / 命中但不够像）拿到同一个
    `below_threshold`**，而 FR-013 明确要求二者可区分。

    失效现场很具体：无关提问的日志打出
    `is_empty=True below_threshold=False ... state=below_threshold` ——
    字段名与状态名互相矛盾，这才被发现。

    正确语义是**独立于 `is_empty`** 的：

        below_threshold = 有候选存在，但其中最高的余弦仍低于阈值

    `fused` 参数因此是必需的 —— 单看 `selected` 无法区分"没候选"与"候选都不达标"。
    """

    passages = [RetrievedPassage.from_candidate(cand) for cand in selected]

    if passages:
        # 关键词路独有命中的余弦是 None，按 0.0 参与比较 —— 与 `fuse.sort_key`
        # 的取值一致。不一致会让"排序依据"与"展示分数"变成两回事。
        top_score = max(passage.score for passage in passages)
        return RetrievalResult(
            passages=passages,
            top_score=top_score,
            is_empty=False,
            below_threshold=top_score < threshold,
        )

    # 无结果。`top_score` 按契约取 None（空集时），但 `below_threshold` 要看
    # **候选层**：一条候选都没有 ≠ 有候选但全不达标。
    if not fused:
        return RetrievalResult(
            passages=[], top_score=None, is_empty=True, below_threshold=False
        )

    best_cosine = max(
        (cand.cosine for cand in fused if cand.cosine is not None), default=None
    )
    return RetrievalResult(
        passages=[],
        top_score=None,
        is_empty=True,
        # 候选全是关键词路独有命中（余弦皆为 None）时 `best_cosine is None` ——
        # 那显然没达到相似度阈值，判为 below_threshold。
        below_threshold=best_cosine is None or best_cosine < threshold,
    )


def _classify(fused: list[Candidate], selected: list[Candidate], result: RetrievalResult) -> str:
    """判定检索状态（data-model.md §6 的前四种；后两种是异常路径）。

    四态的区分是有意义的（FR-013）：把"检索坏了"记成"知识库里没有"，会让一次
    故障看起来像一次正常的空结果，而用户会换个问法继续试。
    """

    if not fused:
        return STATE_NO_CANDIDATES
    if not selected:
        return STATE_BELOW_THRESHOLD
    if result.below_threshold:
        # 结果非空但全部靠关键词纳入 —— "知识库里有这个词，语义模型没理解它"。
        return STATE_LEXICAL_ONLY
    return STATE_OK


def _log_trace(question: str, trace: SearchTrace, threshold: float) -> None:
    """FR-019 的结构化日志。

    ⚠️ MUST NOT 记录：`query_vector` 的数值、片段全文（可达数千字符，且已可
    通过 `chunk_id` 回查）。问题原文也不记 —— 它在 S8 的留存文件里，
    这里再记一遍等于把用户输入写进服务日志。只记长度。

    `lexical_only` 用 WARNING 级：它是"模型/语料不匹配"的早期征兆，
    值得被看见，而 INFO 级的正常日志不会有人定期看。
    """

    top_cosine = trace.result.top_score
    top_rrf = trace.fused[0].rrf_score if trace.fused else 0.0

    message = (
        "检索完成 is_empty=%s below_threshold=%s semantic=%d lexical=%d "
        "fused=%d returned=%d top_cosine=%s top_rrf=%.5f elapsed_ms=%d "
        "state=%s question_len=%d threshold=%.4f"
    )
    args = (
        trace.result.is_empty,
        trace.result.below_threshold,
        trace.semantic_n,
        trace.lexical_n,
        len(trace.fused),
        len(trace.result.passages),
        "None" if top_cosine is None else "%.4f" % top_cosine,
        top_rrf,
        trace.elapsed_ms,
        trace.state,
        len(question),
        threshold,
    )

    if trace.state == STATE_LEXICAL_ONLY:
        logger.warning(message, *args)
    else:
        logger.info(message, *args)
