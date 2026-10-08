"""RRF（倒数排名融合）—— **本特性唯一一处纯计算。**

---

## 为什么是 RRF 而不是"加权求和"

余弦相似度是 `[-1, 1]` 的有界量；BM25 分数是**无上界正数**，且随语料规模增长。
把两者直接加权相加，等于把一个归一化过的数与一个没归一化的数相加 ——
结果由 BM25 的分数量纲单方面决定，**权重参数失去意义**。

这个失效方式是**静默的**：检索照常返回、排名看起来合理、没有任何异常，
只是在某个语料规模下悄悄退化成"实际上只有 BM25 在起作用"。

RRF 只使用**名次**，与两路分数的量纲、取值范围、分布全部无关：

    rrf(d) = Σ_{列表 ∈ {语义, 关键词}} 1/(k + rank_list(d))

某个列表未命中该 d 则不计该项。名次从 1 起。

## k=60 的作用（为什么不是 1）

k 越大，头部名次的差距被压得越平。k=60 时第 1 名与第 2 名的贡献是 `1/61` 与
`1/62`，差距很小 —— 这看起来反直觉，但它正是 RRF 稳健的来源：**它不信任任何
一路的名次精度**。两路都靠前的片段会因"两项相加"而胜出，那才是我们要的信号。

它是**平滑常数，不是调参旋钮**（research R5）。

---

⚠️ **本模块 MUST NOT 做任何 I/O。** 纯函数才能被 `selfcheck` 直接断言 ——
而"融合规则是否真的按设计工作"这件事，在真实数据上看不出来（排名只是变得
不那么好，没有异常）。
"""

from . import DEFAULT_RRF_K
from .models import Candidate, CorpusChunk

__all__ = ["merge", "sort_key", "rank_of"]


def rank_of(hits: list[tuple[str, float]]) -> dict[str, int]:
    """把一路的命中列表转成 `{chunk_id: 名次}`，名次从 1 起。

    ⚠️ 名次是**列表位置**，不是排名的并列名次 —— 即使两条分数相同，
    也按列表顺序给 1、2。这样 RRF 才是确定性的（SC-006）。

    这里的 `hits` 假定已由上游按分数降序排好（`store.search_semantic` 与
    `lexical.search` 都保证这一点）。本函数**不再排序** —— 再排一次等于在
    两个地方实现同一个约定，而它们迟早会不一致。
    """

    return {chunk_id: i for i, (chunk_id, _score) in enumerate(hits, start=1)}


def merge(
    semantic_hits: list[tuple[str, float]],
    lexical_hits: list[tuple[str, float]],
    chunks: dict[str, CorpusChunk],
    k: int = DEFAULT_RRF_K,
) -> list[Candidate]:
    """融合两路命中，返回按 RRF 降序的候选列表。

    参数：
        semantic_hits:  `[(chunk_id, 余弦)]`，已按余弦降序。
        lexical_hits:   `[(chunk_id, BM25)]`，已按 BM25 降序。
        chunks:         `{chunk_id: CorpusChunk}` —— 语料查表。
        k:              RRF 平滑常数。

    返回的 `Candidate` 保留两路的**原始分数与原始名次**。这不是为了调试方便，
    而是因为融合规则的有效性**只能**靠回看两路名次来验证。

    三处刻意的不变量：

    1. **去重发生在融合内部**，不在取前 top_k 之后。否则重复项会先占用名额再被
       丢弃，实际返回条数少于 top_k。
    2. **只出现一次的候选保留两路名次**（本就如此：`rank_of` 是查表，
       同一 chunk 在两路各查到一次，两个名次都留着）。
    3. **查不到的 chunk_id 被丢弃而不是报错**。理论上不该发生（两路都从同一份
       语料出发），但真发生时，"少一条结果"远好于"整个提问失败"。丢弃会记在
       调用方的日志里（`service` 比对候选数与语料数）。
    """

    sem_rank = rank_of(semantic_hits)
    lex_rank = rank_of(lexical_hits)
    sem_score = dict(semantic_hits)
    lex_score = dict(lexical_hits)

    # 两路的并集，按 chunk_id 去重。用 dict 而非 set 是为了让遍历顺序稳定
    # （先语义路出现顺序、再关键词路的补集），这样同分时的排序输入是确定的。
    merged_ids: dict[str, None] = {}
    for chunk_id, _ in semantic_hits:
        merged_ids.setdefault(chunk_id, None)
    for chunk_id, _ in lexical_hits:
        merged_ids.setdefault(chunk_id, None)

    candidates: list[Candidate] = []
    for chunk_id in merged_ids:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            # 见不变量 3。
            continue

        s_rank = sem_rank.get(chunk_id)
        l_rank = lex_rank.get(chunk_id)

        score = 0.0
        if s_rank is not None:
            score += 1.0 / (k + s_rank)
        if l_rank is not None:
            score += 1.0 / (k + l_rank)

        candidates.append(
            Candidate(
                chunk=chunk,
                cosine=sem_score.get(chunk_id),
                semantic_rank=s_rank,
                bm25=lex_score.get(chunk_id),
                lexical_rank=l_rank,
                rrf_score=score,
            )
        )

    candidates.sort(key=sort_key)
    return candidates


def sort_key(cand: Candidate) -> tuple[float, float, str]:
    """排序键：`(-rrf_score, -cosine, chunk_id)`。升序排列即得目标顺序。

    末位的 `chunk_id` 是**必需的**，不是保险：RRF 分是分数级别的浮点数，
    两路名次相同的候选会得到**完全相同**的 RRF 分（比如都在两路排第 2 名）。
    没有末位键时 `list.sort` 虽稳定，但稳定性依赖输入顺序，而输入顺序依赖
    语料拉取顺序 —— 那是个不该被排名的正确性依赖的东西。

    加上 `chunk_id` 后，**相同入参必然得到逐字节相同的输出**（SC-006）。

    余弦取负同样是为了升序：高余弦应排在前面。关键词路独有命中的余弦是
    `None`，按 `0.0` 参与比较 —— 与 `RetrievedPassage.from_candidate` 的
    取值一致，否则"排序依据"与"展示分数"会是两回事。
    """

    return (-cand.rrf_score, -(cand.cosine or 0.0), cand.chunk.chunk_id)
