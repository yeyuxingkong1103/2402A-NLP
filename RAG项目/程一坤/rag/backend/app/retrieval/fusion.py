"""RRF（Reciprocal Rank Fusion）融合与去重：合并关键词路与向量路结果，两路都命中者靠前。

公式 score = Σ 1/(k + rank)：k 为常数（标准取值 60），rank 为该条目在某路
结果中的排名（从 1 开始），对每个 chunk_key 累加它在所有结果列表中的得分。
为什么用 RRF 而不是分数加权：向量余弦分与 BM25 分数量纲不可比，RRF 只用
"排名"融合、免疫分数分布差异。
本文件只负责"多路结果 → 单一有序列表"的纯内存打分与去重，不触碰数据库；
父块解析与归并（子块→父块升级、同父块保留最高分）独立在
app/retrieval/parent_collapse.py。

融合结果的三条保证：
- 每个 chunk_key 只出现一次，score 与 fusion_score 均为累加后的 RRF 分；
- sources 为该 chunk 被哪些路命中的并集（排序后的元组）；
- 同输入同输出：并列分数按固定顺序处理，输出确定性可复现。

为什么父块归并不在本文件：融合是"多路→一路"的分数问题（发生在重排前），
归并是"子块→父块"的内容结构问题（发生在重排后）；触发时机与职责都不同，
拆成两个文件各自单一职责。
RankedItem 定义在本文件：各阶段统一载体，归并模块 import 复用同一类型。
"""
from __future__ import annotations

from dataclasses import dataclass

# 只做纯内存计算：无 DB / 无检索器依赖，单测可直接构造输入验证打分与排序
RRF_K = 60  # RRF 常数，标准取值 60；k 越大排名差异越平缓
# 直观感受 k=60 的量级：rank1≈1/61、rank10≈1/70，单路内排名差约 15%，
# 两路都命中（分数相加）才能显著拉开差距——这正是"两路都命中者靠前"的来源


@dataclass
class RankedItem:
    """排序后的检索结果条目（召回/融合/重排/归并各阶段的统一载体）。

    Args:
        chunk_key: 分块唯一标识；score: 原始分数（召回分/重排分）
        source: 来源标识（vector/keyword/rerank/fused）

    字段按用途分四组：
    1) 身份与排序：chunk_key / score / source
    2) 正文与引用展示：content、document_title、条号/款号/项号等
    3) 来源与各阶段分数：sources 及 vector/keyword/recall/fusion/rerank 分
    4) 法规身份与时效：document_type、jurisdiction、生效失效日期、is_current 等

    其余可选字段保留原始检索结果的元数据，融合/归并时原样传递，
    保证引用字段（条号 article_number / 款号 paragraph_number /
    项号 item_number 等引用必需字段）不丢失。
    """
    # —— 身份与排序字段 ——
    chunk_key: str
    score: float
    source: str
    # —— 正文与引用展示字段：组装上下文、生成引用时直接使用 ——
    content: str | None = None
    document_title: str | None = None
    article_number: str | None = None
    source_url: str | None = None
    parent_chunk_key: str | None = None
    # —— 来源合并字段：被哪些路命中；融合/归并时取并集，供引用与调试追溯 ——
    sources: tuple[str, ...] = ()
    # —— 链路各阶段的分数：随条目传递，便于归并排序时按"最可信分"取舍 ——
    vector_score: float | None = None
    keyword_score: float | None = None
    recall_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    # —— 法规身份字段：术语与 docs/CONTEXT.md 一致 ——
    paragraph_number: str | None = None
    item_number: str | None = None
    document_type: str | None = None
    jurisdiction: str | None = None
    # —— 时效字段：is_current 现行有效性判定的输入 ——
    effective_date: int | str | None = None
    expiration_date: int | str | None = None
    issuing_authority: str | None = None
    is_current: bool | None = None
    document_id: str | None = None

    def __post_init__(self):
        """允许传入任意额外字段，用于保留检索结果的完整元数据。"""


def fuse_results(result_lists: list[list[RankedItem]], k: int = RRF_K) -> list[RankedItem]:
    """使用 RRF 融合多路检索结果。

    Args:
        result_lists: 多路检索结果，每路是一个按相关度降序排列的列表；k: RRF 常数，默认 60

    Returns:
        按 RRF 分数降序的结果列表，相同 chunk_key 只出现一次

    算法（细节见函数内注释）：每路按 1/(k+rank) 计分 → 按 chunk_key 累加
    → 降序排列 → 同 chunk_key 去重并保留首见的完整元数据。
    复杂度 O(总条数)；k 越大不同排名间的分差越小，多路同时命中的加权优势越弱。
    """
    # 空/全空输入直接返回空表：上游按空结果短路，不再触发后续累加与排序
    if not result_lists or all(len(lst) == 0 for lst in result_lists):
        return []

    rrf_scores: dict[str, float] = {}
    # 去重取"首见"而非"最高分"：各路已按相关度排序，首见元数据最完整且避免并列抖动
    first_occurrence: dict[str, RankedItem] = {}
    # sources_by_key/vector_scores/keyword_scores：跨路聚合桶，第三步回填
    sources_by_key: dict[str, set[str]] = {}
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}

    # —— 第一步：逐路逐条按排名计 RRF 分，按 chunk_key 累加 ——
    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            chunk_key = item.chunk_key
            rrf_score = 1.0 / (k + rank)
            rrf_scores[chunk_key] = rrf_scores.get(chunk_key, 0.0) + rrf_score
            sources_by_key.setdefault(chunk_key, set()).add(item.source)
            # 分数值只留一份：同 chunk 在同路多次出现时以最后一次为准（各路内部已去重）
            if item.vector_score is not None:
                vector_scores[chunk_key] = item.vector_score
            if item.keyword_score is not None:
                keyword_scores[chunk_key] = item.keyword_score

            # 首见登记：仅在该 chunk 第一次出现时记录（后续重复出现跳过）
            if chunk_key not in first_occurrence:
                first_occurrence[chunk_key] = item

    # —— 第二步：按累加 RRF 分降序得到全局排序 ——
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    # 输出容器：与输入解耦的新列表，第三步逐条重建填充
    fused_results = []
    # —— 第三步：以首见元数据重建条目（元数据不丢、来源取并集） ——
    # 重建新对象而非复用原条目：融合分/来源与各路原始条目解耦，避免引用被共享修改
    for chunk_key in sorted_keys:
        original_item = first_occurrence[chunk_key]
        fused_item = RankedItem(
            chunk_key=chunk_key,
            score=rrf_scores[chunk_key],  # 使用 RRF 融合分数
            source="fused",  # 标记为融合结果
            content=original_item.content,
            document_title=original_item.document_title,
            article_number=original_item.article_number,
            source_url=original_item.source_url,
            parent_chunk_key=original_item.parent_chunk_key,
            sources=tuple(sorted(sources_by_key[chunk_key])),
            vector_score=vector_scores.get(chunk_key),
            keyword_score=keyword_scores.get(chunk_key),
            recall_score=original_item.recall_score,
            fusion_score=rrf_scores[chunk_key],
            rerank_score=original_item.rerank_score,
            paragraph_number=original_item.paragraph_number,
            item_number=original_item.item_number,
            document_type=original_item.document_type,
            jurisdiction=original_item.jurisdiction,
            effective_date=original_item.effective_date,
            expiration_date=original_item.expiration_date,
            issuing_authority=original_item.issuing_authority,
            is_current=original_item.is_current,
            document_id=original_item.document_id,
        )
        fused_results.append(fused_item)

    # 返回全新列表：调用方可自由排序/截断，不影响任何输入条目
    return fused_results
