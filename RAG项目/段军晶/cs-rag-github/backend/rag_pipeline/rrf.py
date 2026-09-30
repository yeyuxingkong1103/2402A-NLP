# -*- coding: utf-8 -*-
"""
RRF（Reciprocal Rank Fusion）融合模块

职责：
    把多个检索通道的**有序结果列表**融合为单一排序。

设计要点：
    1. 纯函数、无 I/O、无副作用 —— 便于单元测试与复用
    2. 融合基于**名次**而非分数，因此天然免疫各通道分数尺度不一致的问题
       （稠密通道为余弦相似度，稀疏通道为内积，二者不可直接比较）
    3. k 值取自配置 settings.rrf_k（默认 60），不硬编码

融合公式：
    score(chunk) = Σ_c  1 / (k + rank_c(chunk))

    其中 rank_c 为该 chunk 在第 c 通道中的名次（从 1 开始）；
    未出现在某通道则不计该项。

为什么不使用加权融合：
    backend/config.py 中存在 hybrid_sparse_weight 配置项，那是「按分数加权」
    设想的遗留。RRF 是基于名次的融合，不使用权重，故该配置项在本方案中
    保持未被读取（见技术决策记录 ADR-019）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from backend.config import settings

# 融合结果元素：(chunk_id, rrf_score)
# 融合结果只保留「块 ID + 综合得分」两个字段：
# 下游（重排、上下文组装）只需要知道"拿哪一块"和"它凭什么排在前面"，
# 具体是哪个通道把它召回的，已经不需要了。
FusedItem = Tuple[str, float]


def fuse(
    channels: Sequence[Sequence[str]],
    *,
    k: Optional[int] = None,
    top_k: int = 5,
) -> List[FusedItem]:
    """
    按 RRF 融合多个通道的检索结果。

    参数：
        channels : 各通道的 chunk_id 有序列表（按名次从优到劣）。
                   允许空列表，允许各通道长度不同。
        k        : RRF 平滑常数。None 时取 settings.rrf_k（默认 60）。
        top_k    : 返回结果条数。

    返回：
        [(chunk_id, rrf_score), ...] 按分数降序；
        分数相同时按 chunk_id 升序，保证结果**可复现**。
    """
    # 没传 k 就取配置里的默认值（60）。k 越大名次之间的得分差距越平缓，
    # 融合越"平均主义"；k 越小越奖励某个通道里的头部结果。
    if k is None:
        k = settings.rrf_k
    # k 出现在公式分母 1/(k+rank) 里，取 0 或负数会导致除零或算出负分，
    # 与其静默给出错误排序，不如当场报错。
    if k <= 0:
        raise ValueError(f"RRF 的 k 必须为正整数，当前为 {k}")
    # 要 0 条结果就直接返回，跳过整轮打分。
    if top_k <= 0:
        return []

    # 融合得分累加表：chunk_id -> RRF 得分。
    # 同一个块如果在稠密和稀疏两路里都被召回了，两路的得分会在这里相加，
    # 于是"两路都认可的块"会自然浮到前面 —— 这正是混合检索要达到的效果。
    scores: Dict[str, float] = {}
    # 逐个通道累加。相加可交换，所以通道先后顺序不影响最终结果；
    # 将来要加第三路（比如关键词检索），往 channels 里追加一个列表即可。
    for channel in channels:
        # 每个通道单独记一份"已见过"，只在**本通道内**去重：
        # 同一个块在同一路里出现两次（重复入库会产生这种脏数据），
        # 只认它最好的那一次名次，不让它在本通道里重复得分。
        seen = set()
        # rank 从 1 开始算，因为 RRF 公式里的名次是"第 1 名、第 2 名……"；
        # 若从 0 开始，1/(k+0) 会凭空给出一个偏高的分数。
        for rank, chunk_id in enumerate(channel, start=1):
            # 同一通道内重复出现的块只记首次名次，避免重复计分
            # 空 chunk_id 说明这条检索结果本身是脏数据，直接丢弃。
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            # RRF 的核心公式：本通道贡献 1/(k+rank)。名次越靠前贡献越大；
            # 因为只用名次、完全不用原始分数，余弦相似度（稠密路）和内积
            # （稀疏路）这两种不同量纲的分数才能安全地放在一起比较。
            # scores.get(..., 0.0) 保证第一次遇到该块时从 0 开始累加。
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # 按融合得分从高到低排。第二个排序键 chunk_id 升序是**可复现性**的保障：
    # 没有它的话，同分块之间谁在前谁在后取决于字典遍历顺序，
    # 同一个问题两次提问就可能拿到不同的上下文，评测结果也没法复现。
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    # 只交出前 top_k 个。融合出来的候选通常多于此数，
    # 在这里截断是为了控制进入下一步重排的规模 —— 重排要逐条打分，很贵。
    return ranked[:top_k]
