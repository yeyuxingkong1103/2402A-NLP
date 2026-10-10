# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""RRF 融合 + 图像指向词加权。

之所以要「图像指向词加权」：id 5/id 6 的答案只在图里，若纯按相似度排序，
文本块会挤掉图像块，导致图像通道形同虚设。
"""
from __future__ import annotations

import logging
import re

from rag04.config import Settings
from rag04.schema import Hit

logger = logging.getLogger("rag04.hybrid")

_IMAGE_CUES = re.compile(
    r"组织结构图|组织架构图|股权结构图|结构图"
    r"|图中|如图|下图|上图|见图|图示"
    r"|应用结构|增长图|增长率|产业链|示意图|流程图|柱状图|饼图|图表"
)


def is_image_pointing(question: str) -> bool:
    """问句是否明确指向图像内容。"""
    return bool(_IMAGE_CUES.search(question or ""))


def rrf_fuse(rankings: list[list[Hit]], k: int = 60,
             weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion。score = Σ w_i / (k + rank_i)。

    RC2：同一张图的图像命中（CLIP 路）与描述文本命中（dense/BM25 路）携带相同
    ``dedup_key``（= 该 image 块的 chunk_id，见 chunker.chunk_figure_texts），
    在此合并为**一席**并累加分数；代表命中优先取图像块（保留 image_path 与
    图像指向加权）。``dedup_key`` 为空的普通块仍按 chunk_id 去重，行为不变。
    """
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)

    acc: dict[str, float] = {}
    best: dict[str, Hit] = {}                 # 去重键 → 代表命中（best[key] 即代表块）

    for wi, ranking in zip(weights, rankings):
        seen: set[str] = set()
        for rank, h in enumerate(ranking, start=1):
            if h.chunk_id in seen:      # 同一路内去重
                continue
            seen.add(h.chunk_id)
            key = h.dedup_key or h.chunk_id
            acc[key] = acc.get(key, 0.0) + wi / (k + rank)
            if key not in best:
                best[key] = h
            elif h.block_type == "image" and best[key].block_type != "image":
                best[key] = h                # 图像块更能代表「这张图」

    out: list[Hit] = []
    for key, score in sorted(acc.items(), key=lambda x: -x[1]):
        h = best[key]
        out.append(Hit(
            chunk_id=h.chunk_id, doc_id=h.doc_id, page=h.page,
            block_type=h.block_type, source_id=h.source_id, text=h.text,
            score=score, channel="fused", image_path=h.image_path,
            dedup_key=key,
        ))
    return out


def apply_image_boost(hits: list[Hit], question: str, boost: float = 1.5) -> list[Hit]:
    """图像指向问句时提升图像块权重，保证图像进入上下文。"""
    if not is_image_pointing(question):
        return hits
    out = []
    for h in hits:
        s = h.score * boost if h.block_type == "image" else h.score
        out.append(Hit(
            chunk_id=h.chunk_id, doc_id=h.doc_id, page=h.page,
            block_type=h.block_type, source_id=h.source_id, text=h.text,
            score=s, channel=h.channel, image_path=h.image_path,
            dedup_key=h.dedup_key,
        ))
    return sorted(out, key=lambda x: -x.score)


def _coll_snapshot(store) -> str:
    """尽力给出集合点数快照，用于把「空召回」与「空索引」区分开（Fix 8）。

    ``VectorStore.search`` 会吞掉异常返回 []，维度不匹配/集合为空这类配置错误
    在检索侧只表现为「0 命中」。此处不改契约，只把可查到的点数写进告警。
    """
    try:
        counts = getattr(store, "counts", None)
        if callable(counts):
            c = counts()
            if c:
                return "，当前集合点数 " + ", ".join(f"{k}={v}" for k, v in c.items())
    except Exception:
        pass
    return ""


def hybrid_retrieve(question: str, s: Settings, store, bm25,
                    embed_fn, clip_fn=None, k_each: int = 30,
                    boilerplate_ids: set[str] | frozenset[str] | None = None
                    ) -> list[Hit]:
    """三路召回 → RRF 融合 → 图像加权。任一路失败不影响其余。

    ``boilerplate_ids``（RC1）为跨页重复样板集合，一致作用于三路：命中即在
    各路内部剔除（稠密先按 ``boilerplate_fetch_k`` 深挖补位），使页眉/页脚
    不再占据候选与重排名额。

    ``s.use_hybrid`` 是双模式的总开关（Fix 7）：为 False（baseline_03 控制组）
    时**只跑稠密路**，稀疏（BM25）路整路跳过——计划的双模式语义要求 baseline
    是纯稠密，原实现无条件跑 BM25，使「baseline」实际是 dense+sparse 的 RRF。
    CLIP 路由 ``use_clip_retrieval`` 单独控制（属图像能力，见下）；表格/图像
    的入库开关在 ingest 侧生效。
    """
    from rag04.ingest.store import COLL_TEXT, COLL_TABLE

    bp = boilerplate_ids or frozenset()
    # 过滤会吃掉候选：深挖后再截到 k_each，保证每路候选规模不缩水。
    # 无过滤时不加任何深挖开销。
    fetch_k = max(k_each, s.boilerplate_fetch_k) if bp else k_each

    def keep(hits: list[Hit]) -> list[Hit]:
        if not bp:
            return hits
        return [h for h in hits if h.chunk_id not in bp]

    rankings: list[list[Hit]] = []
    weights: list[float] = []

    # 路1：稠密（正文）
    try:
        qv = embed_fn(question, s)
        dense = keep(store.search(COLL_TEXT, qv, k=fetch_k))[:k_each]
        tbl = keep(store.search(COLL_TABLE, qv, k=fetch_k))[:k_each]
        rankings.append(dense + tbl)
        weights.append(1.0)
        if not dense and not tbl:
            # Fix 8：空索引不再静默。store.search 吞异常返回 []，此告警让
            # 「集合为空 / 维度不匹配」这类配置问题可见。
            logger.warning("稠密召回 0 命中（语料集合为空或向量维度不匹配？%s）",
                           _coll_snapshot(store))
    except Exception as e:
        logger.warning("稠密召回失败：%s", e)

    # 路2：稀疏（正文）。排除在 top-k 截断之前生效（见 BM25Index.search）。
    # Fix 7：use_hybrid=False 时整路跳过（baseline_03 必须只跑纯稠密）。
    if not s.use_hybrid:
        logger.info("use_hybrid=False（baseline 控制组）：跳过稀疏通路，仅稠密召回")
    else:
        try:
            sparse_raw = bm25.search(question, k=k_each, exclude_ids=bp or None)
            by_id = {c.chunk_id: c for c in bm25.chunks}
            sparse: list[Hit] = []
            for cid, sc in sparse_raw:
                c = by_id.get(cid)
                if c is None:
                    continue
                sparse.append(Hit(
                    chunk_id=c.chunk_id, doc_id=c.doc_id, page=c.page,
                    block_type=c.block_type, source_id=c.source_id, text=c.text,
                    score=sc, channel="sparse",
                    image_path=(c.extra or {}).get("image_path", ""),
                    dedup_key=(c.extra or {}).get("dedup_key", ""),
                ))
            rankings.append(sparse)
            weights.append(1.0)
            if not sparse:
                # Fix 8：BM25Index.search 空索引/无词元命中都返回 []，须告警点名原因。
                cause = ("稀疏索引为空（bm25.pkl 未构建或语料为空？）"
                         if not getattr(bm25, "chunks", None)
                         else "查询词元未命中任何块（或全部命中均被样板集合排除）")
                logger.warning("稀疏召回 0 命中：%s", cause)
        except Exception as e:
            logger.warning("稀疏召回失败：%s", e)

    # 路3：CLIP 跨模态（文本→图像）
    if s.use_clip_retrieval and clip_fn is not None:
        # RC3：图像指向时收紧 CLIP 路条数。权重 2.0×boost 1.5 下图像得分
        # 恒高于文本理论最高分，CLIP 若取满 k_each，重排窗口会被图像占满、
        # 文本证据（含金标文本块）永远进不了重排。收 k 保留图像加权特性，
        # 同时硬截断（不依赖 clip_fn 自觉）保证文本席位。
        clip_k = min(k_each, s.clip_max_k) if is_image_pointing(question) else k_each
        try:
            img_hits = clip_fn(question, s, store, k=clip_k)
            # 样板过滤同样作用于 CLIP 路的「文本侧」（图像块 payload 文本）
            img_hits = keep(list(img_hits or []))[:clip_k]
            if img_hits:
                rankings.append(img_hits)
                weights.append(2.0 if is_image_pointing(question) else 1.0)
            else:
                # VectorStore.search 会吞掉异常并返回 []，维度不匹配等配置错误会表现为
                # 「无图可召回」。这条日志让纯空召回可排查，而不是静默降级。
                logger.warning("CLIP 跨模态召回 0 命中（图像库为空或向量维度不匹配？%s），"
                               "已跳过该路", _coll_snapshot(store))
        except Exception as e:
            logger.warning("CLIP 跨模态召回失败，已跳过该路：%s", e)

    if not rankings:
        # Fix 8：三路全空/全失败原先直接 return []，检索侧完全静默（各路自己的
        # 失败告警之外没有汇总信息），把「检索不到」与「索引没建好」混为一谈。
        logger.warning("三路召回全部为空或失败，返回空候选%s", _coll_snapshot(store))
        return []

    fused = rrf_fuse(rankings, k=s.rrf_k, weights=weights)
    if not fused:
        logger.warning("融合结果为 0 命中（各路命中数 %s%s）",
                       [len(r) for r in rankings], _coll_snapshot(store))
        return []
    return apply_image_boost(fused, question)
