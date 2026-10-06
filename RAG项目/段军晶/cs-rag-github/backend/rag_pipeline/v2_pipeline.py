# -*- coding: utf-8 -*-
"""
V2 增强问答链路（稠密 + 稀疏 + RRF 混合检索）

与 V1 的差异**只在检索环节**：

    V1：稠密向量检索 Top-K
    V2：稠密候选 Top-N + 稀疏候选 Top-N → RRF 融合 → Top-K

其余环节（会话管理、查询缓存、提示词组装、答案生成、溯源构造、收尾记录）
全部继承自 V1Pipeline，本模块不复制这部分代码。

为什么用继承而不是复制：
    V1Pipeline.answer() 内部调用 self.retrieve(question, top_k=top_k)
    （见 v1_pipeline.py:139），因此**覆盖 retrieve() 即可**让整条 answer
    链路自动走混合检索，无需重写 answer()，也无需改动任何公共逻辑。

为什么 v1_pipeline.py 保持不修改：
    重灌知识库后索引内容已变（图片块追加了视觉描述）。只要 V1 代码保持原样，
    V1 在新索引上的指标变化就能**干净归因到索引改动**；反过来若顺手重构了它，
    就再也分不清指标波动来自代码改动还是索引改动。这是有意为之（ADR-018）。

Trace 约定（V2 第 1 项要求）：
    两路召回结果**在融合前分别记录**，并记录融合后每个块在两路中的原始名次。
    若只记录融合后的结果，就无法回答「某个块是靠稠密还是靠稀疏被召回的」，
    而这正是 V2 归因分析的核心问题。
    分通道的页码取自 Milvus 的冗余标量字段（MySQL 回填发生在融合之后）；
    对外返回的溯源信息仍一律以 MySQL 为准（ADR-005）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.db import milvus_client, mysql
from backend.embedder import get_embedder
from backend.logging_config import get_logger, log_retrieval_trace
from backend.rag_pipeline import rrf
from backend.rag_pipeline.v1_pipeline import V1Pipeline

logger = get_logger(__name__)


# ===========================================================================
# V2 链路
# ===========================================================================

class V2Pipeline(V1Pipeline):
    """
    V2 链路：在 V1 全部能力之上，把检索替换为混合检索。

    继承 V1Pipeline 获得：answer / _build_context / _build_messages /
    _build_sources / _finish 以及防幻觉提示词 SYSTEM_PROMPT。

    本类只覆盖两处：
        PIPELINE_NAME —— 响应体中的版本标识（父类 v1_pipeline.py:337 用到）
        retrieve()    —— 检索环节本身
    """

    # 版本标识，会随响应体的 pipeline 字段返回给前端，
    # 用来区分这条答案究竟是 V1 / V2 / V3 哪条链路产出的。
    PIPELINE_NAME = "v2"

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        V2 混合检索：稠密 + 稀疏 → RRF 融合。

        返回结构与 V1 完全一致（键名、类型、语义均相同），
        因此 answer() / 评测脚本 / API 层无需任何改动即可复用：
            [{chunk_id, score, file_name, page_no, page_nums,
              content, content_type, section_title}, ...]

        其中 score 为 **RRF 融合分数**（量级约 0.01~0.03，取决于 rrf_k），
        与 V1 的余弦相似度不可直接比较 —— 这是刻意的：
        RRF 基于名次融合，本就不产生可跨版本比较的绝对分数。
        """
        # final_k：融合之后最终交出去的块数（即 V1 里那个 Top-K）。
        final_k = top_k or settings.retrieve_top_k
        # candidate_k：每一路各召回多少个候选。两路各取这么多再去融合，
        # 是为了让"两路都认可的块"有机会胜出 —— 如果每路只取 final_k 个，
        # 融合之后候选就不够 final_k 了。
        candidate_k = settings.retrieve_candidate_k

        # ---- ② 问题编码：一次 forward 同时得到稠密向量与稀疏权重 ----
        # ★ 这里就是 V2 与 V1 的分岔点 ★
        # V1 只把问题编成一个稠密向量；这里还要一份稀疏权重，
        # 因为下面要走两条检索通道，各自需要不同形态的查询表示。
        # 两个输出共用同一次前向计算，所以多出来的稀疏那一路几乎不增加推理耗时。
        dense_vectors, sparse_vectors = get_embedder().encode_dense_and_sparse(
            [question]
        )
        # 这里只编了一个问题，所以取第 0 个就是它的表示。
        query_dense = dense_vectors[0] if dense_vectors else []
        query_sparse = sparse_vectors[0] if sparse_vectors else {}
        # 稠密向量拿不到就没法检索，直接报错（上层的降级逻辑会兜住）。
        if not query_dense:
            raise RuntimeError("问题向量化失败，请检查 BGE-M3 模型是否可用")

        # ---- ③ 两路独立检索 ----
        # 第一路：稠密检索（语义相似）—— 问的是「哪些块和这个问题意思接近」。
        # 强项是能处理换了说法、跟原文没有共同词的提问。
        dense_hits = milvus_client.search_dense(query_dense, top_k=candidate_k)
        # 第二路：稀疏检索（字面匹配）—— 问的是「哪些块真的出现了这些词」。
        # 强项是标准编号、专有名词这类必须一字不差才能命中的线索。
        # 稀疏检索的语义：只返回与查询**有词元重叠**的块。
        # 完全无重叠时结果为空，属正常情况（该问题无字面命中线索）。
        sparse_hits = (
            milvus_client.search_sparse(query_sparse, top_k=candidate_k)
            if query_sparse
            else []
        )

        # 两路结果必须在融合前分别落 Trace —— V2 的 Trace 要求核心
        # （融合之后就再也看不出某个块是靠哪一路召回的，归因分析无从做起）
        log_retrieval_trace(
            logger,
            stage="v2_dense_retrieve",
            question=question,
            hits=dense_hits,
            extra={"candidate_k": candidate_k, "collection": settings.milvus_collection},
        )
        log_retrieval_trace(
            logger,
            stage="v2_sparse_retrieve",
            question=question,
            hits=sparse_hits,
            extra={"candidate_k": candidate_k, "collection": settings.milvus_collection},
        )

        # ---- ④ RRF 融合（k 取自 settings.rrf_k，不硬编码）----
        # 只把 chunk_id 的有序列表交给融合器：RRF 完全基于名次计算，
        # 不碰两路的原始分数 —— 这正是它能跨量纲融合的原因
        # （稠密路给的是余弦相似度，稀疏路给的是内积，两个分数根本不可比）。
        dense_ids = [h["chunk_id"] for h in dense_hits]
        sparse_ids = [h["chunk_id"] for h in sparse_hits]
        # 融合成一份统一的名次，取前 final_k 个作为最终候选。
        fused = rrf.fuse([dense_ids, sparse_ids], top_k=final_k)

        # 记录融合结果 + 每个块在两路中的原始名次（0 表示未出现在该通道）。
        # 记录每个chunk在两路的原始排名
        dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
        sparse_rank = {cid: i for i, cid in enumerate(sparse_ids, start=1)}
        # 这份页码只用于 Trace 日志回显，取的是 Milvus 里的冗余字段；
        # 对外返回的溯源信息一律以 MySQL 为准（见下面的第 ⑤ 步）。
        # 从Milvus的冗余字段临时保存page_no，只用于日志展示
        page_of: Dict[str, Any] = {}
        for hit in dense_hits + sparse_hits:
            page_of.setdefault(hit["chunk_id"], hit.get("page_no"))

        # 融合这一步的 Trace 信息量最大：融合结果 + 每个块在两路里的原始名次，
        # 全部写进服务端日志（不进 HTTP 响应体）。
        log_retrieval_trace(
            logger,
            stage="v2_rrf_fuse",
            question=question,
            hits=[
                {
                    "chunk_id": cid,
                    "score": score,
                    "page_no": page_of.get(cid, "?"),
                }
                for cid, score in fused
            ],
            extra={
                "rrf_k": settings.rrf_k,
                "final_k": final_k,
                "dense_hits": len(dense_hits),
                "sparse_hits": len(sparse_hits),
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
            },
        )

        # 两路都没召回任何东西（知识库里确实没有相关内容），
        # 返回空列表，由上层走「未找到相关内容」的分支。
        if not fused:
            return []

        # ---- ⑤ 元数据回填（页码溯源真相源，ADR-005）----
        # ★ 为什么必须回 MySQL 再查一次 ★
        # Milvus 只负责回答"哪些块相关"，它里面那个页码字段只是冗余副本；
        # 页码与文件名的唯一真相源是 MySQL。这样即使 Milvus 集合被整个重建，
        # 溯源信息也不会丢 —— 这正是本项目"页码溯源"硬性要求的落地方式。
        fused_ids = [cid for cid, _ in fused]
        score_map = dict(fused)
        # 一次批量把所有块的元数据查回来，避免在循环里逐条查库。
        chunk_map = {
            c["chunk_id"]: c for c in mysql.fetch_chunks_by_ids(fused_ids)
        }

        ordered: List[Dict[str, Any]] = []
        # 按融合出来的名次逐条组装，保证返回顺序与 RRF 的排序一致。
        for cid in fused_ids:
            meta = chunk_map.get(cid)
            if not meta:
                # 向量库与 MySQL 不一致（如 MySQL 被清空），跳过并告警
                logger.warning("chunk 元数据缺失（MySQL 中不存在）：%s", cid)
                continue
            ordered.append({
                "chunk_id": cid,
                # 注意这里的 score 是 RRF 融合分（量级约 0.01~0.03），
                # 不是 V1 的余弦相似度，两者的绝对值不可直接比较。
                "score": score_map.get(cid, 0.0),
                # ★ 页码与文件名一律以 MySQL 为准 ★
                "file_name": meta.get("source_file") or meta.get("doc_id", ""),
                "page_no": int(meta.get("page_no") or 1),
                "page_nums": meta.get("page_nums") or [int(meta.get("page_no") or 1)],
                "content": meta.get("content", ""),
                "content_type": meta.get("content_type", "text"),
                "section_title": meta.get("section_title", ""),
            })
        return ordered
    # 1.返回顺序严格保持RRF融合排序；
    # 2.score是RRF分数，量级一般0.01~0.03， ** 不能和V1的余弦相似度对比 **；
    # 3.所有对外暴露的溯源信息`file_name / page_no`全部来自MySQL；
    # 4.捕获向量库和MySQL不一致的脏数据场景，打warning日志，跳过无效chunk。


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

# 进程级单例：链路对象持有会话与缓存状态，全进程共用一个。
_pipeline: Optional[V2Pipeline] = None


def get_pipeline() -> V2Pipeline:
    """获取 V2 链路单例（供 get_pipeline_by_name("v2") 调用）"""
    global _pipeline
    if _pipeline is None:
        _pipeline = V2Pipeline()
    return _pipeline
