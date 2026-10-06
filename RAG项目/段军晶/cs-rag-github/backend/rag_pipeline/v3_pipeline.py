# -*- coding: utf-8 -*-
"""
V3 链路：RRF 候选 → BGE-reranker 重排（+ 可选的查询改写）

与 V2 的差异**只在检索环节的尾部**：

    V2：稠密20 + 稀疏20 → RRF → top-5
    V3：稠密20 + 稀疏20 → RRF → 候选10 → **重排** → top-5
        另可开关：检索前先做**查询改写**，多查询结果再做一次 RRF 融合

设计要点：**不复制 V2 的检索代码**
    多查询时对每个查询各调用一次 `super().retrieve()`，再对返回的排名列表
    做跨查询 RRF 融合。因为 `super().retrieve()` 已经完成了「稠密+稀疏+RRF+
    MySQL 回填」，所以 V3 完全复用 V2，无需复制其实现 —— 这也是
    `v2_pipeline.py` 能保持零改动的原因（ADR-018 的冻结策略延伸至 V2）。

重排替代 RRF 作为最终排序（ADR-023）：
    两路 RRF 分数与重排分数尺度完全不同，混合无依据。RRF 只负责
    「选出送入重排的候选」，最终顺序由重排决定。

降级底线（ADR-024）：
    重排或改写任何一个环节失败，都必须退回 V2 的行为并记 WARNING，
    **绝不允许中断问答**。加分项不能变成减分项。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.logging_config import get_logger, log_retrieval_trace
from backend.rag_pipeline import rrf
from backend.rag_pipeline.query_rewrite import get_rewriter
from backend.rag_pipeline.v2_pipeline import V2Pipeline
from backend.reranker import RerankerNotAvailableError, get_reranker

logger = get_logger(__name__)


class V3Pipeline(V2Pipeline):
    """
    V3 链路：在 V2 之上叠加重排，并可选启用查询改写。

    继承链：V3Pipeline → V2Pipeline → V1Pipeline
    因此 answer() / 会话 / 缓存 / 提示词 / 生成 / 溯源 / 收尾全部继承而来。

    `_v2` 属性仅用于测试时注入假实现；生产路径下为 None，走 super()。
    """

    # 链路名。在线接口靠它从注册表里取到本实现（get_pipeline_by_name("v3")），
    # 前端切换 V1 / V2 / V3 靠的也是这个名字。
    PIPELINE_NAME = "v3"

    # 测试用的注入点：跑单测时塞一个假的 V2 检索实现进来，
    # 就不必真去连 Milvus / MySQL。生产路径下它始终是 None，走 super()。
    _v2: Any = None

    def _v2_retrieve(self, query: str, *, top_k: int) -> List[Dict[str, Any]]:
        """调用 V2 的检索实现（测试可通过 `_v2` 注入替身）"""
        # 测试注入的替身分支
        if self._v2 is not None:
            return self._v2.retrieve(query, top_k=top_k)
        # 生产路径：直接复用父类 V2 的检索 —— 它已经替我们做完了
        # 「稠密 + 稀疏 → RRF → 回 MySQL 补页码」全套动作。
        # 正因为能这样复用，V3 一行检索代码都不用重写，V2 也能保持零改动。
        return super().retrieve(query, top_k=top_k)

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        V3 检索：可选改写 → 多路检索 → 跨查询 RRF → 可选重排。

        返回结构与 V1/V2 完全一致，仅新增 `rerank_score` 字段（重排启用时）。
        """
        # final_k：最终要拼进提示词、送给大模型的块数（也就是当初 V2 的 top-5）。
        final_k = top_k or settings.retrieve_top_k
        # candidate_k：先粗捞出多少个候选交给重排。重排是从这批候选里挑 final_k 个，
        # 所以候选池必须比 final_k 大 —— 池子太小，重排再准也没得挑。
        candidate_k = settings.rerank_candidate_k or settings.retrieve_candidate_k

        # ---- ① 可选的查询改写（默认关闭）----
        # 检索用的查询列表，**第一个永远是用户原问题**。
        # 改写出来的查询只是"追加"进来的补充路，不会顶替原问题 ——
        # 这样即使改写跑偏，原问题这一路仍能保证召回质量不下降。
        queries = [question]
        if settings.query_rewrite_enabled:
            try:
                extra = get_rewriter().rewrite(question)
            except Exception as exc:
                # rewrite() 本身已做兜底，此处再保一层，确保绝不中断
                logger.warning("查询改写异常，降级为只用原问题：%s", exc)
                extra = []
            # 改写器返回空列表表示"这次没改出东西"，此时 queries 里仍然只有原问题，
            # 链路照常往下走 —— 这就是降级。
            if extra:
                queries.extend(extra)
            # 把改写结果记进服务端日志（不进 HTTP 响应体），
            # 排查时能看出这次到底改出了哪些关键词。
            log_retrieval_trace(
                logger,
                stage="v3_query_rewrite",
                question=question,
                hits=None,
                extra={"rewritten": extra, "total_queries": len(queries)},
            )

        # ---- ② 逐查询检索，并对结果做跨查询 RRF 融合 ----
        # 每个查询各取 candidate_k * 2 条，给融合留出余量
        #
        # ★ 每个查询单独兜底 ★
        #   被保护的不能只有改写器本身：`_v2_retrieve()` 也会失败（Milvus/MySQL
        #   瞬时故障，或 V2 在向量化失败时抛的 RuntimeError）。此时**原问题的
        #   检索结果往往已经拿到**，若让异常逃逸，整个问答就白抛了 —— 直接违反
        #   ADR-024「加分项失败必须退回 V2 行为、绝不中断问答」。
        #   故：单条查询失败只跳过该条并记 WARNING，其余查询继续。
        # ranked_lists：每个查询各产出一份「按名次排好的 chunk_id 列表」。
        #   交给 RRF 融合的就是这些纯名次信息。
        # by_id：chunk_id -> 块的完整字段（正文、页码、文件名……）。
        #   融合只能算出 id 和分数，最后还得靠这张表把内容取回来。
        ranked_lists: List[List[str]] = []
        by_id: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            try:
                # 每个查询都多取一倍（*2），给后面的 RRF 融合留余量：
                # 各个查询召回的块只有部分重叠，融合之后才够 candidate_k 个。
                hits = self._v2_retrieve(q, top_k=candidate_k * 2)
            except Exception as exc:
                # 单条查询失败只跳过这一条，其余查询继续。
                # 尤其要保住"原问题"那一路的召回结果，不能因为一条补充查询
                # 出错就把整次问答作废。
                logger.warning("查询检索失败，跳过该查询 | query=%r | %s", q, exc)
                continue
            # 只留下 id 的有序列表给 RRF 用：RRF 只看名次，不看具体分数。
            ranked_lists.append([h["chunk_id"] for h in hits])
            for h in hits:
                # setdefault：同一个块被多个查询召回时只留第一次见到的那份数据，
                # 避免重复覆盖（内容本就相同，省一次拷贝）。
                by_id.setdefault(h["chunk_id"], h)

        # 全部查询都失败时 by_id 为空，返回 [] —— 交由上层走「检索无结果」路径，
        # 这正是 V2 在检索为空时的既有行为，属于正确的降级。

        if not by_id:
            return []

        # 只有一路检索（没开改写，或补充查询全失败）时不需要融合：
        # 直接把这一路的名次翻译成候选块即可，省掉一次没有意义的 RRF 计算。
        if len(ranked_lists) == 1:
            candidates = [by_id[cid] for cid in ranked_lists[0]][:candidate_k]
        else:
            # 多路时用 RRF 融合成统一名次。注意：**融合不会产生新的候选**，
            # 它只能给"已经被某一路召回过的块"重新排名 —— 一个块如果所有查询
            # 都没能召回它，RRF 也变不出来。这正是 R09 那类样本只能靠
            # 查询改写解决、而重排救不了的原因。
            fused = rrf.fuse(ranked_lists, top_k=candidate_k)
            candidates = [by_id[cid] for cid, _ in fused if cid in by_id]
            # 记录融合过程：哪些块进了候选池、各自 RRF 多少分、分别落在第几页。
            log_retrieval_trace(
                logger,
                stage="v3_multi_query_fuse",
                question=question,
                hits=[
                    {"chunk_id": cid, "score": s, "page_no": by_id[cid].get("page_no")}
                    for cid, s in fused
                    if cid in by_id
                ],
                extra={
                    # queries 是「尝试数」，succeeded 是「实际参与融合的通道数」。
                    # 有了单查询兜底后两者可能不等（某条查询检索失败被跳过），
                    # 只记 queries 会让排查者误以为融合用了更多通道。
                    "queries": len(queries),
                    "succeeded": len(ranked_lists),
                    "rrf_k": settings.rrf_k,
                },
            )

        # 融合后仍然是空，说明检索真的一条都没命中，
        # 返回空列表交由上层走「根据现有知识库，未找到相关内容」的分支。
        if not candidates:
            return []

        # ---- ③ 重排（主链路默认开启）----
        # 此时 candidates 是 RRF 排出来的候选池，重排要在池内定出最终顺序。
        # 不用再套 try：_rerank 内部已经做了降级，失败时原样返回。
        if settings.rerank_enabled:
            candidates = self._rerank(question, candidates, final_k)

        # 最后截到 final_k 条 —— 这几段就是即将拼进提示词送给大模型的原文。
        return candidates[:final_k]

    # ------------------------------------------------------------------
    # 溯源构造（覆盖 V1 的实现，理由见 docstring）
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_key(item: Dict[str, Any]) -> float:
        """
        排序依据：有 `rerank_score` 就用它，否则退回 `score`。

        `rerank_score` 只在重排成功时由 `reranker.rerank()` 新增；
        重排关闭或降级时该键不存在，此时与 V1/V2 完全一致。
        """
        # 优先取重排分：重排跑成功时，它就是最终顺序的唯一依据。
        value = item.get("rerank_score")
        # 重排没跑（功能关闭或降级）时这个字段不存在，就退回 RRF 分 ——
        # 于是 V3 在降级状态下的排序口径与 V2 完全一致。
        if value is None:
            value = item.get("score", 0.0)
        return float(value)

    @staticmethod
    def _build_sources(ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 按「文件名 + 页码」分组，每组只留得分最高的那一个块。
        # 为什么要去重：检索经常同时命中同一页里相邻的好几个块，
        # 不去重的话前端来源列表会出现好几条一模一样页码 ——
        # 而用户真正关心的是"答案在第几页"。
        best: Dict[tuple, Dict[str, Any]] = {}
        for item in ordered:
            key = (item["file_name"], item["page_no"])
            # 同页取「最优」，判据与下方的排序判据同为 _rank_key（见 docstring）
            if key not in best or V3Pipeline._rank_key(item) > V3Pipeline._rank_key(best[key]):
                best[key] = item

        sources: List[Dict[str, Any]] = []
        # 与父类的差异：去重判据与排序依据均由 `score` 换成 `_rank_key`
        # 按最终得分降序输出，保证 sources[0] 就是答案最可能依据的那一页。
        for item in sorted(best.values(), key=lambda x: -V3Pipeline._rank_key(x)):
            # 把正文压成单行并去掉首尾空白，作为前端"来源"卡片里的摘要预览。
            content = (item["content"] or "").strip().replace("\n", " ")
            sources.append({
                "file_name": item["file_name"],
                "page_no": item["page_no"],
                "page_nums": item["page_nums"],
                "chunk_id": item["chunk_id"],
                # 这里返回的仍是 RRF 分（不是重排分），以保持对外 API 契约不变
                "score": round(float(item["score"]), 4),
                # 摘要超过 120 字就截断加省略号，免得来源卡片太长
                "summary": content[:120] + ("…" if len(content) > 120 else ""),
                "content_type": item["content_type"],
                "section_title": item.get("section_title", ""),
            })
        return sources

    def _rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        final_k: int,
    ) -> List[Dict[str, Any]]:
        """
        重排，失败时**退回原顺序**并记 WARNING（ADR-024）。

        重排使用**用户原始问题**而非改写后的关键词串：重排模型在自然语言
        问答数据上训练，完整问句更契合其训练分布。
        """
        # 先记下重排前的顺序，用于日志里对比"重排到底有没有改动排序"。
        before = [c["chunk_id"] for c in candidates]
        try:
            # 注意传进去的是 question（用户原问题），不是改写出来的关键词串。
            reranked = get_reranker().rerank(question, candidates, top_k=final_k)
        except RerankerNotAvailableError as exc:
            # 重排不可用（模型没装、路径配错、推理超时）—— 这是设计内的降级路径，
            # 原样返回 RRF 顺序，问答照常进行。
            logger.warning("重排不可用，退回 RRF 顺序 | %s", exc)
            return candidates
        except Exception as exc:
            # 再加一层保险：万一抛出契约之外的异常，也绝不让它中断问答。
            logger.warning("重排异常，退回 RRF 顺序 | %s", exc)
            return candidates

        # 记录重排前后的顺序对比，答辩时可以据此说明重排带来的实际变化。
        log_retrieval_trace(
            logger,
            stage="v3_rerank",
            question=question,
            hits=reranked,
            extra={
                "before": before,
                "after": [c["chunk_id"] for c in reranked],
                "changed": before[: len(reranked)] != [c["chunk_id"] for c in reranked],
                # 单独给出重排分：`log_retrieval_trace` 的 hits 预览只格式化
                # 各条的 `score`（即**重排前的 RRF 分**），而此处顺序已按
                # rerank_score 排列 —— 两者并存会让日志读起来像「顺序与分数不符」。
                "rerank_scores": {
                    c["chunk_id"]: c.get("rerank_score") for c in reranked
                },
            },
        )
        return reranked


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

# 进程级单例：链路对象持有会话、缓存等状态，全进程必须共用一个。
_pipeline: Optional[V3Pipeline] = None


def get_pipeline() -> V3Pipeline:
    """获取 V3 链路单例（供 get_pipeline_by_name("v3") 调用）"""
    global _pipeline
    if _pipeline is None:
        _pipeline = V3Pipeline()
    return _pipeline
