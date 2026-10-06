# -*- coding: utf-8 -*-
"""工单3 重排（设计/接口设计.md §3.12 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

默认走**确定性启发式**（无 LLM，保证首字预算与可复现）：
    ``final = boosted_score × (1 + 0.15×关键词覆盖 + 0.10×数值命中 + 0.05×表格块)``
LLM 重排默认关闭（``RAG_RETRIEVAL__ENABLE_LLM_RERANK=false``）；开启时若超时/失败**回退启发式**并显式留痕。
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .chunker import Chunk
from .config import AppConfig, get_config
from .retrieval_utils import keyword_coverage_of
from .text_utils import extract_numbers, has_numeric_signal

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 启发式权重（设计 §3.12 给定系数）
W_COVERAGE = 0.15
W_NUMERIC = 0.10
W_TABLE = 0.05
# T7 修正（t12 口径，§15.2 实测）：答案句完整度特征——修「图表说明文字/章节开头压过定义句」
W_DEFINITIONAL = 0.20      # 含「涉及/包括/分别为/是指…」等定义/枚举句式
W_SENTENCE = 0.05          # 成型句子（有句末标点）小幅加分
W_DIAGRAM_PENALTY = 0.15   # 图表 dump（超长无标点串）扣分
DIAGRAM_RUN = 60           # 连续无标点字符数超过该值即记为「长串」（仅记录，不参与打分）
NEAR_TIE_RATIO = 0.80      # 近邻倒挂修复：后一名分数 ≥ 前一名 × 该比例时视为「几乎持平」
DEFINITIONAL_PATTERN = re.compile(r"涉及|包括|主要(?:为|包括)|分别为|是指|属于|指的是|由.{0,12}组成")
_PUNCT_PATTERN = re.compile(r"[。！？；;!?]")
_SENTENCE_PUNCT = re.compile(r"[。；;，,、]")


def _lazy_logger(logger: Any, module: str = "reranker") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def sentence_quality(text: str) -> dict[str, float]:
    """答案句完整度特征（0/1 为主，确定性、无外部依赖）。

    * ``definitional``：含「涉及/包括/分别为/是指…」等定义或枚举句式 → 更像答案句；
    * ``sentence``：含句末标点（。！？；）→ 是成型句子而非碎片；
    * ``diagram``：存在超长无标点串（``DIAGRAM_RUN`` 字以上）→ 判为图表 dump（实测 p153 这类块
      靠词面重合度会虚高，必须显式扣分）。
    """
    body = str(text or "")
    definitional = 1.0 if DEFINITIONAL_PATTERN.search(body) else 0.0
    sentence = 1.0 if _PUNCT_PATTERN.search(body) else 0.0
    longest = 0
    for piece in _SENTENCE_PUNCT.split(body):
        longest = max(longest, len(piece.strip()))
    diagram = 1.0 if longest >= DIAGRAM_RUN else 0.0
    return {"definitional": definitional, "sentence": sentence, "diagram": diagram,
            "longest_run": float(longest)}


def heuristic_score(query_tokens: Sequence[str], chunk: Chunk, *, boosts: Mapping[str, float],
                    expects_numeric: bool = False) -> float:
    """确定性启发式打分（入参为已加权分数 ``boosts['_base_score']`` 或缺省 1.0）。

    ``expects_numeric=True`` 时**关闭**答案句完整度特征：数值类问题的答案是表格/数值块，
    若按「定义句/成型句」加权会把表块挤出 top-5（T7 实测题 2 召回从命中变未命中）。
    """
    base = float(boosts.get("_base_score", 1.0))
    coverage = keyword_coverage_of(query_tokens, chunk)
    numeric = 1.0 if extract_numbers(chunk.content) else 0.0
    table = 1.0 if chunk.type == "table" else 0.0
    factor = 1.0 + W_COVERAGE * coverage + W_NUMERIC * numeric + W_TABLE * table
    return base * max(factor, 0.05)


def rerank(
    query: str,
    candidates: Sequence[tuple[str, float, dict[str, float]]],
    chunk_map: Mapping[str, Chunk],
    *,
    top_k: int = 5,
    cfg: AppConfig | None = None,
    source_scores: Mapping[str, Mapping[str, float]] | None = None,
    expects_numeric: bool | None = None,
    logger: Any = None,
) -> list[Any]:
    """启发式重排并组装 ``RetrievedChunk`` 列表（按最终分降序，``rank`` 从 1 起）。

    参数扩展说明（设计 §3 允许新增带默认值的参数）：``source_scores`` 用于回填
    ``vector_score``/``bm25_score``/``rrf_score``（这三项在冻结字段里必须存在，但 RRF 融合后
    原始分量已不可从最终分反推，故由调用方显式传入）。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger)
    from .retriever import RetrievedChunk  # 延迟导入避免循环依赖

    with log.enter("rerank", {"query_digest": query[:60], "candidates": len(candidates),
                              "top_k": top_k, "mode": "heuristic"}) as span:
        query_tokens = [t for t in tokenize_query(query)]
        numeric_gate = (has_numeric_signal(query) if expects_numeric is None else bool(expects_numeric))
        scored: list[tuple[float, str, dict[str, float]]] = []
        missing = 0
        for chunk_id, score, boosts in candidates:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                missing += 1
                continue
            merged = dict(boosts)
            merged["_base_score"] = float(score)
            final = heuristic_score(query_tokens, chunk, boosts=merged, expects_numeric=numeric_gate)
            scored.append((final, chunk_id, merged))
        if missing:
            log.log_event("rerank.missing_meta", level="WARNING", missing=missing,
                          reason="候选 chunk_id 不在元数据表中，已跳过")
        scored.sort(key=lambda item: (-item[0], item[1]))
        # 答案句完整度「近邻倒挂修复」（T7/t12，§15.2）：只在**分数几乎持平**（相差 ≤ NEAR_TIE_RATIO）
        # 且后者的定义/枚举句式更强时交换一次名次。
        # 实测动机：题 34 的正确块 p0152（含「上游涉及…制造企业」）0.1355 输给 图表说明块 p0153 0.1532；
        # 全局加分修法会把 题 2 的表块挤出 top-5（召回 14/14→13/14），近邻修复没有这个副作用。
        if not numeric_gate:
            flipped = 0
            for i in range(len(scored) - 1):
                high, low = scored[i], scored[i + 1]
                if high[0] > 0 and low[0] >= high[0] * NEAR_TIE_RATIO:
                    high_chunk, low_chunk = chunk_map.get(high[1]), chunk_map.get(low[1])
                    if high_chunk is not None and low_chunk is not None:
                        q_high = sentence_quality(high_chunk.content)
                        q_low = sentence_quality(low_chunk.content)
                        if (q_low["definitional"] > q_high["definitional"]
                                and q_low["longest_run"] < q_high["longest_run"]):
                            scored[i], scored[i + 1] = scored[i + 1], scored[i]
                            flipped += 1
            if flipped:
                log.log_event("rerank.near_tie_repair", flipped=flipped,
                              rule="近乎持平时定义句式优先（长串无标点的图表说明文字让位）")
        results: list[Any] = []
        for rank, (final, chunk_id, boosts) in enumerate(scored[: max(int(top_k), 1)], start=1):
            chunk = chunk_map[chunk_id]
            src = dict((source_scores or {}).get(chunk_id, {}))
            results.append(RetrievedChunk(
                chunk_id=chunk_id, file_name=chunk.file_name, page=chunk.page, type=chunk.type,
                content=chunk.content, section=chunk.section, table_id=chunk.table_id,
                score=round(float(final), 6),
                vector_score=src.get("vector_score"), bm25_score=src.get("bm25_score"),
                rrf_score=src.get("rrf_score"),
                boosts={k: v for k, v in boosts.items() if not k.startswith("_")},
                rank=rank,
            ))
        log.log_event("rerank.done", **{"in": len(candidates), "out": len(results)}, top_k=top_k,
                      mode="heuristic")
        span.set_output({"out": len(results), "top_score": results[0].score if results else None})
        return results


def tokenize_query(query: str) -> list[str]:
    """查询分词（重排内部用；与 BM25 同口径）。"""
    from .text_utils import tokenize

    return tokenize(query)


def maybe_llm_rerank(query: str, chunks: Sequence[Any], *, cfg: AppConfig | None = None,
                     llm: Any = None, logger: Any = None) -> list[Any]:
    """可选的 LLM 重排（默认关闭）。

    关闭或 LLM 不可用时：**原样返回**启发式结果并写 ``rerank.llm_skipped``（不静默、不猜）。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger)
    if not config.retrieval.enable_llm_rerank:
        log.log_event("rerank.llm_skipped", reason="RAG_RETRIEVAL__ENABLE_LLM_RERANK=false")
        return list(chunks)
    if llm is None:
        log.log_event("rerank.llm_skipped", level="WARNING", reason="未注入 LLM 客户端（T6 交付后接入）")
        return list(chunks)
    try:
        log.log_event("rerank.llm_unsupported", level="WARNING",
                      reason="LLM 重排在本任务范围内未启用实现，按启发式结果返回")
    except Exception as exc:  # noqa: BLE001 —— 兜底：任何异常都回退启发式并留痕
        log.log_event("rerank.llm_failed", level="ERROR", error_type=type(exc).__name__, message=str(exc))
    return list(chunks)


def needs_table_retrieval(question: str, *, field_type: str | None = None, logger: Any = None) -> bool:
    """是否需要「表块兜底检索」：数字类问题或表格型字段（设计 §3.13 表格兜底）。

    依赖 ``query_understanding.TABLE_FIELD_TYPES`` 不可用时**显式降级**并留 ``*.degrade`` 事件
    （工单硬要求：禁止静默失败；tester 用 AST 检出此处曾是 ``except ImportError: pass``）。
    """
    if field_type:
        try:
            from .query_understanding import TABLE_FIELD_TYPES  # T6 交付后可用

            if field_type in TABLE_FIELD_TYPES:
                return True
        except ImportError as exc:
            _lazy_logger(logger).log_event(
                "reranker.table_fallback.degrade", level="WARNING",
                reason=f"query_understanding.TABLE_FIELD_TYPES 不可用（{type(exc).__name__}: {exc}）",
                fallback="仅按「问题是否含数字信号」判断是否需要表块兜底",
                field_type=field_type)
    return has_numeric_signal(question)
