"""重排器：把混合召回的 top20 精排为 top5。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 检索优化（对应 设计/优化方案设计.md §2.3 M3.6、§3.4）

诚实性要求（重要）：
- 本机**断网且无重排模型权重**（环境事实 2.2），因此默认走 ``mode="rule"`` 的**规则重排**；
  只有当本地存在 CrossEncoder 模型目录时才切 ``mode="model"``。
- 日志、`health()`、评估报告必须如实标注使用的是哪种模式，**不得把规则重排描述为
  "使用 bge-reranker-base 重排"**。

规则打分（归一化到 [0,1] 后加权，权重合计 1.0）::

    0.25×关键词覆盖 + 0.20×目标数值覆盖 + 0.10×块类型先验
  + 0.15×位置先验 + 0.30×融合分归一        （再乘检索期释义/碎片降权系数）

> 权重相对设计初稿的修订理由（实测，不修饰）：初稿给"融合分归一"仅 0.10，
> 实测证据被释义块/同主题块挤出 top-5（531 由第 3 掉出、543 由第 11 掉出），
> 说明"语义+BM25 融合名次"信息量最大；修订后证据名次见
> ``研发/data/processed/selftest_retrieval_ranks.json`` 的逐题对照。
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.models.schemas import RetrievedChunk
from app.core.retrieval_utils import (
    matched_domain_keywords,
    numeric_coverage,
    percent_count,
    question_keywords,
    table_hint,
    value_coverage_score,
)

PERCENT_PATTERN = re.compile(r"%|％")


class Reranker:
    """重排器（规则模式默认；有本地模型时用 CrossEncoder）。"""

    def __init__(self, settings=None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._model_error = ""
        self._mode = self._resolve_mode()

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """当前模式：``model`` 或 ``rule``。"""
        return self._mode

    def _resolve_mode(self) -> str:
        """决定重排模式：``auto`` 时有可用本地模型目录才用 model。"""
        configured = self._settings.reranker.mode
        if configured == "off":
            return "off"
        if configured == "rule":
            return "rule"
        model_path = self._settings.reranker.model_path
        available = bool(model_path) and (Path(model_path).is_dir() or os.path.isdir(model_path))
        if not available:
            logger.warning(
                "app.core.reranker",
                "未找到本地重排模型目录，使用规则重排（mode=rule）",
                configured_mode=configured,
                model_path=model_path,
            )
            return "rule"
        if self._load_model():
            return "model"
        return "rule"

    def _load_model(self) -> bool:
        """尝试加载 CrossEncoder；失败自动切规则模式并记 WARNING（不得声称用了模型）。"""
        try:
            from sentence_transformers import CrossEncoder  # 局部导入

            self._model = CrossEncoder(self._settings.reranker.model_path)
            logger.info("app.core.reranker", "重排模型加载完成", model=self._settings.reranker.model_path)
            return True
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception(
                "app.core.reranker",
                "重排模型加载失败，自动切换为规则重排",
                model_path=self._settings.reranker.model_path,
            )
            return False

    # ------------------------------------------------------------------
    @trace
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        """对候选做精排并返回 top_n（不改变入参列表对象）。"""
        try:
            if not candidates:
                return []
            if self._mode == "off":
                return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_n]
            started = time.perf_counter()
            if self._mode == "model" and self._model is not None:
                ordered = self._rerank_model(query, candidates)
            else:
                ordered = self._rerank_rule(query, candidates)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if elapsed_ms > self._settings.reranker.budget_ms:
                logger.warning(
                    "app.core.reranker",
                    "重排耗时超出预算（不中断）",
                    mode=self._mode,
                    elapsed_ms=round(elapsed_ms, 2),
                    budget_ms=self._settings.reranker.budget_ms,
                    candidates=len(candidates),
                )
            logger.info(
                "app.core.reranker",
                "重排完成",
                mode=self._mode,
                candidates=len(candidates),
                top_n=top_n,
                elapsed_ms=round(elapsed_ms, 2),
            )
            return ordered[:top_n]
        except Exception:
            logger.exception("app.core.reranker", "重排失败，回退为融合分排序")
            return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_n]

    # ------------------------------------------------------------------
    def _rerank_model(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """CrossEncoder 打分（云端路径）。"""
        pairs = [(query, item.chunk.content[: self._settings.llm.max_chunk_chars]) for item in candidates]
        scores = self._model.predict(pairs)  # type: ignore[union-attr]
        values = [float(score) for score in scores]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        for item, value in zip(candidates, values):
            normalized = (value - low) / span
            item.rerank_score = round(normalized, 6)
            item.boosts["rerank_mode"] = 1.0
            item.boosts["fusion_score"] = round(item.score, 6)
            item.score = round(normalized, 6)
        return sorted(candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)

    def _rerank_rule(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """规则打分：关键词覆盖 + 目标数值覆盖 + 类型先验 + 位置先验 + 融合分归一。

        实测修订（工单2 自测）：把"融合分归一"的权重上调并乘上检索期降权系数，
        否则释义块/同主题块会靠关键词命中把真实证据挤出 top-5。
        """
        settings = self._settings.reranker
        keywords = question_keywords(query)
        top_fusion = max((item.score for item in candidates), default=1.0) or 1.0
        for item in candidates:
            chunk = item.chunk
            content = chunk.content
            # 1) 关键词覆盖
            if keywords:
                hits = matched_domain_keywords(keywords, content)
                keyword_score = len(hits) / len(keywords)
            else:
                keyword_score = 0.5  # 无领域关键词时给中性分
            # 2) 目标数值覆盖与多值齐备度
            numeric_score = numeric_coverage(query, content)
            value_score = value_coverage_score(query, content)
            # 3) 块类型先验：仅"表格型问题"（含表格提示词且要数值）才偏好表格
            if chunk.type == "table":
                table_question = table_hint(query) and numeric_score > 0
                type_score = 1.0 if table_question else 0.45
            else:
                type_score = 0.7
            # 4) 位置先验：标题栈命中块首 / 表格首行优先；释义块最低
            if chunk.is_boilerplate:
                position_score = 0.15
            elif chunk.section and any(
                segment.strip() and len(segment.strip()) >= 4 and segment.strip() in content[:80]
                for segment in chunk.section.split(" > ")
            ):
                position_score = 0.9
            elif chunk.type == "table":
                position_score = 0.65
            else:
                position_score = 0.55
            # 5) 融合分归一（语义 + BM25 的综合名次，信息量最大）
            fusion_score = item.score / top_fusion
            score = (
                settings.weight_keyword * keyword_score
                + settings.weight_numeric * numeric_score
                + settings.weight_value * value_score
                + settings.weight_type * type_score
                + settings.weight_position * position_score
                + settings.weight_fusion * fusion_score
            )
            # 6) 检索期降权系数（释义页 / 碎片）作为整体乘子：防止它们靠关键词翻上来
            penalty = 1.0
            if settings.apply_retrieval_penalty:
                penalty = min(
                    item.boosts.get("boilerplate_penalty", 1.0),
                    item.boosts.get("fragment_penalty", 1.0),
                )
                score *= penalty
            item.rerank_score = round(score, 6)
            item.boosts["rerank_rule_keyword"] = round(keyword_score, 4)
            item.boosts["rerank_rule_numeric"] = round(numeric_score, 4)
            item.boosts["rerank_rule_value"] = round(value_score, 4)
            item.boosts["rerank_rule_type"] = round(type_score, 4)
            item.boosts["rerank_rule_position"] = round(position_score, 4)
            item.boosts["rerank_rule_fusion"] = round(fusion_score, 4)
            item.boosts["rerank_rule_penalty"] = round(penalty, 4)
            item.boosts["fusion_score"] = round(item.score, 6)
            # 契约：score 置为重排后分数（融合分保留在 boosts.fusion_score 供审计）
            item.score = round(score, 6)
        return sorted(candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)

    # ------------------------------------------------------------------
    def health(self) -> dict[str, object]:
        """健康信息（必须如实反映 mode，供界面与报告标注）。"""
        return {
            "mode": self._mode,
            "model": self._settings.reranker.model_path if self._mode == "model" else "",
            "budget_ms": self._settings.reranker.budget_ms,
            "available": self._mode in {"model", "rule"},
            "model_configured": self._settings.reranker.model_path,
            "model_error": self._model_error,
            "note": "本机无重排模型权重，使用规则重排" if self._mode == "rule" else "",
        }


_reranker: Reranker | None = None
_lock = threading.Lock()


def get_reranker() -> Reranker:
    """获取进程级重排器单例。"""
    global _reranker
    with _lock:
        if _reranker is None:
            _reranker = Reranker()
    return _reranker


def reset_reranker() -> None:
    """重置单例（测试用）。"""
    global _reranker
    with _lock:
        _reranker = None
