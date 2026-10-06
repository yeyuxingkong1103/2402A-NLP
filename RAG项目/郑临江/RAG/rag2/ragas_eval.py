# -*- coding: utf-8 -*-
"""RAG 结果评估（RAGAS）。

封装 [RAGAS](https://docs.ragas.io) 0.4.x（现代组件版）对 RAG 输出打分的指标，
给离线包 ``OfflineRAG`` / 在线 ``HybridRetriever`` 提供「结果得分」评估能力：

    问题 + 检索上下文 + 生成答案 (+ 参考答案) → 各指标得分（0~1）

内置指标（名字 → 说明 / 所需字段）：

    faithfulness          忠实度        （答案是否忠于检索上下文）   需 LLM
    answer_relevancy      答案相关性    （答案是否紧扣问题）         需 LLM + 向量
    context_precision     上下文精确率  （检索上下文是否相关且有序）  需 LLM
    context_recall        上下文召回率  （参考答案能否由上下文推出）  需 LLM
    context_relevance     上下文相关性  （上下文是否切题）           需 LLM
    answer_correctness    答案正确性    （答案相对参考答案）         需 LLM + 向量
    context_entity_recall 上下文实体召回（关键实体是否被召回）       需 LLM
    semantic_similarity   语义相似度    （答案与参考答案语义相似度）   仅向量

用法示例（详见 README 第九节）：

    from rag2 import RagasEvaluator

    ev = RagasEvaluator()                     # 默认接本地 Ollama + 本地 bge-m3
    scores = ev.evaluate(
        question="什么食物富含钾元素？",
        answer="香蕉富含钾元素。",
        retrieved_contexts=["香蕉每100克含钾约256毫克。", "苹果含钾约107毫克。"],
        reference="香蕉是富含钾的水果。",
    )
    print(scores)                             # {'faithfulness': 1.0, 'answer_relevancy': 0.6, ...}

依赖：``ragas>=0.4``、``openai``、``sentence-transformers``（向量），运行环境 ``rags_``。
默认 LLM 为本地 Ollama（``http://localhost:11434/v1``）；对带思考能力的模型（Qwen3.5）
自动传 ``reasoning_effort="none"`` 关闭思考，避免思考链吞掉 token 导致结构化输出截断。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Sequence

logger = logging.getLogger("rag2.ragas_eval")

# ---- 默认后端（可被环境变量 / 构造参数覆盖） ----
DEFAULT_LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", "Qwen3.5:4B")
DEFAULT_LLM_BASE_URL = os.environ.get("RAGAS_LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_LLM_API_KEY = os.environ.get("RAGAS_LLM_API_KEY", "ollama")
DEFAULT_EMBED_MODEL = os.environ.get("RAGAS_EMBED_MODEL", "D:/modelscope/bge-m3")

# 指标规格：name -> (构造名, 所需输入字段, 需要 LLM, 需要向量)
_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "faithfulness":          {"fields": ("user_input", "response", "retrieved_contexts"), "llm": True,  "emb": False},
    "answer_relevancy":      {"fields": ("user_input", "response"),                      "llm": True,  "emb": True},
    "context_precision":     {"fields": ("user_input", "reference", "retrieved_contexts"), "llm": True,  "emb": False},
    "context_recall":        {"fields": ("user_input", "retrieved_contexts", "reference"), "llm": True,  "emb": False},
    "context_relevance":     {"fields": ("user_input", "retrieved_contexts"),             "llm": True,  "emb": False},
    "answer_correctness":    {"fields": ("user_input", "response", "reference"),          "llm": True,  "emb": True},
    "context_entity_recall": {"fields": ("reference", "retrieved_contexts"),              "llm": True,  "emb": False},
    "semantic_similarity":   {"fields": ("reference", "response"),                        "llm": False, "emb": True},
}

# 指标分组（便捷预设）
METRIC_GROUPS: dict[str, list[str]] = {
    "all":          list(_METRIC_SPECS),
    "generation":   ["faithfulness", "answer_relevancy", "answer_correctness", "semantic_similarity"],
    "retrieval":    ["context_precision", "context_recall", "context_relevance", "context_entity_recall"],
    "no_reference": ["faithfulness", "answer_relevancy", "context_relevance"],  # 无需 ground truth
}


def ragas_available() -> bool:
    """ragas 及依赖是否可导入（不真正加载模型）。"""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("ragas", "openai"))


def available_metrics() -> list[str]:
    """返回全部可评估指标名。"""
    return list(_METRIC_SPECS)


def _resolve_metric_names(metrics: str | Iterable[str] | None) -> list[str]:
    """把指标参数解析为标准名列表（支持分组名 / 单个名 / 列表）。"""
    if metrics is None:
        return list(_METRIC_SPECS)
    if isinstance(metrics, str):
        names = [metrics]
    else:
        names = list(metrics)
    resolved: list[str] = []
    for n in names:
        if n in METRIC_GROUPS:
            resolved.extend(x for x in METRIC_GROUPS[n] if x not in resolved)
        elif n in _METRIC_SPECS:
            if n not in resolved:
                resolved.append(n)
        else:
            logger.warning("未知指标，忽略：%s", n)
    return resolved


class RagasEvaluator:
    """RAGAS 评估器：把「问题 / 上下文 / 答案 / 参考答案」打成一串 0~1 的得分。

    参数：
        llm_model:        生成 LLM 模型名（OpenAI 兼容），默认本地 Ollama Qwen3.5:4B。
        llm_base_url:     OpenAI 兼容端点地址。
        llm_api_key:      API Key（Ollama 填任意占位即可）。
        llm_provider:     传给 ragas.llms.llm_factory 的 provider，默认 "openai"。
        embed_model:      sentence-transformers 向量模型路径（本地 bge-m3）；
                          为 None 且未提供 embeddings 时，跳过向量类指标。
        device:           embed_model 加载设备（cuda / cpu）。
        embeddings:       预构建的 ragas BaseRagasEmbedding，优先于 embed_model。
        llm:              预构建的 ragas BaseRagasLLM，优先于 llm_* 自动构建。
        temperature:      生成温度。
        max_tokens:       生成最大 token 数（结构化输出建议 ≥2048）。
        reasoning_effort: 思考强度；本地 Ollama 思考模型填 "none" 关闭思考，
                          接 OpenAI 等端点可设 None 省略该参数。
        timeout:          LLM 请求超时秒数。
        **llm_kwargs:     其余透传给 llm_factory（如 top_p 等）。
    """

    def __init__(
        self,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_provider: str = "openai",
        embed_model: str | None = DEFAULT_EMBED_MODEL,
        device: str = "cpu",
        embeddings: Any = None,
        llm: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        reasoning_effort: str | None = "none",
        timeout: float = 600.0,
        **llm_kwargs: Any,
    ) -> None:
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
        self.llm_api_key = llm_api_key if llm_api_key is not None else DEFAULT_LLM_API_KEY
        self.llm_provider = llm_provider
        self.embed_model = embed_model
        self.device = device
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self.llm_kwargs = dict(llm_kwargs)

        # 预构建对象（优先级高于自动构建）
        self._llm = llm
        self._embeddings = embeddings

    # ------------------------------------------------------------------ 后端构建
    def _build_llm(self) -> Any:
        """构建一个 OpenAI 兼容的 ragas LLM（每次新建，避免跨事件循环复用）。"""
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory

        client = AsyncOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            timeout=self.timeout,
        )
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.llm_kwargs,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return llm_factory(self.llm_model, client=client, provider=self.llm_provider, **kwargs)

    def _ensure_embeddings(self) -> Any | None:
        """懒加载向量模型（可跨多次调用复用，无事件循环绑定）。"""
        if self._embeddings is not None:
            return self._embeddings
        if not self.embed_model:
            return None
        from ragas.embeddings import HuggingFaceEmbeddings

        self._embeddings = HuggingFaceEmbeddings(
            model=self.embed_model, device=self.device, normalize_embeddings=True
        )
        logger.info("加载评估向量模型：%s（device=%s）", self.embed_model, self.device)
        return self._embeddings

    def _instantiate(self, name: str, llm: Any, emb: Any | None):
        """按名构造单个指标实例。"""
        from ragas.metrics.collections import (  # 现代组件版公开导出
            AnswerCorrectness,
            AnswerRelevancy,
            ContextEntityRecall,
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
            Faithfulness,
            SemanticSimilarity,
        )

        ctor = {
            "faithfulness": lambda: Faithfulness(llm),
            "answer_relevancy": lambda: AnswerRelevancy(llm, emb),
            "context_precision": lambda: ContextPrecision(llm),
            "context_recall": lambda: ContextRecall(llm),
            "context_relevance": lambda: ContextRelevance(llm),
            "answer_correctness": lambda: AnswerCorrectness(llm, emb),
            "context_entity_recall": lambda: ContextEntityRecall(llm),
            "semantic_similarity": lambda: SemanticSimilarity(emb),
        }[name]
        return ctor()

    # ------------------------------------------------------------------ 评估
    def evaluate(
        self,
        question: str,
        answer: str | None = None,
        retrieved_contexts: Sequence[str] | None = None,
        reference: str | None = None,
        metrics: str | Iterable[str] | None = None,
    ) -> dict[str, float | None]:
        """评估单条 RAG 结果，返回 ``{指标名: 得分}``（无法计算的指标为 None）。

        参数：
            question:           用户问题。
            answer:             生成答案（缺省则跳过需要答案的指标）。
            retrieved_contexts: 检索到的上下文列表。
            reference:          参考答案 / ground truth（缺省则跳过需要参考答案的指标）。
            metrics:            指标名或分组名（"all"/"generation"/"retrieval"/"no_reference"）
                                或它们的列表；默认全部。
        """
        row = self._make_row(question, answer, retrieved_contexts, reference)
        names = _resolve_metric_names(metrics)
        available = {"user_input", "response", "retrieved_contexts", "reference"} & {
            k for k, v in row.items() if v is not None and v != ""
        }
        llm = self._build_llm()
        # 只为「实际会运行且依赖向量」的指标加载向量模型
        need_emb = any(
            _METRIC_SPECS[n]["emb"] and all(f in available for f in _METRIC_SPECS[n]["fields"])
            for n in names
        )
        emb = self._ensure_embeddings() if need_emb else None
        metrics_objs = self._collect_metrics(names, llm, emb, available)
        if not metrics_objs:
            return {}
        return self._score(metrics_objs, row)[0]

    def evaluate_batch(
        self,
        rows: Sequence[dict],
        metrics: str | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """批量评估。

        参数：
            rows: 记录列表，每条包含键
                  ``question`` / ``answer`` / ``retrieved_contexts`` / ``reference``
                  （缺字段的指标自动跳过）。
            metrics: 同 ``evaluate``。

        返回：
            记录列表，每条在原字段基础上追加各指标得分列。
        """
        rows = list(rows)
        if not rows:
            return []
        names = _resolve_metric_names(metrics)

        # 归一化友好键名 -> ragas 字段名（question->user_input, answer->response）
        norm_rows = [self._normalize_row(r) for r in rows]

        # 以「所有行字段的并集」判断哪些指标可算，避免一行缺字段导致整体失败
        union: set[str] = set()
        for r in norm_rows:
            for k in ("user_input", "response", "retrieved_contexts", "reference"):
                if r.get(k) not in (None, "", []):
                    union.add(k)
        need_emb = any(
            _METRIC_SPECS[n]["emb"] and all(f in union for f in _METRIC_SPECS[n]["fields"])
            for n in names
        )
        llm = self._build_llm()
        emb = self._ensure_embeddings() if need_emb else None
        metrics_objs = self._collect_metrics(names, llm, emb, union)

        # 单次事件循环内并跑「行 × 指标」，规避 AsyncOpenAI 跨事件循环复用问题
        import asyncio
        import inspect

        async def run_all() -> list[list[Any]]:
            coros = []
            for m in metrics_objs:
                params = inspect.signature(m.ascore).parameters
                for r in norm_rows:
                    kw = {k: r[k] for k in params if k in r}
                    coros.append(m.ascore(**kw))
            gathered = await asyncio.gather(*coros, return_exceptions=True)
            # 每行一条结果，顺序：指标外层 × 行内层
            per_row: list[list[Any]] = [[] for _ in norm_rows]
            idx = 0
            for m in metrics_objs:
                for i in range(len(norm_rows)):
                    per_row[i].append((m.name, gathered[idx]))
                    idx += 1
            return per_row

        per_row = asyncio.run(run_all())
        out: list[dict[str, Any]] = []
        for i, r in enumerate(rows):  # 输出保留原始键，仅追加得分列
            item = dict(r)
            for name, res in per_row[i]:
                item[name] = None if isinstance(res, Exception) else res.value
            out.append(item)
        return out

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _make_row(
        question: str,
        answer: str | None,
        retrieved_contexts: Sequence[str] | None,
        reference: str | None,
    ) -> dict[str, Any]:
        return {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": list(retrieved_contexts) if retrieved_contexts else None,
            "reference": reference,
        }

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        """把友好键名（question/answer）补成 ragas 字段名（user_input/response）。"""
        r = dict(row)
        if "user_input" not in r and "question" in r:
            r["user_input"] = r["question"]
        if "response" not in r and "answer" in r:
            r["response"] = r["answer"]
        return r

    def _collect_metrics(
        self,
        names: list[str],
        llm: Any,
        emb: Any | None,
        available_fields: set[str],
    ) -> list[Any]:
        """按可用字段与后端，过滤并实例化可计算的指标。"""
        metrics_objs: list[Any] = []
        for name in names:
            spec = _METRIC_SPECS[name]
            missing = [f for f in spec["fields"] if f not in available_fields]
            if missing:
                logger.info("跳过 %s：缺少字段 %s", name, missing)
                continue
            if spec["emb"] and emb is None:
                logger.info("跳过 %s：未配置向量模型", name)
                continue
            metrics_objs.append(self._instantiate(name, llm, emb))
        return metrics_objs

    @staticmethod
    def _score(metrics_objs: list[Any], row: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, str]]:
        """在单个事件循环内并发打分，返回 (得分, 错误信息)。"""
        import asyncio
        import inspect

        async def run():
            coros = []
            for m in metrics_objs:
                params = inspect.signature(m.ascore).parameters
                kw = {k: row[k] for k in params if k in row}
                coros.append(m.ascore(**kw))
            return await asyncio.gather(*coros, return_exceptions=True)

        results = asyncio.run(run())
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for m, res in zip(metrics_objs, results):
            if isinstance(res, Exception):
                scores[m.name] = None
                errors[m.name] = f"{type(res).__name__}: {res}"
                logger.warning("指标 %s 计算失败：%s", m.name, res)
            else:
                scores[m.name] = res.value
        return scores, errors


def evaluate(
    question: str,
    answer: str | None = None,
    retrieved_contexts: Sequence[str] | None = None,
    reference: str | None = None,
    metrics: str | Iterable[str] | None = None,
    evaluator: RagasEvaluator | None = None,
) -> dict[str, float | None]:
    """单条 RAG 结果打分的模块级便捷函数（等价于 ``RagasEvaluator().evaluate(...)``）。"""
    ev = evaluator or RagasEvaluator()
    return ev.evaluate(
        question=question,
        answer=answer,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
        metrics=metrics,
    )


def evaluate_rag(
    rag: Any,
    question: str,
    answer: str | None = None,
    reference: str | None = None,
    top_k: int = 3,
    mode: str = "hybrid",
    metrics: str | Iterable[str] | None = None,
    evaluator: RagasEvaluator | None = None,
) -> dict[str, float | None]:
    """对某个检索器（OfflineRAG / HybridRetriever）检索结果 + 答案打分。

    先用 ``rag.search(question, top_k, mode)`` 取回上下文，再调用 ``evaluate``。
    返回 ``{指标名: 得分}``。
    """
    hits = rag.search(question, top_k=top_k, mode=mode)
    contexts = [h.text for h in hits]
    ev = evaluator or RagasEvaluator()
    return ev.evaluate(
        question=question,
        answer=answer,
        retrieved_contexts=contexts,
        reference=reference,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- 自测
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("ragas 可用：", ragas_available())
    print("可用指标：", available_metrics())
    print("指标分组：", METRIC_GROUPS)

    # 端到端演示：需本地 Ollama（Qwen3.5:4B）与本地 bge-m3 权重
    ev = RagasEvaluator()
    scores = ev.evaluate(
        question="什么食物富含钾元素？",
        answer="香蕉富含钾元素。",
        retrieved_contexts=["香蕉每100克含钾约256毫克，是常见水果中钾含量较高的。", "苹果含钾约107毫克。"],
        reference="香蕉是富含钾的水果，钾含量约256mg/100g。",
        metrics="all",
    )
    print("\n单条评估得分：")
    for k, v in scores.items():
        print(f"  {k:22s} = {v}")

    print("\n批量评估：")
    rows = [
        {"question": "什么食物富含钾元素？", "answer": "香蕉富含钾元素。",
         "retrieved_contexts": ["香蕉含钾高。", "苹果含钾较低。"], "reference": "香蕉富含钾。"},
        {"question": "什么食物富含钾元素？", "answer": "苹果富含钾元素。",
         "retrieved_contexts": ["香蕉含钾高。", "苹果含钾较低。"], "reference": "香蕉富含钾。"},
    ]
    for r in ev.evaluate_batch(rows, metrics="generation"):
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
