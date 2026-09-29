import asyncio
import importlib.util
import logging
import math
import numbers
import re
import sys
import time
import types
from dataclasses import asdict, dataclass

from app.rag.service import RAGService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", left.lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _install_ragas_compatibility_shim() -> None:
    """Bridge RAGAS 0.4.x to langchain-community versions that moved Vertex AI classes."""
    module_name = "langchain_community.chat_models.vertexai"
    if importlib.util.find_spec(module_name) is not None:
        return
    shim = types.ModuleType(module_name)
    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except ImportError:
        class ChatVertexAI:  # type: ignore[no-redef]
            pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def _rows(samples: list[EvaluationSample]) -> list[dict[str, object]]:
    return [
        {
            "user_input": sample.question,
            "response": sample.answer,
            "retrieved_contexts": sample.contexts,
            "reference": sample.ground_truth,
        }
        for sample in samples
    ]


def _proxy_metrics(samples: list[EvaluationSample]) -> dict:
    grounded = sum(
        1.0
        if sample.contexts
        and any(
            lexical_overlap(sample.answer, context) > 0 for context in sample.contexts
        )
        else 0.0
        for sample in samples
    ) / len(samples)
    relevance = sum(
        lexical_overlap(sample.answer, sample.question) for sample in samples
    ) / len(samples)
    reference_count = sum(bool(sample.ground_truth) for sample in samples)
    correctness = (
        sum(
            lexical_overlap(sample.answer, sample.ground_truth)
            for sample in samples
            if sample.ground_truth
        )
        / max(1, reference_count)
    )
    return {
        "provider": "local_proxy",
        "sample_count": len(samples),
        "metrics": {
            "faithfulness_proxy": round(grounded, 4),
            "answer_relevancy_proxy": round(relevance, 4),
            "answer_correctness_proxy": round(correctness, 4),
        },
        "samples": [asdict(sample) for sample in samples],
    }


async def _evaluate_with_ragas(
    rag_service: RAGService,
    samples: list[EvaluationSample],
) -> dict | None:
    """Run RAGAS in a worker thread so the async API event loop stays responsive."""
    started = time.perf_counter()
    settings = rag_service.settings
    try:
        _install_ragas_compatibility_shim()
        from openai import OpenAI  # type: ignore
        from ragas import EvaluationDataset, evaluate  # type: ignore
        from ragas.llms import llm_factory  # type: ignore
        from ragas.metrics.collections import ContextRelevance, Faithfulness  # type: ignore

        dataset = EvaluationDataset.from_list(_rows(samples))
        client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        evaluator_llm = llm_factory(
            settings.llm_model,
            provider="openai",
            client=client,
        )
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ContextRelevance(llm=evaluator_llm),
        ]
        result = await asyncio.to_thread(
            evaluate,
            dataset,
            metrics=metrics,
            llm=evaluator_llm,
            raise_exceptions=False,
            show_progress=False,
            allow_nest_asyncio=False,
        )

        metric_values: dict[str, float] = {}
        for metric_name in ("faithfulness", "context_relevance"):
            values = [
                row.get(metric_name)
                for row in result.scores
                if isinstance(row, dict)
            ]
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, numbers.Real) and math.isfinite(float(value))
            ]
            if numeric_values:
                metric_values[metric_name] = round(
                    sum(numeric_values) / len(numeric_values), 4
                )

        if not metric_values:
            raise RuntimeError("RAGAS returned no numeric metric scores")
        logger.info(
            "RAGAS evaluation complete",
            extra={
                "sample_count": len(samples),
                "metrics": metric_values,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "provider": "ragas",
            "sample_count": len(samples),
            "metrics": metric_values,
            "samples": [asdict(sample) for sample in samples],
        }
    except Exception as exc:
        logger.exception("RAGAS evaluation failed; using local proxy metrics: %s", exc)
        return None


async def evaluate_samples(
    rag_service: RAGService,
    samples: list[EvaluationSample],
) -> dict:
    """Use real RAGAS only when explicitly enabled; otherwise use offline proxy metrics."""
    if not samples:
        return {"provider": "none", "sample_count": 0, "metrics": {}}

    settings = rag_service.settings
    if (
        settings.ragas_enabled
        and settings.llm_provider.lower() != "mock"
        and settings.llm_api_key
    ):
        result = await _evaluate_with_ragas(rag_service, samples)
        if result is not None:
            return result
    elif settings.ragas_enabled:
        logger.warning(
            "RAGAS enabled but a non-mock LLM and LLM_API_KEY are required; using local proxy metrics"
        )
    else:
        logger.info("RAGAS disabled; using local proxy metrics")

    return _proxy_metrics(samples)
