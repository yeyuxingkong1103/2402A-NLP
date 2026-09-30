#!/usr/bin/env python3
"""Run a real RAGAS evaluation against the configured Milvus/DeepSeek stack."""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def install_ragas_compat() -> None:
    """Bridge RAGAS 0.4.3's stale optional Vertex import."""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:
        if str(exc).split("No module named ")[-1].strip("'") != module_name:
            raise
        from langchain_core.language_models import BaseChatModel

        shim = types.ModuleType(module_name)
        shim.ChatVertexAI = type("ChatVertexAI", (BaseChatModel,), {})
        sys.modules[module_name] = shim


def load_config() -> tuple[Any, list[dict[str, str]]]:
    from app.core.config import get_settings

    dataset_path = ROOT / "data/evaluation/ragas_dataset.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return get_settings(), payload["samples"]


def build_samples(settings: Any, test_cases: list[dict[str, str]]) -> list[dict[str, Any]]:
    from app.rag.retriever import KnowledgeRetriever
    from app.security.safety import CRISIS_RESPONSE, is_crisis_message
    from app.services.deepseek_client import DeepSeekClient

    retriever = KnowledgeRetriever(settings)
    client = DeepSeekClient(settings)
    rows = []
    for index, case in enumerate(test_cases, start=1):
        question = case["question"]
        contexts = retriever.retrieve(question, top_k=5)
        context_texts = [item["text"] for item in contexts]
        if is_crisis_message(question):
            answer = CRISIS_RESPONSE
        else:
            answer = client.answer(question, [], contexts)
        rows.append(
            {
                "id": case["id"],
                "question": question,
                "reference": case["reference"],
                "answer": answer,
                "contexts": context_texts,
                "context_count": len(context_texts),
            }
        )
        print(f"[{index}/{len(test_cases)}] {case['id']}: contexts={len(context_texts)}")
    return rows


def run_ragas(rows: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    install_ragas_compat()
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._nv_metrics import ContextRelevance
    from ragas.run_config import RunConfig
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    class LegacyEmbeddingAdapter:
        """Expose the LangChain embedding methods expected by old RAGAS metrics."""

        def __init__(self, embedding: Any) -> None:
            self.embedding = embedding

        def embed_query(self, text: str) -> list[float]:
            return self.embedding.embed_text(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embedding.embed_texts(texts)

    llm_client = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.01,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )
    llm = LangchainLLMWrapper(llm_client, bypass_n=True)
    base_embeddings = HuggingFaceEmbeddings(
        model=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
    )
    embeddings = LegacyEmbeddingAdapter(base_embeddings)
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            reference=row["reference"],
            retrieved_contexts=row["contexts"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        ContextRelevance(),
    ]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=180, max_retries=2, max_workers=2, max_wait=8),
        raise_exceptions=False,
        show_progress=True,
    )
    result_dict = result.to_pandas().to_dict(orient="records")
    if "nv_context_relevance" in result_dict[0]:
        for record in result_dict:
            record["context_relevancy"] = record.pop("nv_context_relevance")
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevancy",
    ]
    averages = {}
    for name in metric_names:
        values = [record.get(name) for record in result_dict]
        numeric = [
            float(value)
            for value in values
            if value is not None and not (isinstance(value, float) and math.isnan(value))
        ]
        averages[name] = sum(numeric) / len(numeric) if numeric else None
    return {"rows": result_dict, "averages": averages}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/evaluation/ragas_latest.json")
    args = parser.parse_args()
    settings, test_cases = load_config()
    rows = build_samples(settings, test_cases)
    evaluation = run_ragas(rows, settings)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/evaluation/ragas_dataset.json",
        "sample_count": len(rows),
        "metrics": evaluation["averages"],
        "samples": evaluation["rows"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RAGAS 平均指标 ===")
    for name, value in report["metrics"].items():
        print(f"{name}: {value:.4f}" if value is not None else f"{name}: N/A")
    print(f"\n逐题报告已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
