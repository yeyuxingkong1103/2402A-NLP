from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.config import AppSettings
    from backend.app.embeddings import BgeM3Embedder
    from backend.app.storage import JsonStateStore
    from backend.app.vector_store import QdrantVectorStore

# 评测集类型
SET_RETRIEVAL = "retrieval"
SET_GENERATION = "generation"
SET_SAFETY = "safety"

SET_FILES = ["retrieval_eval.json", "generation_eval.json", "safety_eval.json"]

# 三个标准指标：召回率、MRR、拒答率（另有 generation_coverage 作生成集补充）
METRIC_RECALL_AT_K = "recall_at_k"
METRIC_MRR = "mrr"
METRIC_REFUSAL_ACC = "refusal_accuracy"
METRIC_GEN_COVERAGE = "generation_coverage"


@dataclass(frozen=True)
class EvalOutcome:
    """单条评测结果（由 evaluator 产出）。"""

    answer: str
    citations: list[Any]
    fallback: bool
    latency_s: float


Evaluator = Callable[[dict[str, Any]], EvalOutcome]


def load_eval_items(eval_path: Path) -> list[dict[str, Any]]:
    """加载评测集条目，兼容旧版顶层列表与新版 {description, version, items}。"""
    with eval_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    return list(data.get("items", []))


def infer_set_type(items: list[dict[str, Any]]) -> str:
    """根据条目字段推断评测集类型。"""
    first = items[0] if items else {}
    if "expected_page" in first:
        return SET_RETRIEVAL
    if "answer_points" in first:
        return SET_GENERATION
    return SET_SAFETY


def _normalize_citations(citations: Sequence[Any]) -> list[dict[str, Any]]:
    from collections.abc import Sequence

    normalized: list[dict[str, Any]] = []
    for citation in citations:
        if hasattr(citation, "model_dump"):
            normalized.append(citation.model_dump(mode="json"))
        elif isinstance(citation, dict):
            normalized.append(citation)
        elif hasattr(citation, "__dict__"):
            normalized.append(dict(citation.__dict__))
        else:
            normalized.append(dict(citation))
    return normalized


def _retrieved_pages(citations: list[dict[str, Any]]) -> list[int]:
    pages: list[int] = []
    for citation in citations:
        try:
            page_number = int(citation.get("page", 0))
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            pages.append(page_number)
    return pages


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def score_item(item: dict[str, Any], outcome: EvalOutcome, set_type: str) -> dict[str, Any]:
    """按评测集类型打分，输出标准指标（recall@k / mrr / refusal_accuracy）。"""
    citations = _normalize_citations(outcome.citations)
    pages = _retrieved_pages(citations)

    if set_type == SET_RETRIEVAL:
        if item.get("should_refuse"):
            # 拒答用例：合规则 refusal_accuracy 记 1
            ok = 1.0 if outcome.fallback else 0.0
            return {METRIC_RECALL_AT_K: None, METRIC_MRR: None, METRIC_REFUSAL_ACC: ok, "expected_page": None}
        expected_page = item.get("expected_page")
        rank = next((index for index, page in enumerate(pages, start=1) if page == expected_page), None)
        recall_at_k = 1.0 if rank else 0.0
        mrr = 1.0 / rank if rank else 0.0
        return {METRIC_RECALL_AT_K: recall_at_k, METRIC_MRR: mrr, METRIC_REFUSAL_ACC: None, "expected_page": expected_page}

    if set_type == SET_GENERATION:
        answer = (outcome.answer or "").strip()
        points = [str(p) for p in (item.get("answer_points") or []) if p]
        if outcome.fallback or not answer or not points:
            matched = 0
        else:
            matched = sum(1 for point in points if point and point in answer)
        coverage = round(matched / len(points), 4) if points else 0.0
        return {
            METRIC_GEN_COVERAGE: coverage,
            "answer_points_total": len(points),
            "answer_points_matched": matched,
        }

    # safety
    if item.get("should_refuse"):
        ok = 1.0 if outcome.fallback else 0.0
    else:
        ok = 1.0 if not outcome.fallback else 0.0
    return {METRIC_REFUSAL_ACC: ok, "expected_page": None}


def run_eval_set(eval_path: Path, output_path: Path, evaluator: Evaluator, set_type: str) -> dict[str, Any]:
    items = load_eval_items(eval_path)
    scored = []
    for item in items:
        outcome = evaluator(item)
        score = score_item(item, outcome, set_type)
        scored.append(
            {
                "id": item.get("id", item.get("question", "")),
                "category": item.get("category"),
                "question": item.get("question"),
                "answer": outcome.answer,
                "fallback": outcome.fallback,
                "latency_s": round(outcome.latency_s, 4),
                "citations": _normalize_citations(outcome.citations),
                **score,
            }
        )

    recall_vals = [s[METRIC_RECALL_AT_K] for s in scored if s.get(METRIC_RECALL_AT_K) is not None]
    mrr_vals = [s[METRIC_MRR] for s in scored if s.get(METRIC_MRR) is not None]
    refusal_vals = [s[METRIC_REFUSAL_ACC] for s in scored if s.get(METRIC_REFUSAL_ACC) is not None]
    gen_vals = [s[METRIC_GEN_COVERAGE] for s in scored if s.get(METRIC_GEN_COVERAGE) is not None]

    summary = {
        "total": len(scored),
        METRIC_RECALL_AT_K: _mean(recall_vals),
        f"{METRIC_RECALL_AT_K}_count": len(recall_vals),
        METRIC_MRR: _mean(mrr_vals),
        f"{METRIC_MRR}_count": len(mrr_vals),
        METRIC_REFUSAL_ACC: _mean(refusal_vals),
        f"{METRIC_REFUSAL_ACC}_count": len(refusal_vals),
        METRIC_GEN_COVERAGE: _mean(gen_vals),
        f"{METRIC_GEN_COVERAGE}_count": len(gen_vals),
        "avg_latency_s": _mean([s["latency_s"] for s in scored]),
    }
    payload = {
        "source": str(eval_path),
        "set_type": set_type,
        "summary": summary,
        "items": scored,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _aggregate_payload(per_set_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总多套评测集的召回率/MRR/拒答率（按各自有效题数加权，避免被无关集拖低）。"""
    total_items = sum(item["summary"]["total"] for item in per_set_payloads)
    divisor = max(1, total_items)

    def weighted(metric: str) -> float:
        values = [
            (item["summary"].get(metric), item["summary"].get(f"{metric}_count", 0))
            for item in per_set_payloads
        ]
        values = [(value, count) for value, count in values if value is not None and count]
        if not values:
            return 0.0
        return round(sum(value * count for value, count in values) / sum(count for _, count in values), 4)

    return {
        "sets": per_set_payloads,
        "summary": {
            "total_sets": len(per_set_payloads),
            "total_items": total_items,
            METRIC_RECALL_AT_K: weighted(METRIC_RECALL_AT_K),
            METRIC_MRR: weighted(METRIC_MRR),
            METRIC_REFUSAL_ACC: weighted(METRIC_REFUSAL_ACC),
            METRIC_GEN_COVERAGE: weighted(METRIC_GEN_COVERAGE),
            "avg_latency_s": round(
                sum(item["summary"]["avg_latency_s"] * item["summary"]["total"] for item in per_set_payloads)
                / divisor,
                4,
            ),
        },
    }


def _write_comparison(v1_payload: dict[str, Any], v2_payload: dict[str, Any], output_path: Path) -> None:
    """写出 v1/v2 召回率/MRR/拒答率对比报告。"""
    s1 = v1_payload["summary"]
    s2 = v2_payload["summary"]
    metrics = (METRIC_RECALL_AT_K, METRIC_MRR, METRIC_REFUSAL_ACC)
    comparison = {
        "baseline": "v1",
        "metrics": list(metrics),
        "v1": {key: s1.get(key) for key in metrics},
        "v2": {key: s2.get(key) for key in metrics},
        "delta": {
            key: round((s2.get(key) or 0.0) - (s1.get(key) or 0.0), 4)
            for key in metrics
        },
        "latency": {
            "v1_avg_s": s1.get("avg_latency_s"),
            "v2_avg_s": s2.get("avg_latency_s"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_item_evaluator(
    settings: "AppSettings",
    store: "JsonStateStore",
    embedder: "BgeM3Embedder",
    vector_store: "QdrantVectorStore",
    variant: str,
    set_type: str,
) -> Evaluator:
    from backend.app.qa import QaService, build_citations
    from backend.app.rerank import RetrievalChain, CoarseReranker, FineReranker, build_coarse_scorer

    document_names = {document.document_id: document.file_name for document in store.list_documents()}

    chain = None
    if variant == "v2":
        coarse = CoarseReranker(build_coarse_scorer(settings), settings.coarse_top_k)
        fine = FineReranker(
            settings.reranker_model_path,
            settings.rerank_batch_size,
            settings.final_top_k,
            settings.reranker_device,
        )
        chain = RetrievalChain(vector_store, embedder, settings, coarse, fine)

    qa = None
    if set_type in (SET_GENERATION, SET_SAFETY):
        qa = QaService(settings.ollama_model, settings.ollama_base_url)

    def evaluate_item(item: dict[str, Any]) -> EvalOutcome:
        question = str(item.get("question", "")).strip()
        if not question:
            return EvalOutcome(answer="", citations=[], fallback=True, latency_s=0.0)
        start = time.perf_counter()
        if chain is not None:
            results = chain.retrieve(question)
        else:
            results = vector_store.search(question, embedder, settings.final_top_k)
            results = [result for result in results if result.score >= settings.min_retrieval_score]
        citations = build_citations(results, document_names)
        answer = ""
        fallback = not citations
        if qa is not None and citations:
            response = qa.answer(question, citations)
            answer = response.answer
            fallback = bool(response.fallback)
        latency_s = time.perf_counter() - start
        return EvalOutcome(answer=answer, citations=citations, fallback=fallback, latency_s=latency_s)

    return evaluate_item


def build_and_run_eval(pdf_path: Path, eval_sets_dir: Path, variant: str = "v2") -> dict[str, Any]:
    # NOTE: 直接评估当前真实检索库（data/qdrant + data/state.json），不再 reset/rebuild，
    # 避免测评删掉用户已入库的数据。pdf_path 参数保留仅为 CLI/测试兼容，不再用于建库。
    from backend.app.config import AppSettings
    from backend.app.embeddings import BgeM3Embedder
    from backend.app.storage import JsonStateStore
    from backend.app.vector_store import QdrantVectorStore

    settings = AppSettings()

    store = JsonStateStore(Path("data/state.json"))
    embedder = BgeM3Embedder(settings.bge_m3_model_path)
    vector_store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection)

    if not store.list_documents():
        print("!! 真实检索库为空：请先上传并构建文档（例如 SF6 标准 PDF）后再运行测评。")

    results_dir = Path("eval/results")
    baseline_dir = Path("eval/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    def run_sets(which: str) -> dict[str, Any]:
        per_set_payloads = []
        for name in SET_FILES:
            set_path = eval_sets_dir / name
            items = load_eval_items(set_path)
            set_type = infer_set_type(items)
            evaluator = _build_item_evaluator(settings, store, embedder, vector_store, variant=which, set_type=set_type)
            payload = run_eval_set(
                set_path,
                results_dir / f"{name.replace('.json', '')}_metrics.json",
                evaluator,
                set_type,
            )
            per_set_payloads.append(payload)
        return _aggregate_payload(per_set_payloads)

    def write_payload(payload: dict[str, Any], target: Path) -> None:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if variant == "v1":
        payload = run_sets("v1")
        write_payload(payload, results_dir / "v1_metrics.json")
        write_payload(payload, baseline_dir / "v1_metrics.json")
        return payload

    if variant == "both":
        v1_payload = run_sets("v1")
        v2_payload = run_sets("v2")
        write_payload(v1_payload, results_dir / "v1_metrics.json")
        write_payload(v2_payload, results_dir / "v2_metrics.json")
        _write_comparison(v1_payload, v2_payload, results_dir / "v1_vs_v2_comparison.json")
        return v2_payload

    v2_payload = run_sets("v2")
    write_payload(v2_payload, results_dir / "v2_metrics.json")
    baseline_path = baseline_dir / "v1_metrics.json"
    if baseline_path.exists():
        v1_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        _write_comparison(v1_payload, v2_payload, results_dir / "v1_vs_v2_comparison.json")
    return v2_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 评测（recall@k / MRR / refusal_accuracy）")
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "both"],
        default=os.environ.get("SF6_EVAL_VARIANT", "v2"),
        help="评测变体：v1（基线）、v2（粗排+精排）、both（两版对比）",
    )
    args, _ = parser.parse_known_args()

    pdf_path = Path(
        os.environ.get(
            "SF6_EVAL_PDF_PATH",
            r"C:\Users\tirito\Downloads\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf",
        )
    )
    eval_sets_dir = Path(os.environ.get("SF6_EVAL_SETS_DIR", "eval/sets"))
    build_and_run_eval(pdf_path, eval_sets_dir, variant=args.variant)


if __name__ == "__main__":
    main()
