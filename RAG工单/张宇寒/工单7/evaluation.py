from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "evaluation_questions.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if len(questions) != 10:
        raise ValueError("固定评测集必须恰好包含10道题")
    if len({item.get("id") for item in questions}) != 10:
        raise ValueError("评测题ID必须唯一")
    return questions


def _targets(question: dict) -> list[tuple[str, int]]:
    if question.get("targets"):
        return [
            (target["stock_code"], int(page))
            for target in question["targets"]
            for page in target.get("pages", [])
        ]
    return [
        (code, int(page))
        for code in question.get("stock_codes", [])
        for page in question.get("target_pages", [])
    ]


def score_retrieval(hits: list[dict], question: dict) -> dict:
    targets = set(_targets(question))
    allowed_codes = {code for code, _ in targets}
    fact_groups = question.get("expected_facts", [])
    relevant_ranks: list[int] = []
    found: set[tuple[str, int]] = set()
    found_facts: set[int] = set()
    for rank, hit in enumerate(hits, 1):
        key = (str(hit.get("stock_code", "")), int(hit.get("page", 0)))
        searchable = _compact(f"{hit.get('document', '')} {hit.get('text', '')}")
        matching_facts = {
            index
            for index, alternatives in enumerate(fact_groups)
            if any(_compact(option) in searchable for option in alternatives)
        }
        fact_relevant = (not allowed_codes or key[0] in allowed_codes) and bool(matching_facts)
        if key in targets or fact_relevant:
            relevant_ranks.append(rank)
        if key in targets:
            found.add(key)
        if fact_relevant:
            found_facts.update(matching_facts)
    recall = (
        len(found_facts) / len(fact_groups)
        if fact_groups
        else (len(found) / len(targets) if targets else 0.0)
    )
    return {
        "hit": int(bool(relevant_ranks)),
        "recall": round(recall, 4),
        "reciprocal_rank": round(1 / relevant_ranks[0], 4) if relevant_ranks else 0.0,
    }


def _compact(text: str) -> str:
    return "".join(str(text).replace(",", "").split()).lower()


def score_facts(answer: str, expected_facts: list[list[str]]) -> dict:
    normalized = _compact(answer)
    matched = sum(any(_compact(option) in normalized for option in alternatives) for alternatives in expected_facts)
    total = len(expected_facts)
    return {"matched": matched, "total": total, "accuracy": round(matched / total, 4) if total else 0.0}


async def evaluate_all(rag, questions: list[dict] | None = None, top_k: int = 5, mode: str = "hybrid") -> dict:
    items = questions or load_questions()
    results: list[dict] = []
    for item in items:
        try:
            answer = await rag.answer(item["question"], top_k=top_k, mode=mode)
            retrieval_score = score_retrieval(answer["hits"], item)
            fact_score = score_facts(answer["answer"], item["expected_facts"])
            results.append(
                {
                    **item,
                    **answer,
                    "status": "ok",
                    "retrieval_score": retrieval_score,
                    "fact_score": fact_score,
                }
            )
        except Exception as exc:
            results.append({**item, "status": "error", "error": str(exc), "hits": []})

    completed = [row for row in results if row["status"] == "ok"]
    retrieval = [row["retrieval_score"] for row in completed]
    times = [float(row.get("total_ms", 0.0)) for row in completed]
    sorted_times = sorted(times)
    p95_index = max(0, min(len(sorted_times) - 1, int(len(sorted_times) * 0.95))) if sorted_times else 0
    summary = {
        "total": len(results),
        "completed": len(completed),
        "hit_at_k": round(statistics.mean(row["hit"] for row in retrieval), 4) if retrieval else 0.0,
        "recall_at_k": round(statistics.mean(row["recall"] for row in retrieval), 4) if retrieval else 0.0,
        "mrr": round(statistics.mean(row["reciprocal_rank"] for row in retrieval), 4) if retrieval else 0.0,
        "fact_accuracy": round(statistics.mean(row["fact_score"]["accuracy"] for row in completed), 4) if completed else 0.0,
        "average_ms": round(statistics.mean(times), 2) if times else 0.0,
        "p95_ms": round(sorted_times[p95_index], 2) if sorted_times else 0.0,
    }
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {"top_k": top_k, "mode": mode},
        "summary": summary,
        "results": results,
    }
