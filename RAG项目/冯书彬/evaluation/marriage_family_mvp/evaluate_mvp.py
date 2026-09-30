from dataclasses import dataclass
from typing import Any

SCORING_CATEGORIES = {
    "basis_accuracy": 2,
    "citation_traceability": 2,
    "high_risk_handling": 2,
    "no_fabrication": 2,
    "follow_up_accuracy": 1,
    "answer_success": 1,
    "interface_success": 1,
}


@dataclass(frozen=True)
class EvaluationResult:
    # 评估结果只保存分数和缺口，不保存用户原始敏感提问。
    total_score: int
    max_score: int
    passed: bool
    missing_points: list[str]


def evaluate_answer(case: dict[str, Any], answer: dict[str, Any]) -> EvaluationResult:
    # MVP 骨架用显式期望点做确定性评分，后续可替换为人工或模型辅助评测。
    answer_text = str(answer.get("answer") or "")
    citations = answer.get("citations") or []
    missing_points: list[str] = []
    score = 0
    for point in case.get("expected_answer_points", []):
        if str(point) in answer_text:
            score += 1
        else:
            missing_points.append(str(point))
    if citations:
        score += 1
    max_score = len(case.get("expected_answer_points", [])) + 1
    return EvaluationResult(total_score=score, max_score=max_score, passed=score >= max_score, missing_points=missing_points)
