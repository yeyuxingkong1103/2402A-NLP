"""Question-level deduplication and conservative near-duplicate discovery."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable

_ACTION_TERMS = (
    "提供", "申请", "联系", "拍照", "照片", "订单", "退款", "退货", "换货",
    "处理", "审核", "寄回", "运费", "维修", "重新发货",
)


def _answer_score(record: Any) -> tuple[int, int, int]:
    """Prefer actionable, complete answers and use source order as a tie-breaker."""
    answer = record.bot
    action_count = sum(term in answer for term in _ACTION_TERMS)
    useful_length = min(len(answer), 300)
    return action_count, useful_length, -record.source_line


def duplicate_groups(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Describe every exact normalized-question group without discarding variants."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.question_group_id].append(record)
    output = []
    for group_id, variants in grouped.items():
        if len(variants) < 2:
            continue
        canonical = max(variants, key=_answer_score)
        output.append({
            "question_group_id": group_id,
            "question": variants[0].user,
            "variant_count": len(variants),
            "selected_source_line": canonical.source_line,
            "source_lines": [item.source_line for item in variants],
            "scenes": sorted({item.scene for item in variants}),
            "answers": [
                {
                    "source_line": item.source_line,
                    "answer": item.bot,
                    "selected": item is canonical,
                }
                for item in variants
            ],
        })
    return sorted(output, key=lambda item: (-item["variant_count"], item["question_group_id"]))


def select_canonical_questions(records: Iterable[Any]) -> list[Any]:
    """Select one deterministic high-quality answer for each normalized question."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    order: list[str] = []
    for record in records:
        if record.question_group_id not in grouped:
            order.append(record.question_group_id)
        grouped[record.question_group_id].append(record)
    return [max(grouped[group_id], key=_answer_score) for group_id in order]


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    length_ratio = min(len(left), len(right)) / max(len(left), len(right))
    if length_ratio < 0.72:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def near_duplicate_pairs(
    records: list[Any], *, threshold: float = 0.92, max_pairs: int = 5000
) -> list[dict[str, Any]]:
    """Find likely paraphrases for review; never removes them automatically."""
    if not 0.8 <= threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be between 0.8 and 1.0")
    pairs: list[dict[str, Any]] = []
    by_scene: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_scene[record.scene].append(record)
    for scene, candidates in by_scene.items():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                score = _similarity(left.user, right.user)
                if score < threshold:
                    continue
                pairs.append({
                    "scene": scene,
                    "similarity": round(score, 4),
                    "left": {"source_line": left.source_line, "question": left.user},
                    "right": {"source_line": right.source_line, "question": right.user},
                })
                if len(pairs) >= max_pairs:
                    return sorted(pairs, key=lambda item: -item["similarity"])
    return sorted(pairs, key=lambda item: -item["similarity"])
