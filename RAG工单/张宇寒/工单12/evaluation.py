"""无需外部服务的 RAGAS 风格评测。"""

import re


def _normalize(text):
    return re.sub(r"[\s,，。；;：:（）()]", "", (text or "")).lower()


def _fact_hit(text, aliases):
    normalized = _normalize(text)
    return any(_normalize(alias) in normalized for alias in aliases)


def evaluate_result(question: dict, result: dict):
    contexts = result.get("contexts", [])
    texts = [c.get("text", "") for c in contexts]
    facts = question.get("expected_facts", [])
    relevant = sum(any(_fact_hit(text, aliases) for aliases in facts) for text in texts)
    recalled = sum(_fact_hit(" ".join(texts), aliases) for aliases in facts)
    answer = result.get("answer", "")
    supported = sum(_fact_hit(answer, aliases) and _fact_hit(" ".join(texts), aliases) for aliases in facts)
    answered = sum(_fact_hit(answer, aliases) for aliases in facts)
    q_chars = set(_normalize(question.get("question", "")))
    a_chars = set(_normalize(answer))
    relevancy = len(q_chars & a_chars) / max(1, len(q_chars))
    return {
        "context_precision": round(relevant / max(1, len(texts)), 3),
        "context_recall": round(recalled / max(1, len(facts)), 3),
        "faithfulness": round(supported / max(1, answered), 3) if answer else 0.0,
        "answer_relevancy": round(min(1.0, relevancy * 1.7), 3),
    }

