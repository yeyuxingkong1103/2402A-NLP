from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = ROOT / "sample_questions.md"

TOPIC_EXPANSIONS = {
    "盈利增长": ["业务结构", "零售业务", "资产质量", "拨备覆盖率"],
    "创新商业模式": ["金融+生活", "生活场景", "生态银行", "生态圈"],
    "宏观经济周期": ["风险管理", "拨备覆盖率", "资产质量", "逾期贷款", "贷款结构"],
    "经济周期波动": ["风险管理", "资本充足", "资产负债", "绿色金融", "科技金融", "数字化"],
}


def _alternatives(value: str) -> list[list[str]]:
    return [[item.strip() for item in group.split("|") if item.strip()] for group in value.split(";") if group.strip()]


def load_benchmarks(path: Path = BENCHMARK_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"参考问题文件不存在：{path}")
    records: list[dict] = []
    current: dict[str, str] = {}
    identifier = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^##\s+(\d+)\s*$", raw.strip())
        if heading:
            if current:
                records.append(_finish(identifier, current))
            identifier, current = int(heading.group(1)), {}
            continue
        field = re.match(r"^-\s+([A-Za-z]+):\s*(.*)$", raw.strip())
        if field:
            current[field.group(1).lower()] = field.group(2).strip()
    if current:
        records.append(_finish(identifier, current))
    if not records:
        raise ValueError("参考问题文件中没有有效问题")
    return records


def _finish(identifier: int | None, fields: dict[str, str]) -> dict:
    required = {"question", "reference", "facts"}
    missing = required - fields.keys()
    if identifier is None or missing:
        raise ValueError(f"参考问题缺少字段：{', '.join(sorted(missing))}")
    return {
        "id": identifier,
        "question": fields["question"],
        "retrieval": fields.get("retrieval", fields["question"]),
        "reference": fields["reference"],
        "stocks": [item.strip() for item in fields.get("stocks", "").split(",") if item.strip()],
        "facts": _alternatives(fields["facts"]),
    }


def rewrite_query(question: str, preferred: str = "") -> tuple[str, dict]:
    normalized = re.sub(r"\s+", " ", question or "").strip()
    entities: list[str] = []
    relations: list[str] = []
    if "xx银行" in normalized:
        normalized = normalized.replace("xx银行", "平安银行")
        entities.append("平安银行")
    if "平安银行" in normalized and "平安银行" not in entities:
        entities.append("平安银行")
    for topic, terms in TOPIC_EXPANSIONS.items():
        if topic in normalized:
            relations.extend(terms)
    rewritten = " ".join(dict.fromkeys(filter(None, [preferred.strip(), normalized, *relations])))
    return rewritten, {"original_query": question, "rewritten_query": rewritten, "entities": entities, "relations": list(dict.fromkeys(relations))}


def _plain(value: str) -> str:
    return re.sub(r"[\s,，。；;：:、（）()]+", "", value or "").lower()


def evaluate_hits(hits: list[dict], benchmark: dict) -> dict:
    facts = benchmark.get("facts") or []
    texts = [_plain(str(hit.get("text", ""))) for hit in hits]
    relevance = [any(_plain(term) in text for alternatives in facts for term in alternatives) for text in texts]
    relevant = sum(relevance)
    running = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(relevance, 1):
        if is_relevant:
            running += 1
            precision_sum += running / rank
    covered = sum(any(_plain(term) in text for text in texts for term in alternatives) for alternatives in facts)
    context_precision = precision_sum / relevant if relevant else 0.0
    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(covered / len(facts), 4) if facts else 0.0,
        "relevant_chunks": relevant,
        "retrieved_chunks": len(texts),
        "covered_facts": covered,
        "total_facts": len(facts),
        "precision_target_met": context_precision >= 0.8,
        "recall_target_met": bool(facts) and covered / len(facts) >= 0.9,
    }
