import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "case_id",
    "domain",
    "user_question",
    "fictional_facts",
    "facts_to_confirm",
    "risk_level",
    "expected_legal_basis",
    "expected_answer_points",
    "must_avoid",
    "scoring_rubric",
    "source_refs",
}

DOMAIN_COUNTS = {
    "离婚与婚姻关系": 40,
    "抚养与探望": 30,
    "夫妻财产与债务": 30,
}

RISK_LEVELS = {"normal", "medium", "high"}


def _is_non_empty_string(value: Any) -> bool:
    # 字符串字段必须去除空白后仍有内容，避免空记录通过校验。
    return isinstance(value, str) and bool(value.strip())


def _is_non_empty_list(value: Any) -> bool:
    # 列表字段至少包含一项，确保评测有明确依据和评分点。
    return isinstance(value, list) and len(value) > 0


def validate_record(record: dict[str, Any]) -> list[str]:
    # 单条记录校验只返回错误列表，不抛异常，便于批量报告全部问题。
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        errors.append(f"缺少字段：{', '.join(sorted(missing))}")
        return errors
    for field in ["case_id", "domain", "user_question", "fictional_facts", "risk_level"]:
        if not _is_non_empty_string(record.get(field)):
            errors.append(f"{field} 必须是非空字符串")
    if record.get("domain") not in DOMAIN_COUNTS:
        errors.append("domain 不在允许范围内")
    if record.get("risk_level") not in RISK_LEVELS:
        errors.append("risk_level 不在允许范围内")
    for field in ["facts_to_confirm", "expected_legal_basis", "expected_answer_points", "must_avoid", "source_refs"]:
        if not _is_non_empty_list(record.get(field)):
            errors.append(f"{field} 必须是非空列表")
    if not isinstance(record.get("scoring_rubric"), dict) or not record["scoring_rubric"]:
        errors.append("scoring_rubric 必须是非空对象")
    return errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSONL 逐行读取，定位错误时保留行号上下文。
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象")
        records.append(value)
    return records


def validate_dataset(records: list[dict[str, Any]]) -> list[str]:
    # 数据集级别校验负责数量、唯一 ID、领域分布和逐条 schema。
    errors: list[str] = []
    if len(records) != 100:
        errors.append(f"数据集必须正好 100 条，当前为 {len(records)} 条")
    case_ids = [record.get("case_id") for record in records]
    duplicates = [case_id for case_id, count in Counter(case_ids).items() if count > 1]
    if duplicates:
        errors.append(f"case_id 重复：{', '.join(str(item) for item in duplicates)}")
    domain_counts = Counter(record.get("domain") for record in records)
    for domain, expected_count in DOMAIN_COUNTS.items():
        actual_count = domain_counts.get(domain, 0)
        if actual_count != expected_count:
            errors.append(f"{domain} 应为 {expected_count} 条，当前为 {actual_count} 条")
    for index, record in enumerate(records, start=1):
        for error in validate_record(record):
            errors.append(f"第 {index} 条：{error}")
    return errors


def main(argv: list[str]) -> int:
    # 命令行入口只读取目标数据集，不触碰 public 原始资料目录。
    if len(argv) != 2:
        print("用法：python validate_dataset.py <dataset.jsonl>")
        return 2
    records = load_jsonl(Path(argv[1]))
    errors = validate_dataset(records)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("PASS: 100 records validated with counts 40/30/30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
