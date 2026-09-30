"""Deterministic cleaning and routing for customer-service JSONL datasets."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([!！?？。,.，、；;：:])")
_REPEAT_PUNCT_RE = re.compile(r"([!！?？。,.，、；;：:])\1+")
_DECORATION_RE = re.compile(r"^[\s\-_=~*#·•●○※★☆]+|[\s\-_=~*#·•●○※★☆]+$")
_QUESTION_KEY_RE = re.compile(r"[^\w一-鿿]+", re.UNICODE)
_CJK_PUNCTUATION = {
    "，": "",
    "。": "",
    "？": "",
    "！": "",
    "：": "",
    "；": "",
    "、": "",
}

SCENE_ALIASES = {
    "退货": "退换货",
    "换货": "退换货",
    "质量": "质量问题",
    "质量投诉": "质量问题",
    "质量反馈": "质量问题",
    "差评处理": "差评",
    "补偿": "补偿方案",
}
TONE_ALIASES = {
    "道歉": "致歉",
    "歉意": "致歉",
    "抱歉": "致歉",
    "抚慰": "安抚",
    "致歉安抚": "致歉+安抚",
}
AFTER_SALES_SCENES = {
    "退换货", "退款", "质量问题", "维权投诉", "差评", "补偿方案",
}
AFTER_SALES_TERMS = (
    "退货", "换货", "退款", "售后", "物流", "快递", "发货", "收货",
    "破损", "瑕疵", "质量", "差评", "投诉", "补偿", "运费", "保修",
)


@dataclass(slots=True)
class CleanRecord:
    """A normalized record ready for JSONL output."""

    scene: str
    user: str
    bot: str
    tone: str
    category: str
    question_group_id: str
    source_line: int
    normalized_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "scene": self.scene,
            "user": self.user,
            "bot": self.bot,
            "tone": self.tone,
            "category": self.category,
            "question_group_id": self.question_group_id,
            "source_line": self.source_line,
        }
        if self.normalized_fields:
            data["normalized_fields"] = list(self.normalized_fields)
        return {key: value for key, value in data.items() if value not in (None, "")}


@dataclass(slots=True, frozen=True)
class RejectedRecord:
    source_line: int
    reason: str
    raw: str

    def as_dict(self) -> dict[str, Any]:
        return {"source_line": self.source_line, "reason": self.reason, "raw": self.raw}


@dataclass(slots=True)
class CleaningReport:
    """Auditable counters produced by one cleaning run."""

    input_lines: int = 0
    valid_records: int = 0
    output_records: int = 0
    curated_records: int = 0
    invalid_records: int = 0
    exact_duplicates_removed: int = 0
    question_duplicate_records: int = 0
    question_duplicate_groups: int = 0
    question_variants_removed: int = 0
    normalized_fields: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    curated_category_counts: dict[str, int] = field(default_factory=dict)
    scene_counts: dict[str, int] = field(default_factory=dict)
    tone_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_lines": self.input_lines,
            "valid_records": self.valid_records,
            "output_records": self.output_records,
            "curated_records": self.curated_records,
            "invalid_records": self.invalid_records,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "question_duplicate_records": self.question_duplicate_records,
            "question_duplicate_groups": self.question_duplicate_groups,
            "question_variants_removed": self.question_variants_removed,
            "normalized_fields": self.normalized_fields,
            "category_counts": self.category_counts,
            "curated_category_counts": self.curated_category_counts,
            "scene_counts": self.scene_counts,
            "tone_counts": self.tone_counts,
        }


@dataclass(slots=True)
class CleaningResult:
    records: list[CleanRecord]
    rejected: list[RejectedRecord]
    report: CleaningReport


def _normalize_unicode(value: str) -> str:
    protected = value
    for punctuation, marker in _CJK_PUNCTUATION.items():
        protected = protected.replace(punctuation, marker)
    normalized = unicodedata.normalize("NFKC", protected)
    for punctuation, marker in _CJK_PUNCTUATION.items():
        normalized = normalized.replace(marker, punctuation)
    return normalized


def normalize_text(value: Any) -> str:
    """Normalize Unicode, whitespace, markup, and decorative punctuation safely."""
    if not isinstance(value, str):
        return ""
    text = _normalize_unicode(value)
    text = _CONTROL_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _DECORATION_RE.sub("", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def canonical_scene(value: Any) -> str:
    scene = normalize_text(value)
    labels = [item.strip() for item in re.split(r"[,，、]+", scene) if item.strip()]
    normalized = {SCENE_ALIASES.get(label, label) for label in labels}
    if normalized & AFTER_SALES_SCENES:
        priorities = ("退换货", "退款", "质量问题", "维权投诉", "差评", "补偿方案")
        return next(label for label in priorities if label in normalized)
    return SCENE_ALIASES.get(scene, scene)


def canonical_tone(value: Any) -> str:
    tone = normalize_text(value)
    return TONE_ALIASES.get(tone, tone)


def question_key(user: str) -> str:
    """Return a comparison key insensitive to spacing and punctuation variants."""
    return _QUESTION_KEY_RE.sub("", normalize_text(user).casefold())


def question_group_id(user: str) -> str:
    return hashlib.sha256(question_key(user).encode("utf-8")).hexdigest()[:16]


def classify_record(scene: str, user: str, bot: str) -> str:
    """Route a record to a product or after-sales output."""
    if scene in AFTER_SALES_SCENES:
        return "after_sales"
    haystack = f"{scene} {user} {bot}"
    return "after_sales" if any(term in haystack for term in AFTER_SALES_TERMS) else "product"


def _raw_text(raw: dict[str, Any], field_name: str) -> str:
    value = raw.get(field_name)
    return value if isinstance(value, str) else ""


def _validate(raw: Any, line_number: int) -> CleanRecord:
    if not isinstance(raw, dict):
        raise ValueError("record must be a JSON object")
    user = normalize_text(raw.get("user"))
    bot = normalize_text(raw.get("bot"))
    if not user or not bot:
        raise ValueError("user and bot must be non-empty strings")
    scene = canonical_scene(raw.get("scene"))
    tone = canonical_tone(raw.get("tone"))
    values = {"scene": scene, "user": user, "bot": bot, "tone": tone}
    changed = tuple(
        key for key, value in values.items() if _raw_text(raw, key) != value
    )
    return CleanRecord(
        scene=scene,
        user=user,
        bot=bot,
        tone=tone,
        category=classify_record(scene, user, bot),
        question_group_id=question_group_id(user),
        source_line=line_number,
        normalized_fields=changed,
    )


def _record_key(record: CleanRecord) -> str:
    fields = (
        question_key(record.scene),
        question_key(record.user),
        question_key(record.bot),
        question_key(record.tone),
    )
    return "\x1f".join(fields)


def clean_records(lines: Iterable[str]) -> CleaningResult:
    """Parse, normalize, validate, exact-deduplicate, and classify JSONL."""
    report = CleaningReport()
    records: list[CleanRecord] = []
    rejected: list[RejectedRecord] = []
    exact_seen: set[str] = set()
    question_groups: Counter[str] = Counter()
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        report.input_lines += 1
        try:
            record = _validate(json.loads(raw_line), line_number)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            report.invalid_records += 1
            rejected.append(RejectedRecord(line_number, str(exc), raw_line.rstrip("\r\n")))
            continue
        report.valid_records += 1
        report.normalized_fields += len(record.normalized_fields)
        exact_key = _record_key(record)
        if exact_key in exact_seen:
            report.exact_duplicates_removed += 1
            continue
        exact_seen.add(exact_key)
        records.append(record)
        question_groups[record.question_group_id] += 1
    duplicates = [count for count in question_groups.values() if count > 1]
    report.output_records = len(records)
    report.question_duplicate_groups = len(duplicates)
    report.question_duplicate_records = sum(count - 1 for count in duplicates)
    report.category_counts = dict(sorted(Counter(item.category for item in records).items()))
    report.scene_counts = dict(sorted(Counter(item.scene for item in records).items()))
    report.tone_counts = dict(sorted(Counter(item.tone for item in records).items()))
    return CleaningResult(records, rejected, report)


def clean_jsonl(source: str | Path) -> CleaningResult:
    path = Path(source)
    return clean_records(path.read_text(encoding="utf-8-sig").splitlines())


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_jsonl(records: Iterable[Any], target: str | Path) -> None:
    lines = []
    for record in records:
        payload = record.as_dict() if hasattr(record, "as_dict") else record
        lines.append(json.dumps(payload, ensure_ascii=False))
    _atomic_write(Path(target), "\n".join(lines) + ("\n" if lines else ""))


def write_json(payload: dict[str, Any], target: str | Path) -> None:
    _atomic_write(Path(target), json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
