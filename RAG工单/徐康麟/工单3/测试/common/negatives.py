# -*- coding: utf-8 -*-
"""T8 负例集加载：不可答负例 + 同页互污染（N-5/N-4/N-6）用例。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import paths
from .assertions import Check, Report


@dataclass
class UnknownCase:
    """不可答负例：提问必须回「不清楚」，且不得出现被禁止的编造内容。"""

    id: str
    question: str
    file_names: list[str] = field(default_factory=list)
    reason_expected: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    category: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转字典。"""
        return asdict(self)


@dataclass
class LeakageCase:
    """同页互污染/要素齐备用例：``required`` 必含、``forbidden`` 必不含。"""

    id: str
    question: str
    corpus: str = ""
    required_substrings: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    expect_unknown: bool = False
    category: str = ""
    note: str = ""

    @property
    def file_hint(self) -> str:
        """语料解析提示（``pdf2`` → ``"2"``）。"""
        return "2" if self.corpus == "pdf2" else "1"

    def to_dict(self) -> dict[str, Any]:
        """转字典。"""
        return asdict(self)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 JSONL；文件缺失抛 ``FileNotFoundError``，坏行抛 ``ValueError``（不静默）。"""
    if not path.exists():
        raise FileNotFoundError(f"负例集缺失：{path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
    return rows


def load_unknown_cases(path: Path | str | None = None) -> list[UnknownCase]:
    """加载不可答负例（``测试/测试数据/unknown_questions.jsonl``）。"""
    target = Path(path) if path else paths.UNKNOWN_FIXTURE
    return [UnknownCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        file_names=[str(x) for x in raw.get("file_names", []) or []],
        reason_expected=[str(x) for x in raw.get("reason_expected", []) or []],
        must_not_contain=[str(x) for x in raw.get("must_not_contain", []) or []],
        category=str(raw.get("category", "")),
        note=str(raw.get("note", "")),
    ) for raw in _read_jsonl(target)]


def load_leakage_cases(path: Path | str | None = None) -> list[LeakageCase]:
    """加载互污染/要素齐备用例（``测试/测试数据/leakage_cases.jsonl``）。"""
    target = Path(path) if path else paths.LEAKAGE_FIXTURE
    return [LeakageCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        corpus=str(raw.get("corpus", "")),
        required_substrings=[str(x) for x in raw.get("required_substrings", []) or []],
        forbidden_substrings=[str(x) for x in raw.get("forbidden_substrings", []) or []],
        expect_unknown=bool(raw.get("expect_unknown", False)),
        category=str(raw.get("category", "")),
        note=str(raw.get("note", "")),
    ) for raw in _read_jsonl(target)]


def fixtures_integrity_report(unknown: Sequence[UnknownCase] | None = None,
                              leakage: Sequence[LeakageCase] | None = None) -> Report:
    """负例集完整性：题面非空、禁止项非空、必备项非空、互污染用例方向性正确。"""
    uk = list(unknown) if unknown is not None else load_unknown_cases()
    lk = list(leakage) if leakage is not None else load_leakage_cases()
    report = Report("负例集完整性（不可答 + 互污染）")

    report.check("不可答负例非空", bool(uk), f"{len(uk)} 条")
    report.check("互污染/要素用例非空", bool(lk), f"{len(lk)} 条")
    report.check("每条不可答负例都有题面",
                 all(case.question.strip() for case in uk),
                 f"空题面 {[c.id for c in uk if not c.question.strip()]}")
    report.check("每条不可答负例都给出「不得出现」的内容",
                 all(case.must_not_contain for case in uk),
                 f"缺 must_not_contain 的用例 {[c.id for c in uk if not c.must_not_contain]}")

    # N-4：PDF2 的「本次发行募集资金 + 补充流动资金」同页共现 = 空集 → 必须不可答
    n4 = [case for case in uk if "补充流动资金" in case.question]
    report.check("包含 N-4（PDF2 补流必拒答）用例", bool(n4), f"{[c.id for c in n4]}")
    report.check("N-4 用例禁止出现 15,000 相关编造",
                 all(any("15,000" in token or "15000" in token for token in case.must_not_contain)
                     for case in n4) if n4 else False,
                 "必须写明不得输出 15,000（PDF1 的金额）作为 PDF2 的答案")

    # N-5：题 4 必须含 7 家企业、禁自然人；题 3 反之
    t4 = [case for case in lk if case.id == "N-5-①"]
    t3 = [case for case in lk if case.id == "N-5-②"]
    report.check("N-5① 题 4 用例：禁「赵马克」且必含力源贸易/普芯达",
                 bool(t4) and "赵马克" in t4[0].forbidden_substrings
                 and {"力源贸易", "普芯达"} <= set(t4[0].required_substrings),
                 f"{t4[0].to_dict() if t4 else '缺失'}")
    seven = {"融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达"}
    report.check("N-5② 题 3 用例：禁 7 家企业且必含赵马克/42.35%",
                 bool(t3) and seven <= set(t3[0].forbidden_substrings)
                 and {"赵马克", "42.35"} <= set(t3[0].required_substrings),
                 f"{t3[0].to_dict() if t3 else '缺失'}")
    report.check("N-5③ 企业不得被误删（力源贸易/普芯达 在 required 侧）",
                 bool(t4) and {"力源贸易", "普芯达"} <= set(t4[0].required_substrings),
                 "captain 裁定：二者是企业，属题 4 答案")

    # N-6：题 34/793 不得被闸门判不可答（作为用例存在性断言）
    n6 = [case for case in lk if case.id == "N-6"]
    report.check("包含 N-6（题 34/793 闸门 fail-open）用例", bool(n6), f"{[c.id for c in n6]}")
    return report
