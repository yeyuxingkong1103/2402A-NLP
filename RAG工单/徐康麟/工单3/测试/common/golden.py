# -*- coding: utf-8 -*-
"""T8 的 14 题固定输入与期望输出（golden）加载与完整性校验。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

数据来源（``测试/测试数据/golden_qa_14.jsonl``）：
    * id 260/95/33/34/957/793/795/543/531/207 —— 沿用只读基准
      ``E:\\gao6gongdan\\工单1\\data\\eval\\golden_qa.jsonl``（answer/evidence/evidence_pages 原样）；
    * id 1~4 —— 力源信息（``招股说明书2.pdf``），题面按 ``设计/需求分析.md`` §4.2 语义还原，
      ``evidence`` / ``evidence_verbatim`` / ``citation_quote`` 由 tester 从**真实物理页**逐字提取
      （取证副本见 ``测试/测试数据/pdf2_evidence_pages.txt``，只读提取，未改 PDF）。

字段约定：
    ``evidence``          判定用证据（207 为**合成串 + 编辑注记**，14 题中唯一非逐字原文）；
    ``evidence_verbatim`` 逐字原文（**命中判定优先用它**，见 ``assertions.evidence_hit``）；
    ``citation_quote``    引用页回溯核验用的最小原文片段；
    ``evidence_pages``    1-based **物理页**（唯一页码口径）；
    ``corpus``            ``pdf1`` / ``pdf2``（自动发现解析，禁硬编码文件名）；
    ``pending_pdf2``      该题依赖 ``招股说明书2.pdf``；文件缺席时用例自动标 pending 而非失败。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import paths
from .assertions import Check, Report

#: 14 题 id 全集（顺序即展示顺序）
EXPECTED_IDS: tuple[int, ...] = (1, 2, 3, 4, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207)
#: 需要 PDF2 的题（PDF2 缺席时自动 pending，不判失败）
PDF2_IDS: tuple[int, ...] = (1, 2, 3, 4)
#: PDF1 题（id 260 … 207）
PDF1_IDS: tuple[int, ...] = tuple(i for i in EXPECTED_IDS if i not in PDF2_IDS)
#: 14 题中唯一非逐字 golden 的题（合成串 + 编辑注记）
SYNTHETIC_EVIDENCE_IDS: tuple[int, ...] = (207,)
#: 证据页（1-based 物理页）——来自 ``设计/验收标准.md`` §4 的 captain 复核结果
DESIGN_ANCHOR_PAGES: dict[int, tuple[int, ...]] = {
    1: (22, 24),
    2: (22, 30, 306, 144),
    3: (157, 158, 259, 21),
    4: (157, 158, 259),
    260: (129, 130, 131),
    95: (160, 27),
    33: (129, 130, 131),
    34: (152, 153),
    957: (154, 332),
    793: (152, 153),
    795: (27, 157),
    543: (22,),
    531: (22,),
    207: (479, 490, 30),
}


@dataclass
class GoldenItem:
    """一道题的固定输入 + 期望（含判定所必需的全部字段）。"""

    id: int
    question: str
    answer: str
    evidence: str
    evidence_verbatim: str
    evidence_pages: list[int]
    corpus: str
    category: str = ""
    should_be_unknown: bool = False
    citation_quote: str = ""
    required_substrings: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    pending_pdf2: bool = False
    min_hit_top_k: int = 5
    max_first_token_ms: float = 3000.0
    question_source: str = ""
    question_alt: str = ""      # 同一题的另一版题面（如设计文档里的措辞），仅供人工比对
    notes: str = ""

    # ---- 便捷属性 ----
    @property
    def primary_pages(self) -> list[int]:
        """设计锚点页（缺失时退回 ``evidence_pages``）。"""
        anchors = DESIGN_ANCHOR_PAGES.get(self.id)
        return list(anchors) if anchors else list(self.evidence_pages)

    @property
    def is_synthetic_evidence(self) -> bool:
        """``evidence`` 是否为非逐字原文（只有题 207）。"""
        return self.id in SYNTHETIC_EVIDENCE_IDS

    @property
    def file_hint(self) -> str:
        """语料解析提示：``pdf2`` → ``"2"``（由 ``paths.resolve_corpus`` 自动发现）。"""
        return "2" if self.corpus == "pdf2" else "1"

    def to_dict(self) -> dict[str, Any]:
        """转字典（写留痕用）。"""
        return asdict(self)


def load_golden(path: Path | str | None = None) -> list[GoldenItem]:
    """读取 golden JSONL。坏行抛 ``ValueError``（不静默丢弃）。"""
    target = Path(path) if path else paths.GOLDEN_FIXTURE
    if not target.exists():
        raise FileNotFoundError(f"golden 固定输入缺失：{target}")
    items: list[GoldenItem] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{lineno} JSON 解析失败：{exc}") from exc
        items.append(GoldenItem(
            id=int(raw["id"]),
            question=str(raw["question"]),
            answer=str(raw.get("answer", "")),
            evidence=str(raw.get("evidence", "")),
            evidence_verbatim=str(raw.get("evidence_verbatim", "")),
            evidence_pages=[int(p) for p in raw.get("evidence_pages", [])],
            corpus=str(raw.get("corpus", "")),
            category=str(raw.get("category", "")),
            should_be_unknown=bool(raw.get("should_be_unknown", False)),
            citation_quote=str(raw.get("citation_quote", "")),
            required_substrings=list(raw.get("required_substrings", []) or []),
            forbidden_substrings=list(raw.get("forbidden_substrings", []) or []),
            pending_pdf2=bool(raw.get("pending_pdf2", False)),
            min_hit_top_k=int(raw.get("min_hit_top_k", 5)),
            max_first_token_ms=float(raw.get("max_first_token_ms", 3000.0)),
            question_source=str(raw.get("question_source", "")),
            question_alt=str(raw.get("question_alt", "")),
            notes=str(raw.get("notes", "")),
        ))
    return items


def by_id(items: Sequence[GoldenItem]) -> dict[int, GoldenItem]:
    """按 id 建索引（重复 id 抛错）。"""
    out: dict[int, GoldenItem] = {}
    for item in items:
        if item.id in out:
            raise ValueError(f"golden 出现重复 id={item.id}")
        out[item.id] = item
    return out


def unlocked_items(items: Sequence[GoldenItem] | None = None) -> list[GoldenItem]:
    """可跑题集：PDF2 存在时 14 题全跑；PDF2 缺席时跳过依赖它的 4 题（标记 pending）。"""
    data = list(items) if items is not None else load_golden()
    have_pdf2 = paths.resolve_corpus("2") is not None
    if have_pdf2:
        return data
    return [item for item in data if item.id not in PDF2_IDS]


def pending_items(items: Sequence[GoldenItem] | None = None) -> list[GoldenItem]:
    """因语料缺席而暂不可跑的题（报告里必须显式列出，不得静默跳过）。"""
    data = list(items) if items is not None else load_golden()
    have_pdf2 = paths.resolve_corpus("2") is not None
    if have_pdf2:
        return []
    return [item for item in data if item.id in PDF2_IDS]


def corpus_path(item: GoldenItem) -> Path | None:
    """解析该题语料对应的真实 PDF（自动发现；找不到返回 ``None``）。"""
    return paths.resolve_corpus(item.file_hint)


def integrity_report(items: Sequence[GoldenItem] | None = None) -> Report:
    """golden 固定输入自身完整性（不依赖产品代码，任何时刻都能跑）。

    断言：14 题齐备且 id 与设计一致 / 页码均为 ≥1 整数且落在真实页数内 /
    每题至少有 ``evidence`` 或 ``evidence_verbatim`` / 语料可由自动发现解析 /
    题的语料归属与设计一致（PDF1 兴图新科、PDF2 力源信息）。
    """
    data = list(items) if items is not None else load_golden()
    report = Report("golden 14 题固定输入完整性")

    ids = [item.id for item in data]
    report.check("题数 = 14", len(data) == TOTAL_IDS(), f"实际 {len(data)}")
    report.check("id 集合与设计一致",
                 set(ids) == set(EXPECTED_IDS) and len(set(ids)) == len(ids),
                 f"实际 {sorted(ids)}")
    report.check("id 无重复", len(set(ids)) == len(ids))

    missing_evidence = [item.id for item in data if not (item.evidence or item.evidence_verbatim)]
    report.check("每题都有 evidence 或 evidence_verbatim", not missing_evidence, f"缺 {missing_evidence}")

    bad_pages = [item.id for item in data
                 if not item.evidence_pages or any(not isinstance(p, int) or p < 1 for p in item.evidence_pages)]
    report.check("evidence_pages 均为 ≥1 的整数（1-based 物理页）", not bad_pages, f"异常题 {bad_pages}")

    # 语料可解析 + 页数范围
    unresolved: list[int] = []
    out_of_range: list[int] = []
    for item in data:
        path = corpus_path(item)
        if path is None:
            unresolved.append(item.id)
            continue
        from . import pdf_probe  # noqa: PLC0415 —— 延迟导入

        total = pdf_probe.page_count(str(path))
        for page in item.evidence_pages:
            if page > total:
                out_of_range.append(item.id)
                break
    report.check("语料可由自动发现解析（禁硬编码文件名）", not unresolved,
                 f"未解析到 PDF 的题 {unresolved}")
    report.check("evidence_pages 均在真实页数范围内", not out_of_range, f"越界题 {out_of_range}")

    # 语料归属：PDF1 = 兴图新科，PDF2 = 力源信息（第 1 页发行人称谓）
    wrong_corpus: list[int] = []
    for item in data:
        path = corpus_path(item)
        if path is None:
            continue
        from . import pdf_probe  # noqa: PLC0415

        head = pdf_probe.page_text(str(path), 1)
        expect = "武汉力源信息技术股份有限公司" if item.corpus == "pdf2" else "武汉兴图新科电子股份有限公司"
        if expect not in head:
            wrong_corpus.append(item.id)
    report.check("每题语料归属与设计一致", not wrong_corpus, f"不符题 {wrong_corpus}")

    report.check("题 207 标注为合成证据（唯一非逐字 golden）",
                 all(by_id(data)[i].is_synthetic_evidence for i in SYNTHETIC_EVIDENCE_IDS),
                 f"合成证据题 {SYNTHETIC_EVIDENCE_IDS}")
    pdf2 = unlocked_items(data)
    report.check("PDF2 到位即可解锁 id 1~4（当前可跑题数）",
                 len(pdf2) == 14 or all(i.id not in PDF2_IDS for i in pdf2),
                 f"当前可跑 {len(pdf2)} 题；pending {[i.id for i in pending_items(data)]}")
    return report


def TOTAL_IDS() -> int:
    """题数常量（14）。"""
    return len(EXPECTED_IDS)


def write_snapshot(items: Iterable[GoldenItem], path: Path | str) -> Path:
    """把当前 golden（含逐字证据）快照写到指定路径（测试留痕用）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in items]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
