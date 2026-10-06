# -*- coding: utf-8 -*-
"""T8 判定口径（**唯一实现**，三级测试共用）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么必须唯一：同一口径若在各测试文件里各写一份，必然漂移，最终把**正确实现判成失败**
（captain 已多轮裁定，见 ``设计/验收标准.md`` §2/§3）。因此本模块是唯一实现，
测试文件只允许调用，不允许另写判据。

固化的口径（逐条对应来源）：
    1. **命中**（§2.2）：``hit = text_utils.evidence_contains(chunk.content, evidence, 0.90)``
       对 top-k 任一 chunk 成立；**禁止**用「引用页 == evidence_pages」判命中
       （会把 957 误计为命中）。题 207 的 golden ``evidence`` 是**合成串 + 编辑注记**
       （14 题中唯一非逐字原文，任何 chunk 不可能包含）→ 该题优先用 ``evidence_verbatim``。
    2. **引用可回溯**（§2.4 + ``需求分析`` §4.4）：页码必须是 **1-based 物理页**且该页
       真实含支撑原文；引用页 ≠ ``evidence_pages`` 是**允许**的；页脚印刷页码（``1-1-128``）
       与 0-based 索引（``128``）**禁止**用于引用（二者恒等于物理页 − 1，用「错页」判定陷阱抓）。
    3. **首字**（§2.3）：``first_token_ms`` **逐题**判定，取最大值；冷启动**不豁免**
       （预热是硬要求，零调用点即缺陷，见 §3.4）。
    4. **表格缺陷**（§3.1）：缺陷**只认「行内横向重复」**；跨行纵向同值（PDF1 物理 65/66
       真实重复「陈爱民/程家明」）**不是**缺陷，禁止设纵向门槛。
    5. **非缺陷白名单**（§3.2/§3.3）：退化表不进索引、引用页 ≠ evidence_pages、
       闸门 ``counted=False``（fail-open）、``[ ◆ ]`` = 「未披露」，一律不得判缺陷。
    6. **RAGAS**：本机断网不可用，报告必须显式标注「RAGAS 未运行（依赖不可用，本机断网）」，
       严禁伪造数值；本模块提供常量与强制注入函数。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 常量（全部来自设计文档，禁止在测试里另写魔数）
# ---------------------------------------------------------------------------
WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
RAGAS_BANNER = "RAGAS 未运行（依赖不可用，本机断网）"

FIRST_TOKEN_BUDGET_MS = 3000.0          # 验收 4：每题首字 ≤ 3 s（取最大值判定）
TOTAL_QUESTIONS = 14                    # 验收 3：14 题全集
ACCURACY_THRESHOLD = 0.90               # 验收 3：≥ 90%
ACCURACY_MIN_CORRECT = 13               # 14 题 ≥ 90% ⇔ ≥ 13 题
EVIDENCE_THRESHOLD = 0.90               # §2.2 命中阈值
FUZZY_THRESHOLD = 0.62                  # §2.1 判分口径（工单1 Evaluator）
TOP_K = 5                               # §8 默认 ``RAG_RETRIEVAL__TOP_K``

#: 印刷页码/索引形态（禁止用于引用；见 ``需求分析`` §4.4）
PRINTED_PAGE_PATTERNS = (
    re.compile(r"^\s*\d{1,3}-\d{1,3}-(\d{1,4})\s*$"),   # PDF1 页脚 1-1-128
    re.compile(r"^\s*(\d{1,4})\s*$"),                     # PDF2 页眉第 2 行 128
)

#: 答案里的引用标记：``[招股说明书1.pdf: 129]`` / ``[招股说明书1.pdf:129]`` / ``【…：129】``
CITATION_RE = re.compile(r"[\[【]\s*([^\[\]【】:：]{2,120}?)\s*[:：]\s*(\d{1,4})\s*[\]】]")

# ---------------------------------------------------------------------------
# 0) 题面与分类契约（**防漂移第二道闸**，captain 裁定 1）
# ---------------------------------------------------------------------------
#: 14 题期望主体类型（N-7）：剥离发行人全称后 any 10 / person 2（题 3、531）/ organization 2（题 4、34）
EXPECTED_SUBJECT_TYPE: dict[int, str] = {
    1: "any", 2: "any", 3: "person", 4: "organization",
    260: "any", 95: "any", 33: "any", 34: "organization",
    957: "any", 793: "any", 795: "any", 543: "any", 531: "person", 207: "any",
}
#: 数值信号（N-8）：True / False 两组，分类一律取自**本轮原文**
NUMERIC_TRUE_IDS: tuple[int, ...] = (1, 3, 33, 207, 260, 543)
NUMERIC_FALSE_IDS: tuple[int, ...] = (2, 4, 34, 95, 531, 793, 795, 957)
#: 表格兜底（宽松集合，仅用于追加表块检索）
PREFER_TABLE_IDS: tuple[int, ...] = (1, 2, 3, 4, 33, 207, 260, 543)
#: 剥离后必须保留的数值信号词（防「分类被改写破坏」）
KEEP_SIGNAL_TOKENS: dict[int, tuple[str, ...]] = {
    207: ("的多少",), 260: ("分别",), 33: ("分别",), 1: ("是多少",), 543: ("是多少",),
}
#: 同页互污染题（PDF2 物理 157 两张表）：题 3 = 自然人、题 4 = 企业
SEVEN_COMPANIES: tuple[str, ...] = ("融冰投资", "武汉博润", "上海博润", "听音投资",
                                    "联众聚源", "力源贸易", "普芯达")

#: 题面契约（**题面 + 预期分类 + 预期 allowed 集合**；captain 裁定：题面唯一来源 = T5 版）
QUESTION_CONTRACT: dict[int, dict[str, Any]] = {
    3: {
        "question": "武汉力源信息技术股份有限公司存在控制关系的关联方是谁？其持股比例与本公司关系是什么？",
        "expected_subject": "person",
        "corpus": "pdf2",
        "evidence_pages": (157, 158, 259, 21),
        "required_allowed": ("赵马克",),          # allowed 必须含自然人
        "forbidden_allowed": SEVEN_COMPANIES,      # 自然人题不得把企业算作 allowed
        "answer_must_contain": ("赵马克", "42.35"),
        "answer_must_not_contain": SEVEN_COMPANIES,
    },
    4: {
        "question": "武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
        "expected_subject": "organization",
        "corpus": "pdf2",
        "evidence_pages": (157, 158, 259),
        "required_allowed": SEVEN_COMPANIES,       # allowed 必须覆盖 7 家企业
        "forbidden_allowed": ("赵马克",),           # 企业题不得把自然人算作 allowed
        "answer_must_contain": SEVEN_COMPANIES,
        "answer_must_not_contain": ("赵马克",),
    },
}
#: 误拒答断言的**首个案例**（captain 实测：题 4 曾被闸门误判 org_leaked 而拒答）
FALSE_REFUSAL_FIRST_CASE = 4
#: 主体闸门拒绝原因（14 题里出现任一即误拒答）
GATE_REFUSAL_REASONS = ("subject_gate:org_leaked", "subject_gate:person_leaked",
                        "subject_gate:triple_incomplete")


#: **枚举型问题**（captain 裁定 2026-10-04：判分器第①步偏松是工单1 尺子的既有特性、**不修**；
#: 因此「答案必须给全枚举」由产品侧自检兜底）——这些题的 ``required_substrings`` 必须**全部命中**：
#: 题 33 必须给全 4 个比重（82.10/97.31/94.84/94.34），不能只给 1 个而靠尺子漏放。
ENUMERATED_QUESTIONS: tuple[int, ...] = (1, 2, 3, 4, 33, 260, 543)


def expected_subject_type(question_id: int) -> str:
    """该题期望的主体类型（``person`` / ``organization`` / ``any``）。"""
    return EXPECTED_SUBJECT_TYPE.get(int(question_id), "any")


def expected_allowed_subset(question_id: int) -> tuple[str, ...]:
    """该题 allowed 集合**必须覆盖**的取值（题 3 = 赵马克；题 4 = 7 家企业）。"""
    contract = QUESTION_CONTRACT.get(int(question_id))
    return tuple(contract["required_allowed"]) if contract else ()


def check_question_text(question_id: int, question: str) -> Check:
    """题面不得漂移（captain 裁定：T5 版为唯一题面）。"""
    contract = QUESTION_CONTRACT.get(int(question_id))
    if not contract:
        return Check(f"题面契约（题 {question_id}）", True, "该题未纳入题面契约（仅题 3/题 4 需要）")
    return Check(f"题面与契约一致（题 {question_id}）",
                 str(question or "").strip() == contract["question"],
                 f"期望 {contract['question']!r}；实测 {str(question or '').strip()!r}")


def check_gate_expectation(gate: Any, question_id: int) -> list[Check]:
    """题 3/题 4 的闸门期望：不得拒答、expected 正确、counted=True、allowed 覆盖必备取值。

    ★ 对这两题 ``counted=False``（fail-open）**不算通过**：二者证据里都有主体表
    （``设计/验收标准.md`` §2.1 N-6 的反向断言）。
    """
    contract = QUESTION_CONTRACT.get(int(question_id))
    if not contract:
        return [Check(f"闸门契约（题 {question_id}）", True, "该题不在题 3/题 4 契约内")]
    ok = bool(_attr(gate, "ok", False))
    expected = str(_attr(gate, "expected", "") or "")
    counted = bool(_attr(gate, "counted", False))
    allowed = tuple(str(v) for v in (_attr(gate, "allowed", ()) or ()))
    leaked = tuple(str(v) for v in (_attr(gate, "leaked", ()) or ()))
    found = tuple(str(v) for v in (_attr(gate, "found", ()) or ()))
    reason = str(_attr(gate, "reason", "") or "")

    return [
        Check(f"闸门不得拒答（题 {question_id}）", ok and not reason.endswith("leaked"),
              f"ok={ok} reason={reason} leaked={list(leaked[:5])}"),
        Check(f"期望主体类型 = {contract['expected_subject']}（题 {question_id}）",
              expected == contract["expected_subject"], f"实测 {expected!r}"),
        Check(f"闸门真正生效（counted=True，题 {question_id}）", counted,
              f"counted={counted} reason={reason}（counted=False 不算通过，见 N-6 反向断言）"),
        Check(f"allowed 覆盖必备取值（题 {question_id}）",
              all(value in allowed for value in contract["required_allowed"]),
              f"缺 {[v for v in contract['required_allowed'] if v not in allowed]}；allowed={list(allowed[:10])}"),
        Check(f"allowed 不得含对侧主体（题 {question_id}）",
              not any(value in allowed for value in contract["forbidden_allowed"]),
              f"越界 {[v for v in contract['forbidden_allowed'] if v in allowed]}"),
        Check(f"found 非空（题 {question_id}）", bool(found), f"found={list(found[:6])}"),
    ]


def check_no_false_refusal(rows: Sequence[Mapping[str, Any]],
                           *, first_case: int = FALSE_REFUSAL_FIRST_CASE) -> list[Check]:
    """**误拒答 = 0**：14 题的 ``unknown_ids`` 必须为空（任何一题回「不清楚」即失败）。

    ``rows`` 每项至少要有 ``id`` 与 ``is_unknown``（可选 ``unknown_reason``）。
    首个案例固定为题 4（captain 实测它曾被闸门误判 ``subject_gate:org_leaked`` 而拒答）。
    """
    refused = [row for row in rows if bool(row.get("is_unknown"))]
    unknown_ids = [int(row.get("id")) for row in refused]
    first_row = next((row for row in rows if int(row.get("id")) == int(first_case)), None)
    bad_reasons = [row.get("unknown_reason") for row in refused
                   if str(row.get("unknown_reason") or "") in GATE_REFUSAL_REASONS]
    return [
        Check("14 题 unknown_ids 必须为空（误拒答 = 0）", not unknown_ids,
              f"被拒答题 {unknown_ids}；原因 {[row.get('unknown_reason') for row in refused]}"),
        Check(f"首个案例（题 {first_case}）不得被拒答",
              first_row is not None and not bool(first_row.get("is_unknown")),
              f"题 {first_case}：is_unknown={None if first_row is None else first_row.get('is_unknown')}　"
              f"reason={None if first_row is None else first_row.get('unknown_reason')}"),
        Check("不存在主体闸门误拒答（subject_gate:*）", not bad_reasons, f"{bad_reasons}"),
    ]


#: 非缺陷白名单（§3.2/§3.3）——出现这些现象时**禁止**判缺陷
NON_DEFECT_WHITELIST: tuple[dict[str, str], ...] = (
    {"item": "跨行纵向同值", "rule": "同一列多行出现相同值不算缺陷；缺陷只认行内横向重复",
     "source": "验收标准.md §3.1（PDF1 物理 65/66 陈爱民/程家明 真实重复）"},
    {"item": "退化表不进索引", "rule": "degenerate=1 的块不进 chunk/索引是设计选择（证据页上退化块 = 0）",
     "source": "验收标准.md §3.2"},
    {"item": "引用页 ≠ evidence_pages", "rule": "同一原文可多处出现；只要引用页含支撑原文即有效",
     "source": "验收标准.md §3.3"},
    {"item": "闸门 counted=False", "rule": "题 34/793 等正文证据题的 fail-open 正常结果，不是缺陷",
     "source": "验收标准.md §2.1 N-6 / §3.3"},
    {"item": "表格占位符 [ ◆ ]", "rule": "原文即占位符，按「未披露」表述；不是 None、不是 0",
     "source": "验收标准.md §3.3"},
    {"item": "表格计数口径差异", "rule": "continued_tables/repeated_headers_removed 只作留痕，不作门槛",
     "source": "验收标准.md §3.3"},
    {"item": "首次请求冷启动", "rule": "由启动预热吸收；**预热调用点缺失**才判缺陷",
     "source": "验收标准.md §3.4"},
)


# ---------------------------------------------------------------------------
# 断言结果载体（所有测试统一用它落盘留痕）
# ---------------------------------------------------------------------------
@dataclass
class Check:
    """一条断言结果。"""

    name: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        """``✅/❌ 名称　细节`` 形式，便于 stdout 留痕直接阅读。"""
        mark = "✅" if self.ok else "❌"
        return f"{mark} {self.name}" + (f"　{self.detail}" if self.detail else "")

    def to_dict(self) -> dict[str, Any]:
        """转字典（写 JSON 留痕用）。"""
        return asdict(self)


@dataclass
class Report:
    """一次测试运行/一个用例集的断言汇总。"""

    title: str
    checks: list[Check] = field(default_factory=list)

    # ---- 记录 ----
    def check(self, name: str, ok: Any, detail: str = "") -> bool:
        """记录一条断言，返回 ``bool(ok)`` 方便 ``assert report.check(...)``。"""
        flag = bool(ok)
        self.checks.append(Check(name=name, ok=flag, detail=detail))
        return flag

    def add(self, item: Check) -> Check:
        """直接追加一个 ``Check``。"""
        self.checks.append(item)
        return item

    def extend(self, items: Iterable[Check]) -> None:
        """批量追加。"""
        self.checks.extend(items)

    # ---- 汇总 ----
    @property
    def passed(self) -> int:
        """通过条数。"""
        return sum(1 for c in self.checks if c.ok)

    @property
    def failed(self) -> int:
        """失败条数。"""
        return sum(1 for c in self.checks if not c.ok)

    @property
    def ok(self) -> bool:
        """全部通过。"""
        return self.failed == 0 and bool(self.checks)

    def failures(self) -> list[Check]:
        """失败明细（报告必须逐条打印，禁止只给总数）。"""
        return [c for c in self.checks if not c.ok]

    def summary(self) -> str:
        """多行汇总文本（含失败明细 + RAGAS 标注）。"""
        lines = [f"【{self.title}】✅{self.passed} / {len(self.checks)}　❌{self.failed}"]
        for item in self.failures():
            lines.append("  " + item.render())
        lines.append("  注：" + RAGAS_BANNER)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """结构化结果（含工单编号与 RAGAS 标注，供留痕 JSON 使用）。"""
        return {
            "work_order": WORK_ORDER,
            "title": self.title,
            "total": len(self.checks),
            "passed": self.passed,
            "failed": self.failed,
            "ok": self.ok,
            "ragas": RAGAS_BANNER,
            "checks": [c.to_dict() for c in self.checks],
        }

    def write(self, path: Path | str) -> Path:
        """把结果写 JSON 文件（父目录自动创建），返回写入路径。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_dict()
        rows["checks"] = [c.to_dict() for c in self.checks]
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def print(self) -> None:
        """打印逐条断言（不依赖 pytest 的捕获）。"""
        print(f"\n【{self.title}】")
        for item in self.checks:
            print("  " + item.render())
        print(f"  合计：✅{self.passed} / {len(self.checks)}　❌{self.failed}")
        print("  注：" + RAGAS_BANNER)


# ---------------------------------------------------------------------------
# 1) 命中判定（唯一入口：委托 retrieval_utils.is_evidence_hit → evidence_contains）
# ---------------------------------------------------------------------------
@dataclass
class HitOutcome:
    """命中判定结果（含实际使用的判据，便于报告自证没用「引用页相等」）。"""

    hit: bool
    criterion: str
    used_field: str
    matched_chunk_ids: list[str]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """转字典。"""
        return asdict(self)


class _ChunkLike:
    """把纯文本包装成「带 ``content`` 的对象」，供 ``is_evidence_hit`` 复用同一实现。"""

    __slots__ = ("content", "chunk_id")

    def __init__(self, content: str, chunk_id: str = "") -> None:
        self.content = content
        self.chunk_id = chunk_id


def _content_of(chunk: Any) -> str:
    """取块的正文（兼容纯字符串 / dataclass / JSONL dict 三种形态）。"""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        return str(chunk.get("content", "") or "")
    return str(getattr(chunk, "content", "") or "")


def _chunk_id_of(chunk: Any, index: int) -> str:
    """取块 id（dict 与 dataclass 都支持；缺失时用 ``#<序号>`` 占位便于定位）。"""
    if isinstance(chunk, Mapping):
        return str(chunk.get("chunk_id", "") or f"#{index + 1}")
    return str(getattr(chunk, "chunk_id", "") or f"#{index + 1}")


def chunk_texts(chunks: Sequence[Any]) -> list[str]:
    """取出每个 chunk 的 ``content``（兼容 ``RetrievedChunk`` / ``Chunk`` / dict / 纯字符串）。"""
    return [_content_of(chunk) for chunk in chunks]


def evidence_hit(
    chunks: Sequence[Any],
    *,
    evidence: str | None = None,
    verbatim: str | None = None,
    prefer_verbatim: bool = True,
    threshold: float = EVIDENCE_THRESHOLD,
) -> HitOutcome:
    """**命中判定唯一实现**：证据原文是否落在返回块中。

    参数：
        chunks：``RetrievedChunk`` / ``Chunk`` / 纯文本序列（``top-k`` 结果）；
        evidence：golden 的 ``evidence``；
        verbatim：golden 的 ``evidence_verbatim``（逐字原文）；
        prefer_verbatim：为 ``True``（默认）且 ``verbatim`` 非空时，用 ``verbatim`` 判命中
            —— 题 207 的 golden ``evidence`` 是合成串+编辑注记，用它判命中必然漏判（基准缺陷）。

    实现：委托 ``app.core.retrieval_utils.is_evidence_hit``（它会调
    ``text_utils.evidence_contains``），**不使用**任何「引用页 == evidence_pages」判据。
    """
    from . import paths as _paths  # noqa: PLC0415 —— 避免循环导入

    _paths.ensure_dev_on_path()
    from app.core import retrieval_utils  # noqa: PLC0415

    text = str(verbatim or "").strip() if prefer_verbatim else ""
    used_field = "evidence_verbatim"
    if not text:
        text = str(evidence or "").strip()
        used_field = "evidence"
    if not text:
        return HitOutcome(False, "evidence_contains", "none", [],
                          "证据与逐字原文均为空 → 不判命中（避免空证据假命中）")

    probes: list[Any] = []
    matched: list[str] = []
    for index, chunk in enumerate(chunks):
        content = _content_of(chunk)
        chunk_id = _chunk_id_of(chunk, index)
        wrapper = _ChunkLike(content, chunk_id)
        probes.append(wrapper)
        if retrieval_utils.is_evidence_hit([wrapper], text, threshold=threshold):
            matched.append(wrapper.chunk_id)

    detail = (f"判据=evidence_contains(threshold={threshold})　字段={used_field}　"
              f"匹配块={matched or '无'}　证据片段={text[:48]!r}")
    return HitOutcome(bool(matched), "evidence_contains", used_field, matched, detail)


def hit_is_evidence_based(outcome: HitOutcome) -> bool:
    """自证断言：命中必须来自 ``evidence_contains``（而不是引用页相等这类退化判据）。"""
    return outcome.criterion == "evidence_contains"


# ---------------------------------------------------------------------------
# 2) 引用核验（1-based 物理页 + 真实支撑原文 + 反 0-based 陷阱）
# ---------------------------------------------------------------------------
@dataclass
class ParsedCitation:
    """从答案文本里解析出的引用。"""

    file_name: str
    page: int
    raw: str


def parse_citations(text: str) -> list[ParsedCitation]:
    """解析答案文本中的 ``[文件名: 页码]``（中英文冒号与全角括号都识别）。"""
    out: list[ParsedCitation] = []
    for match in CITATION_RE.finditer(str(text or "")):
        name = match.group(1).strip()
        try:
            page = int(match.group(2))
        except ValueError:                                        # pragma: no cover - 正则已限数字
            continue
        out.append(ParsedCitation(file_name=name, page=page, raw=match.group(0)))
    return out


def looks_like_printed_page_form(token: str) -> bool:
    """判断字符串是否为「页脚印刷页码 / 0-based 索引」形态（``1-1-128`` 或裸数字）。"""
    return any(pattern.match(str(token or "")) for pattern in PRINTED_PAGE_PATTERNS)


def squash(text: str) -> str:
    """本地归一化（去空白与常见中英标点、全角转半角、小写）——只用于**支撑原文**比对。

    命中判定**不用**它（必须走产品 ``evidence_contains``）；这里用于引用页核验，
    使「空格/换行/标点差异」不会误判引用无效。
    """
    body = str(text or "")
    # 先做全角→半角（仅常见标点与空格），再去掉空白与标点
    table = str.maketrans({"（": "(", "）": ")", "：": ":", "％": "%", "　": " ", "\u00a0": " "})
    body = body.translate(table)
    return re.sub(r"""[\s,，。；;、|*#>·\-—–_/\\"'“”‘’()\[\]【】{}:]+""", "", body).lower()


def support_present(page_content: str, support: str) -> bool:
    """支撑原文是否落在该页文本里（三级判定：精确包含 → 归一化包含 → **有序子序列**）。

    第三级为什么必要：表格证据常常是「一列多行的值序列」（如题 4 的 7 家企业名），
    单元格之间夹着「与本公司关系」的说明文字，归一化后并不连续；
    只要这些证据片段**按原顺序**都能在页文本里找到，就说明该页确实承载该证据。
    """
    needle = str(support or "").strip()
    if not needle:
        return False
    if needle in str(page_content or ""):
        return True
    flat_needle, flat_page = squash(needle), squash(page_content)
    if flat_needle and flat_needle in flat_page:
        return True
    return ordered_terms_present(flat_page, flat_needle)


def ordered_terms_present(flat_page: str, flat_needle: str, *, min_term: int = 2) -> bool:
    """证据片段是否**按顺序**出现在页文本中（允许片段之间夹着其它文字）。

    切分规则：把归一化后的证据串按长度切块（默认 4 字），逐块在页文本里顺序前移查找；
    全部找到即认为该页承载该证据。块过短（< ``min_term``）的碎片直接忽略，
    避免用单字造成假命中。
    """
    if not flat_needle or len(flat_needle) < min_term:
        return False
    step = max(min_term, 4)
    cursor = 0
    chunks = [flat_needle[i:i + step] for i in range(0, len(flat_needle), step)]
    chunks = [piece for piece in chunks if len(piece) >= min_term]
    if not chunks:
        return False
    for piece in chunks:
        found = flat_page.find(piece, cursor)
        if found < 0:
            return False
        cursor = found + len(piece)
    return True


def printed_page_forms(physical_page: int) -> list[str]:
    """给定物理页，返回**禁止用于引用**的数字形态（0-based 索引与页脚印刷页码）。

    实现委托 ``pdf_probe.printed_page_forms``（同一口径），便于测试文件统一从本模块调用。
    """
    from . import pdf_probe  # noqa: PLC0415 —— 延迟导入避免循环依赖

    return pdf_probe.printed_page_forms(physical_page)


#: 产品在表块正文里注入的标注行（表格 Markdown 化时加的可读标签）——核验页码支撑时必须剥掉，
#: 否则 `[表标题] …`/`[来源] … 第 N 页` 这类**产品自造文本**会被误判成「该页不含支撑原文」
PRODUCT_MARKER_RE = re.compile(r"^\s*\[(?:表标题|表注|来源|页码|关键数字|表格|section|title)\]")


def strip_product_markers(text: str) -> str:
    """剥掉产品注入的标注行与页眉/页脚行（``1-1-128``、发行人+招股意向书、裸数字）。"""
    out: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or PRODUCT_MARKER_RE.match(stripped):
            continue
        if re.fullmatch(r"\d{1,3}-\d{1,3}-\d{1,4}", stripped):        # PDF1 页脚
            continue
        if re.fullmatch(r"\d{1,4}", stripped):                        # PDF2 页眉裸数字
            continue
        if ("招股意向书" in stripped and len(stripped) <= 40):        # 页眉标题行
            continue
        out.append(stripped)
    return "\n".join(out)


def page_support_ratio(chunk_content: str, page_text: str, *, gram: int = 4) -> float:
    """块正文有多少比例（按 ``gram`` 字切片）能在该页文本里找到 —— 引用页可回溯的量化判据。

    为什么不用「引用自带的 quote 是否逐字出现在页上」：产品给表块的 ``quote`` 是
    ``[表标题] … [来源] … 第 22 页`` 这类**拼装标签**，天然不会出现在 PDF 原文里；
    真正的判据是「**这个块的内容就在它声称的那一页上**」。
    """
    cleaned = squash(strip_product_markers(chunk_content))
    flat_page = squash(page_text)
    if not cleaned or not flat_page:
        return 0.0
    if len(cleaned) <= gram:
        return 1.0 if cleaned in flat_page else 0.0
    ngrams = [cleaned[i:i + gram] for i in range(len(cleaned) - gram + 1)]
    hit = sum(1 for piece in ngrams if piece in flat_page)
    return hit / len(ngrams)


def check_citation_traceable(
    citation: Any,
    *,
    page_lookup: Callable[[str, int], str],
    page_counts: Mapping[str, int],
    discovered_files: Sequence[str],
    support: str | None = None,
    chunk_lookup: Callable[[str], Any] | None = None,
    min_page_support: float = 0.5,
) -> Check:
    """单条引用四点核验（§2.4）：文件名真实 / 页码 1-based 在范围 / **chunk 回查一致** / 引用页承载该块。

    ``citation`` 支持 ``Citation`` 对象或 dict（``file_name`` / ``page`` / ``chunk_id`` / ``quote``）。

    核验顺序：
        ① 文件名 ∈ 自动发现结果；② ``page`` 为整数且 1 ≤ page ≤ 页数（越界或落入 0-based 区间会给出提示）；
        ③ 给了 ``chunk_lookup`` 时回查 ``chunk_id``：块必须存在、其 ``page`` 与引用页一致；
        ④ 引用页必须**承载该块内容**（``page_support_ratio`` ≥ ``min_page_support``，
           标注行与页眉页脚先剥掉）；块跨页时允许在块覆盖的其它物理页上找到支撑，但会注明页码；
        ⑤ 反 0-based 陷阱：引用页没有、下一页有 → 判「索引当页码」。
    """
    if isinstance(citation, Mapping):
        file_name = str(citation.get("file_name", ""))
        page = citation.get("page", 0)
        quote = str(citation.get("quote", "") or "")
        chunk_id = str(citation.get("chunk_id", "") or "")
    else:
        file_name = str(getattr(citation, "file_name", ""))
        page = getattr(citation, "page", 0)
        quote = str(getattr(citation, "quote", "") or "")
        chunk_id = str(getattr(citation, "chunk_id", "") or "")
    label = f"{file_name}:{page}"

    if file_name not in set(discovered_files):
        return Check(f"引用文件名真实（{label}）", False,
                     f"文件名不在自动发现结果里：{sorted(discovered_files)}")
    if not isinstance(page, int) or isinstance(page, bool):
        return Check(f"引用页码为整数（{label}）", False, f"page 类型 {type(page).__name__}：{page!r}")
    total = int(page_counts.get(file_name, 0))
    if page < 1 or page > total:
        hint = ""
        if 0 <= page <= total:
            hint = "　⚠️ 该值落在 0-based 索引区间（0..页数-1）→ 疑似用 0-based 索引当页码"
        return Check(f"引用页码在 1..{total} 且为 1-based（{label}）", False, f"越界{hint}")

    try:
        text = page_lookup(file_name, page)
    except (KeyError, ValueError) as exc:
        return Check(f"引用页可读取（{label}）", False, f"{type(exc).__name__}: {exc}")

    # ③ chunk 回查（给了就必查：chunk_id 可查 + 页码一致）
    if chunk_lookup is not None and chunk_id:
        chunk = (chunk_lookup.get(chunk_id) if isinstance(chunk_lookup, Mapping)
                 else chunk_lookup(chunk_id))
        if chunk is None:
            return Check(f"引用 chunk 可回查（{label}）", False, f"chunk_id={chunk_id!r} 在 chunks.jsonl 里不存在")
        chunk_page = int((chunk.get("page") if isinstance(chunk, Mapping)
                          else getattr(chunk, "page", 0)) or 0)
        if chunk_page != page:
            return Check(f"引用 chunk 页码一致（{label}）", False,
                         f"chunk {chunk_id} 的 page={chunk_page} 与引用页 {page} 不一致")
        content = str(chunk.get("content") if isinstance(chunk, Mapping)
                      else getattr(chunk, "content", "") or "")
        if not content.strip():
            return Check(f"引用 chunk 正文非空（{label}）", False, f"chunk_id={chunk_id}")
        ratio = page_support_ratio(content, text)
        if ratio >= min_page_support:
            return Check(f"引用可回溯（{label}）", True,
                         f"chunk {chunk_id} 的 {ratio:.0%} 内容落在该页（阈值 {min_page_support:.0%}）")
        # 跨页块：允许在块覆盖的其它页上找到支撑，但必须注明
        span: list[int] = []
        for key in ("page_start", "page_end"):
            value = chunk.get(key) if isinstance(chunk, Mapping) else getattr(chunk, key, None)
            if isinstance(value, int):
                span.append(value)
        best = (ratio, page)
        for other in range(min(span) if span else page, (max(span) if span else page) + 1):
            if other == page or not (1 <= other <= total):
                continue
            try:
                other_ratio = page_support_ratio(content, page_lookup(file_name, other))
            except (KeyError, ValueError):
                continue
            if other_ratio > best[0]:
                best = (other_ratio, other)
        if best[0] >= min_page_support:
            return Check(f"引用可回溯（{label}）", True,
                         f"支撑原文在块覆盖的物理第 {best[1]} 页（该块 {best[0]:.0%} 命中；"
                         f"引用页 {page} 仅 {ratio:.0%}）")
        next_page = page + 1
        off_by_one = ""
        if next_page <= total:
            try:
                if page_support_ratio(content, page_lookup(file_name, next_page)) >= min_page_support:
                    off_by_one = (f"　⚠️ 该块内容出现在物理第 {next_page} 页（= 引用页 + 1）"
                                  f"→ 疑似 0-based 索引当页码（需求分析 §4.4 禁止）")
            except (KeyError, ValueError):
                off_by_one = ""
        return Check(f"引用可回溯（{label}）", False,
                     f"chunk {chunk_id} 只有 {ratio:.0%} 内容在该页（阈值 {min_page_support:.0%}）{off_by_one}")

    # ④ 无 chunk 回查时的兜底：用支撑原文（quote / golden citation_quote）核验
    needle = str(support or quote or "").strip()
    needle = strip_product_markers(needle) or needle
    if support_present(text, needle):
        return Check(f"引用可回溯（{label}）", True, f"支撑原文命中：{needle[:40]!r}")

    next_page = page + 1
    off_by_one = ""
    if next_page <= total:
        try:
            if support_present(page_lookup(file_name, next_page), needle):
                off_by_one = (f"　⚠️ 支撑原文出现在物理第 {next_page} 页（= 引用页 + 1）"
                              f"→ 判定为 0-based 索引当页码（需求分析 §4.4 禁止）")
        except (KeyError, ValueError):
            off_by_one = ""
    prev_hint = ""
    if page - 1 >= 1:
        try:
            if support_present(page_lookup(file_name, page - 1), needle):
                prev_hint = f"　（注：支撑原文也出现在第 {page - 1} 页）"
        except (KeyError, ValueError):
            prev_hint = ""
    return Check(f"引用可回溯（{label}）", False,
                 f"引用页未含支撑原文：{needle[:40]!r}{off_by_one}{prev_hint}")


def citation_accuracy(checks: Sequence[Check]) -> float:
    """引用正确率 = 有效引用条数 / 引用总条数（§2.4）。空集合返回 0.0。"""
    if not checks:
        return 0.0
    return sum(1 for c in checks if c.ok) / len(checks)


# ---------------------------------------------------------------------------
# 3) 首字（逐题判定；冷启动不豁免）
# ---------------------------------------------------------------------------
def check_first_token(
    latencies: Mapping[str, float],
    *,
    budget_ms: float = FIRST_TOKEN_BUDGET_MS,
) -> list[Check]:
    """逐题断言首字 ≤ 预算（取最大值判定），并给出每题数值。"""
    checks: list[Check] = []
    for key, value in latencies.items():
        ms = float(value or 0.0)
        checks.append(Check(
            f"首字 ≤ {int(budget_ms)} ms（{key}）", 0.0 < ms <= float(budget_ms), f"实测 {ms:.1f} ms"))
    if latencies:
        worst = max(float(v or 0.0) for v in latencies.values())
        checks.append(Check("首字最大值 ≤ 预算（逐题口径，非平均）", 0.0 < worst <= float(budget_ms),
                            f"最大值 {worst:.1f} ms / 样本 {len(latencies)} 题"))
    return checks


def first_token_hint() -> str:
    """口径提示：冷启动不作为豁免理由（写进失败报告，避免复用「首次慢是正常」的说法）。"""
    return ("首字按**每题**判定且不接受「冷启动慢属正常」的豁免：预热正是为消除该尖峰而设，"
            "缺预热调用点即缺陷（验收标准.md §3.4）。")


# ---------------------------------------------------------------------------
# 4) 表格缺陷（只认行内横向重复）+ 非缺陷白名单
# ---------------------------------------------------------------------------
@dataclass
class RowDuplication:
    """行内横向重复（缺陷）或行内重复候选。"""

    row_index: int
    col_index: int
    value: str
    defect: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """转字典。"""
        return asdict(self)


#: 占位符（原文本来就填这些符号表示「无数据」）——**不得**被当成「重复填充的逻辑值」
PLACEHOLDER_VALUES = frozenset({"-", "--", "---", "/", "—", "——", "－", "无", "不适用", "N/A", "n/a", "[ ◆ ]", "◆"})


def _cell(value: Any) -> str:
    """单元格归一化（供比较用）：去换行/空白、去首尾、全角空格折叠。"""
    return re.sub(r"\s+", "", str(value or "").replace("\u00a0", "")).strip()


def _is_meaningful(value: str) -> bool:
    """是否是有意义的「逻辑值」：非空、非占位符、且含中文或字母数字。

    §3.1 要抓的是「同一个**逻辑值**被复制填充到同一行的多个子列」；
    原文里成串的 ``-``（无数据占位）不是逻辑值，把它算成缺陷就是误判
    （PDF1 物理 130 表里就有成串 ``-``，属真实数据形态）。
    """
    if not value or value in PLACEHOLDER_VALUES:
        return False
    return bool(re.search(r"[\u4e00-\u9fff0-9a-zA-Z]", value))


def scan_row_duplications(rows: Sequence[Sequence[Any]], *,
                          ignore_placeholders: bool = True) -> list[RowDuplication]:
    """扫描**行内**（同一行相邻列）相同非空值。**不做任何跨行（纵向）判定**。

    这是 §3.1「缺陷正确定义」的实现基础：纵向同值（如 PDF1 物理 65/66 的
    「陈爱民/程家明」各期真实重复）在本函数里**结构上不可能**被判缺陷。
    ``-`` 这类无数据占位符默认跳过（见 ``PLACEHOLDER_VALUES``）。
    """
    found: list[RowDuplication] = []
    for row_index, row in enumerate(rows):
        cells = [_cell(c) for c in row]
        for col_index in range(len(cells) - 1):
            left, right = cells[col_index], cells[col_index + 1]
            if not (left and right and left == right):
                continue
            if ignore_placeholders and not _is_meaningful(left):
                continue
            found.append(RowDuplication(row_index, col_index, left, True,
                                        "同一行相邻列出现相同非空逻辑值"))
    return found


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """读取属性或映射键（兼容 ``TableBlock`` dataclass 与 JSONL 读出的 dict 两种形态）。"""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def table_horizontal_defects(block: Any) -> list[RowDuplication]:
    """按 §3.1 判据判定 ``TableBlock``（或 JSONL dict）的**行内横向重复缺陷**。

    判据 = 「同一行相邻列出现相同非空值」**且**「该行存在被拆分的合并表头」。
    后者用块自身的证据判断（``none_cells > 0`` 说明原表确有合并/None 被拆列；
    或 ``flat_cols > logical_cols`` 说明两级表头被压平产生了重复列）。
    退化表（``degenerate=True``）不进索引，其内容不作缺陷判定对象。
    """
    if bool(_attr(block, "degenerate", False)):
        return []
    rows = list(_attr(block, "rows", []) or [])
    candidates = scan_row_duplications(rows)
    if not candidates:
        return []
    merged_header = bool(int(_attr(block, "none_cells", 0) or 0) > 0)
    flattened = int(_attr(block, "flat_cols", 0) or 0) > int(_attr(block, "logical_cols", 0) or 0)
    if not (merged_header or flattened):
        for item in candidates:
            item.defect = False
            item.reason = "无「被拆分的合并表头」证据 → 按 §3.1 不判缺陷"
    return candidates


def non_defect_verdicts() -> list[dict[str, str]]:
    """返回非缺陷白名单副本（用例断言「未据此判缺陷」时可逐条打印）。"""
    return [dict(item) for item in NON_DEFECT_WHITELIST]


# ---------------------------------------------------------------------------
# 5) 「不清楚」/ 编造 / 互污染
# ---------------------------------------------------------------------------
UNKNOWN_TEXTS = ("不清楚", "不知道", "无法回答", "未披露相关", "no idea", "unknown", "cannot answer")
#: 可答性判定原因（``设计/接口设计.md`` §3.19 **枚举冻结**）——权威来源，不得自造枚举值
ANSWERABILITY_REASONS: tuple[str, ...] = (
    "empty_retrieval", "low_score", "low_coverage", "numeric_missing", "issuer_mismatch", "ok",
)
#: 主体类型闸门的 ``reason`` 取值（同节冻结）
GATE_REASONS: tuple[str, ...] = (
    "ok", "no_evidence_table", "not_applicable", "disabled",
    "person_leaked", "org_leaked", "triple_incomplete", "missing_number_or_relation",
)
#: ``Answer.unknown_reason`` 允许取值 = 可答性枚举 + 闸门原因（``qa_engine`` 以
#: ``subject_gate:<reason>`` 落库，见 captain 实测的 ``subject_gate:org_leaked``）
UNKNOWN_REASONS: tuple[str, ...] = (
    ANSWERABILITY_REASONS + GATE_REASONS + tuple(f"subject_gate:{item}" for item in GATE_REASONS)
)


def check_unknown_answer(answer: Any, *, label: str = "") -> list[Check]:
    """无依据拒答断言：``is_unknown=True``、正文含「不清楚」、**无任何引用**。"""
    text = str(getattr(answer, "text", "") or "")
    citations = list(getattr(answer, "citations", []) or [])
    reason = getattr(answer, "unknown_reason", None)
    return [
        Check(f"无依据回「不清楚」（{label}）", bool(getattr(answer, "is_unknown", False)),
              f"is_unknown={getattr(answer, 'is_unknown', None)} text={text[:40]!r}"),
        Check(f"拒答正文规范（{label}）", any(word in text for word in UNKNOWN_TEXTS),
              f"text={text[:60]!r}"),
        Check(f"拒答不带任何引用（{label}）", not citations,
              f"引用 {[getattr(c, 'render', lambda: c)() for c in citations][:3]}"),
        Check(f"unknown_reason 在枚举内（{label}）", reason in UNKNOWN_REASONS, f"reason={reason!r}"),
    ]


def check_no_fabrication(answer: Any, *, forbidden_substrings: Sequence[str], label: str = "") -> list[Check]:
    """编造判定：拒答答案里**不得**出现被禁止的数值/结论片段（如 15,000 万元）。

    数值型禁项走 :func:`token_in_text` 的数字边界匹配，避免 ``15,000`` 被 ``115,000`` 误伤/漏伤。
    """
    text = str(getattr(answer, "text", "") or "")
    return [Check(f"未编造「{token}」（{label}）", not token_in_text(text, token),
                  f"text={text[:60]!r}") for token in forbidden_substrings]


def token_in_text(text: str, token: str) -> bool:
    """归一化后在文本里查找 token，且**数值型 token 必须数字边界匹配**。

    为什么要边界：``check_leakage`` 用禁项 ``0万元`` 去判「占位符是否被当成 0」时，
    朴素子串会命中 ``3,393.40万元`` 里的 ``0万元`` → 假阳性（实测把一道正确答案判红）。
    因此凡 token 以数字开头的，都要求左边不是数字/小数点。
    """
    flat = squash(text)
    needle = squash(token)
    if not needle:
        return False
    start = 0
    while True:
        index = flat.find(needle, start)
        if index < 0:
            return False
        if not needle[0].isdigit():
            return True
        left = flat[index - 1] if index > 0 else ""
        if not (left.isdigit() or left == "."):
            return True
        start = index + 1


def check_leakage(
    answer_text: str,
    *,
    forbidden: Sequence[str] = (),
    required: Sequence[str] = (),
    label: str = "",
) -> list[Check]:
    """同页互污染（N-5）与要素齐备断言：``required`` 必须全含，``forbidden`` 必须全不含。

    注意（captain 裁定）：题 4 的 ``forbidden`` 含自然人「赵马克」是**严格**的；
    而「力源贸易 / 普芯达」是**企业**、属题 4 答案，**不得**被误删（写进 ``required``）。
    """
    out: list[Check] = []
    for token in required:
        out.append(Check(f"答案含必需要素「{token}」（{label}）", token_in_text(answer_text, token),
                         f"答案摘要={str(answer_text)[:60]!r}"))
    for token in forbidden:
        out.append(Check(f"答案未混入「{token}」（{label}）", not token_in_text(answer_text, token),
                         f"答案摘要={str(answer_text)[:80]!r}"))
    return out


# ---------------------------------------------------------------------------
# 6) 判分口径（产品 bridge → 工单1 权威 Evaluator（只读）→ 本地副本，来源必标注）
# ---------------------------------------------------------------------------
def judge_answer(answer_text: str, golden: str) -> tuple[bool, str, str]:
    """调用判分口径，返回 ``(是否正确, 理由, 口径来源)``。

    判分链与来源标注见 ``common/judge_local.py``。实测：产品冻结模块
    ``app/core/evaluator_bridge.py``（设计 §3.23）缺失，因此会落到 **工单1 权威 Evaluator（只读）**；
    来源字符串必须原样写进测试报告，便于 T11 判断「口径是否合规」。
    """
    from . import judge_local  # noqa: PLC0415 —— 延迟导入

    return judge_local.judge(answer_text, golden)


def accuracy_gate(correct: int, total: int = TOTAL_QUESTIONS,
                  threshold: float = ACCURACY_THRESHOLD) -> list[Check]:
    """准确率门槛：``correct/total ≥ 0.90``（14 题即 ≥13 题）。"""
    ratio = (correct / total) if total else 0.0
    return [
        Check(f"语义正确题数 ≥ {ACCURACY_MIN_CORRECT}/{total}", correct >= ACCURACY_MIN_CORRECT,
              f"实测 {correct}/{total}"),
        Check(f"准确率 ≥ {threshold:.2f}", ratio >= threshold, f"实测 {ratio:.4f}"),
    ]


# ---------------------------------------------------------------------------
# 7) 主体类型闸门（N-6 / N-5）
# ---------------------------------------------------------------------------
def check_subject_gate_fail_open(gate: Any, *, label: str) -> list[Check]:
    """N-6：正文证据题（题 34/793）闸门必须 fail-open：``ok=True`` 且 ``counted=False``。

    反向断言同样固化：**不得**把 ``counted=False`` 当作「闸门真正生效」的通过证据。
    """
    ok = bool(getattr(gate, "ok", False))
    counted = bool(getattr(gate, "counted", False))
    reason = str(getattr(gate, "reason", "") or "")
    allowed = list(getattr(gate, "allowed", []) or [])
    return [
        Check(f"闸门 fail-open：ok=True（{label}）", ok, f"ok={ok} reason={reason}"),
        Check(f"闸门 counted=False 且 allowed 为空（{label}）", (not counted) and not allowed,
              f"counted={counted} allowed={allowed}"),
        Check(f"闸门不得判「不可答」（{label}）", reason in ("no_evidence_table", "not_applicable", "no_allowed_set", ""),
              f"reason={reason!r}"),
    ]


# ---------------------------------------------------------------------------
# 8) RAGAS 标注（强制注入，禁止伪造数值）
# ---------------------------------------------------------------------------
def ragas_banner() -> str:
    """返回必须原样写入报告首行的 RAGAS 未运行标注。"""
    return RAGAS_BANNER


def warn_ragas_banner(path: Path | str) -> Path:
    """在报告文件里**前置**注入 RAGAS 标注（已存在则只校验，不重复插入）。"""
    target = Path(path)
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    if text.startswith(RAGAS_BANNER):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(RAGAS_BANNER + "\n" + text, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# 9) 工具自检（合成数据；证明「工具不误判」，与产品实现无关）
# ---------------------------------------------------------------------------
def toolkit_self_check() -> Report:
    """用合成数据自检断言工具本身：纵向重复不误报、行内重复必报、0-based 引用必抓。"""
    report = Report("断言工具自检（合成数据，不依赖产品代码）")

    # ① 纵向（跨行）同值：PDF1 65/66 场景造形 —— 必须 0 条缺陷
    vertical = [["陈爱民", "程家明"], ["陈爱民", "程家明"], ["陈爱民", "程家明"]]
    report.check("纵向同值不算缺陷（陈爱民/程家明 造形）", scan_row_duplications(vertical) == [],
                 f"扫描结果 {len(scan_row_duplications(vertical))} 条")

    # ② 行内横向重复：合并单元格被拆列后每列填同值 —— 必须抓到
    horizontal = [["军种一", "军种一", "100", "200"]]
    hits = scan_row_duplications(horizontal)
    report.check("行内横向重复必被抓到", len(hits) == 1 and hits[0].col_index == 0,
                 f"命中 {[(h.row_index, h.col_index, h.value) for h in hits]}")

    # ②b 占位符成串**不算**缺陷（PDF1 物理 130 真实形态：原文用 '-' 表示无数据）
    dash_row = [["-", "-", "-", "1", "2"], ["", "", "3", "4", "5"]]
    report.check("无数据占位符 '-' 成串不算缺陷", scan_row_duplications(dash_row) == [],
                 f"命中 {[(h.row_index, h.col_index, h.value) for h in scan_row_duplications(dash_row)]}")

    # ③ 无「被拆分合并表头」证据时不判缺陷（避免误报）
    class _Block:
        """最小 TableBlock 造形。"""

        degenerate = False
        none_cells = 0
        logical_cols = 4
        flat_cols = 4
        rows = [["A", "A", "1", "2"]]

    report.check("无合并表头证据 → 不判缺陷",
                 all(not item.defect for item in table_horizontal_defects(_Block())),
                 "§3.1 判据要求「存在被拆分的合并表头」")

    # ④ 印刷页码形态识别
    report.check("识别页脚印刷页码 1-1-128", looks_like_printed_page_form("1-1-128"))
    report.check("识别裸数字索引 128", looks_like_printed_page_form("128"))
    report.check("识别非页码文本", not looks_like_printed_page_form("序号1"))

    # ⑤ 引用 0-based 陷阱：引用页 = 128（真实物理页 129 的 0-based 索引）→ 必须判失败
    page_store = {"p.pdf": {128: "无关内容", 129: "证据原文在此：82.10%、97.31%"}}

    def lookup(file_name: str, page: int) -> str:
        """合成页文本查询。"""
        return page_store[file_name][page]

    bad = check_citation_traceable({"file_name": "p.pdf", "page": 128},
                                   page_lookup=lookup, page_counts={"p.pdf": 300},
                                   discovered_files=["p.pdf"], support="82.10%、97.31%")
    report.check("0-based 错页引用必被判失败", not bad.ok and "0-based" in bad.detail, bad.detail)
    good = check_citation_traceable({"file_name": "p.pdf", "page": 129},
                                    page_lookup=lookup, page_counts={"p.pdf": 300},
                                    discovered_files=["p.pdf"], support="82.10%、97.31%")
    report.check("正确 1-based 引用判通过", good.ok, good.detail)

    # ⑥ 首字逐题口径：3300 ms 那一题必须失败（最大值判定同样失败）
    checks = check_first_token({"q1": 1200.0, "q2": 3300.0})
    report.check("首字逐题判定（3300 ms 必须失败）",
                 sum(1 for c in checks if c.ok) == 1 and not all(c.ok for c in checks),
                 "　".join(c.detail for c in checks))

    # ⑦ 互污染判定方向性（题 4：力源贸易必须在 required 一侧）
    leak = check_leakage("7 家企业为融冰投资、武汉博润、上海博润、听音投资、联众聚源、力源贸易、普芯达",
                         required=["力源贸易", "普芯达"], forbidden=["赵马克"], label="题4造形")
    report.check("互污染判定：企业必含、自然人必无", all(c.ok for c in leak), leak[0].detail)

    # ⑧ RAGAS 标注不得被改写
    report.check("RAGAS 标注原文一致", ragas_banner() == "RAGAS 未运行（依赖不可用，本机断网）",
                 ragas_banner())

    # ⑨ 误拒答判定（captain 实测案例：题 4 曾被闸门误判 org_leaked）
    refused_rows = [{"id": 3, "is_unknown": False},
                    {"id": 4, "is_unknown": True, "unknown_reason": "subject_gate:org_leaked"}]
    refused_checks = check_no_false_refusal(refused_rows)
    report.check("误拒答必被抓到（题 4 org_leaked 造形）",
                 not all(c.ok for c in refused_checks),
                 "　".join(c.render() for c in refused_checks if not c.ok))
    clean_rows = [{"id": qid, "is_unknown": False} for qid in (1, 2, 3, 4)]
    report.check("无误拒答时判通过", all(c.ok for c in check_no_false_refusal(clean_rows)))

    # ⑩ 闸门期望判定（题 4：ok=True、organization、counted=True、allowed 覆盖 7 家）
    class _Gate:
        """最小 SubjectGateResult 造形。"""

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    good_gate = _Gate(ok=True, expected="organization", counted=True,
                      allowed=list(SEVEN_COMPANIES), found=list(SEVEN_COMPANIES[:3]),
                      leaked=[], reason="ok")
    bad_gate = _Gate(ok=False, expected="organization", counted=True,
                     allowed=list(SEVEN_COMPANIES), found=[], leaked=["赵马克"], reason="org_leaked")
    report.check("闸门期望：正常结果判通过",
                 all(c.ok for c in check_gate_expectation(good_gate, 4)))
    report.check("闸门期望：误拒答结果必被判失败",
                 not all(c.ok for c in check_gate_expectation(bad_gate, 4)),
                 "　".join(c.render() for c in check_gate_expectation(bad_gate, 4) if not c.ok))
    report.check("题面契约可查询（题 4 题面 + allowed 必备集合）",
                 check_question_text(4, QUESTION_CONTRACT[4]["question"]).ok
                 and expected_allowed_subset(4) == SEVEN_COMPANIES
                 and not check_question_text(4, "改过的题面").ok)
    return report
