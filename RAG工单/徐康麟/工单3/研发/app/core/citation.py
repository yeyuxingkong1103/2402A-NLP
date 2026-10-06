# -*- coding: utf-8 -*-
"""工单3 引用管理（设计/接口设计.md §2.4、§3.17 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

引用格式（对外统一）：
    * 中文：``[招股说明书1.pdf: 129]``（等价简写 ``[页码: 129]``，不带文件名时由 chunk 反查补全）
    * 英文：``[招股说明书1.pdf: Page: 129]``（等价简写 ``[Page: 129]``）
页码一律 **1-based 物理 PDF 页码**；校验时用 ``doc[page-1]`` 回读，越界即非法。

校验四规则（§3.17）：
    ① 页码合法（``1 <= page <= page_counts[file_name]``，文件名必须在 page_counts 里）；
    ② 块可查（有 ``chunk_id`` 时必须能查到，且 ``chunk.page == citation.page``）；
    ③ 有支撑（引用处的证据文本需含 ``quote``；给了 ``answer_text`` 时，**答案正文本身**必须能在引用处核实）；
    ④ 主体类型（给了 ``subject_gate`` 且其 ``ok is False`` → 不计 valid 并追加 reason）。

T6 补充（实测依据，见 部署/配置/环境事实.md §14）：
    * 表格块的证据是**表格 Markdown**（如 ``| 注册资本 | 5,520.00 万元 |``），它天然不等于页正文文本层
      （正文层里是空白分隔的散字），因此证据文本按「引用块内容 ∪ 引用页正文」取；
    * 3B 级小模型会照抄提示词里的示例页码（旧模板写了 ``129``），故生成后必须**回验页码**：
      引用处核不出答案 → 在返回块里找真正支撑答案的块并**改指**（retarget）；找不到 → 该引用判非法并剔除。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import Any, Callable, Mapping, Sequence

from .chunker import Chunk
from .errors import CitationError
from .text_utils import STOPWORDS, evidence_contains, extract_numbers, squash_text, tokenize

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
UNKNOWN_TEXT = "不清楚"        # 无依据时的统一答复（与 answerability.UNKNOWN_TEXT 同值；本地定义避免循环导入）

# 答案可回溯硬判据（T6 在 14 题实测上标定，见 §14.6）
SENTENCE_MIN_COVERAGE = 0.60   # 答案 token 在引用处证据里的覆盖率下限
NUMBER_MIN_HIT = 0.50          # 答案数值在引用处证据里的命中率下限（数值保真优先）
MIN_ANSWER_CHARS = 4           # 答案正文最小长度（少于 4 字视为空答案）
PHRASE_WITH_NUMBER = 4         # 有数值锚点时要求的最短原话片段（4 字）
PHRASE_NO_NUMBER = 6           # 无数值锚点时要求的最短原话片段（6 字，避开各页页眉的通用词）

# 「答案：」「答：」「Answer:」等标签；「引用：」「References:」等引用行
_LABEL_PATTERN = re.compile(r"^[\s【\[]*(?:答案正文|答案|回答|结论|答|answer|conclusion)[\s】\]]*[:：]?", re.I)
_REF_LINE_PATTERN = re.compile(r"^[\s【\[]*(?:引用(?:来源|依据)?|来源|参考(?:文献)?|references?|citations?|sources?)[\s】\]]*[:：]", re.I)
_NUM_CLEAN_PATTERN = re.compile(r"[,，\s\u00a0]")
# 数值在 token 覆盖率里会被切碎（15 / 000 / 00 各算一个 token）从而虚低压分；数值另有专判，分词前先摘掉
_NUMBER_LITERAL_PATTERN = re.compile(r"\d[\d,，.·%]*")
# 答案开头的「发行人全称」在每页页眉都会出现，做原话片段判定时必须先剔除，否则会假命中
_LEADING_ORG_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9（）()]{2,40}?(?:股份有限公司|有限责任公司|有限公司)")


def longest_common_phrase_length(answer: str, evidence: str, *, lengths: Sequence[int] = (8, 6, 4)) -> int:
    """答案与证据的**最长连续原话**长度（滑动窗口包含判定，避免 O(n·m) 动态规划）。

    做判定前先剔除答案开头的发行人全称：它在每一页页眉都出现，否则会与任意页假命中
    （实测：题 957 的答案前缀「武汉兴图新科电子股份有限公司」会命中 p129 页眉）。
    """
    squashed_answer, squashed_evidence = squash_text(answer), squash_text(evidence)
    if not squashed_answer or not squashed_evidence:
        return 0
    trimmed = _LEADING_ORG_PATTERN.sub("", squashed_answer, count=1)
    ordered = sorted((int(n) for n in lengths), reverse=True)
    source = trimmed if len(trimmed) >= max(ordered) else squashed_answer
    for length in ordered:
        if len(source) < length:
            continue
        for start in range(0, len(source) - length + 1):
            candidate = source[start:start + length]
            if candidate and candidate in squashed_evidence:
                return length
    return 0

# 引用正文：可选的「文件名: 」前缀（或模型照抄标签文字的「文件名: 」占位）+ 可选的「页码:/Page: 」标签 + 数字页码
# 尾部容忍模型照抄片段标签的残余（如 ``[文件名: 94 / 类型: 正文]``）
_CITE_CORE = (r"\[\s*(?:(?P<file>[^\[\]:：]*?\.pdf)\s*[:：]\s*|文件名\s*[:：]\s*)?"
              r"(?:(?:{label})\s*[:：]\s*)?(?P<page>\d{{1,4}})(?:\s*/\s*[^\[\]]*)?\s*\]")
# 片段标签回声（小模型会把「[1] 招股书.pdf: 129 / 类型: 正文」当引用抄出来）：解析前先剔除，
# 否则其中「[1]」会被误当成页码 1
_LABEL_ECHO_PATTERN = re.compile(r"\[\d{1,3}\]\s*[^\[\]]*?\.pdf\s*[:：]\s*\d{1,4}[^\[\]]*")
CITATION_PATTERN_ZH = re.compile(_CITE_CORE.format(label=r"页码"))
CITATION_PATTERN_EN = re.compile(_CITE_CORE.format(label=r"Page"))
# 兼容写法（同时认「页码」「Page」两种标签）；解析时统一用它
CITATION_PATTERN_ANY = re.compile(_CITE_CORE.format(label=r"页码|Page"))


def _lazy_logger(logger: Any, module: str = "citation") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class Citation:
    """一条引用（``page`` 为 1-based 物理页）。"""

    file_name: str
    page: int
    chunk_id: str | None = None
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Citation":
        names = {f.name for f in dc_fields(cls)}
        payload = {k: v for k, v in data.items() if k in names}
        payload.setdefault("file_name", "")
        payload["page"] = int(payload.get("page") or 0)
        return cls(**payload)

    def key(self) -> tuple[str, int]:
        return (self.file_name, int(self.page))

    def render(self, *, language: str = "zh") -> str:
        """渲染为对外引用串。"""
        label = "Page" if str(language).lower().startswith("en") else "页码"
        if self.file_name:
            return f"[{self.file_name}: {self.page}]" if label == "页码" else f"[{self.file_name}: {label}: {self.page}]"
        return f"[{label}: {self.page}]"


@dataclass(slots=True)
class CitationReport:
    """引用校验报告（``accuracy = valid/total``）。"""

    total: int
    valid: int
    invalid: list[Citation] = field(default_factory=list)
    accuracy: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "valid": self.valid,
                "invalid": [c.to_dict() for c in self.invalid],
                "accuracy": self.accuracy, "reasons": self.reasons}


def format_citation(file_name: str, page: int) -> str:
    """``("[招股说明书1.pdf: 129]")``——对外唯一格式。"""
    return f"[{file_name}: {int(page)}]"


def parse_citations(text: str) -> list[Citation]:
    """从答案文本中解析引用（不带文件名时 ``file_name=""``，由 attach 阶段补全）。"""
    cleaned = _LABEL_ECHO_PATTERN.sub(" ", str(text or ""))
    found: list[Citation] = []
    for match in CITATION_PATTERN_ANY.finditer(cleaned):
        file_name = (match.group("file") or "").strip()
        page = int(match.group("page"))
        found.append(Citation(file_name=file_name, page=page))
    return found


def attach_citations(text: str, chunks: Sequence[Any], *, restrict_to_chunks: bool = True,
                     logger: Any = None) -> list[Citation]:
    """解析引用并用 chunk 反查补全文件名。

    ``restrict_to_chunks=True``（默认）时，**只保留能落到本次返回块上的引用**：
    不带文件名的页码须有同页块；带文件名的引用须命中 (文件名, 页码) 相同的返回块。
    原因（T6 实测）：小模型会凭空写出未在上下文里的页码（如照抄提示词示例的 ``129``），
    这类引用无法回溯到检索证据，一律丢弃；真正依据由 ``retarget_citations`` 重新落位。
    """
    log = _lazy_logger(logger)
    with log.enter("attach_citations", {"text_chars": len(str(text or "")), "chunks": len(chunks),
                                        "restrict_to_chunks": restrict_to_chunks}) as span:
        by_page: dict[int, Any] = {}
        for chunk in chunks:
            by_page.setdefault(int(getattr(chunk, "page", 0) or 0), chunk)
        citations: list[Citation] = []
        dropped = 0
        for cite in parse_citations(text):
            matched = next((c for c in chunks
                            if (not cite.file_name or str(getattr(c, "file_name", "")) == cite.file_name)
                            and int(getattr(c, "page", 0)) == cite.page), None)
            if matched is None:
                if restrict_to_chunks:
                    dropped += 1
                    log.log_event("citation.attach_dropped", level="WARNING", page=cite.page,
                                  file_name=cite.file_name,
                                  reason="引用页码不在返回块内（不可回溯到检索证据）")
                    continue
                if not cite.file_name:                       # 旧行为：仅补全文件名用
                    chunk = by_page.get(cite.page)
                    if chunk is None:
                        dropped += 1
                        log.log_event("citation.attach_dropped", level="WARNING", page=cite.page,
                                      reason="引用页码不在返回块中（未落地的页码不采信）")
                        continue
                    cite.file_name = str(getattr(chunk, "file_name", ""))
                    cite.chunk_id = str(getattr(chunk, "chunk_id", "")) or None
                    cite.quote = str(getattr(chunk, "content", ""))[:120]
            else:
                if not cite.file_name:
                    cite.file_name = str(getattr(matched, "file_name", ""))
                cite.chunk_id = str(getattr(matched, "chunk_id", "")) or None
                cite.quote = str(getattr(matched, "content", ""))[:120]
            citations.append(cite)
        log.log_event("citation.attach", count=len(citations), pages=[c.page for c in citations],
                      dropped=dropped)
        span.set_output({"citations": len(citations), "dropped": dropped})
        return citations


def dedupe_citations(citations: Sequence[Citation]) -> list[Citation]:
    """按 (文件名, 页码) 去重（保留首个）。"""
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for cite in citations:
        key = cite.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(cite)
    return out


def normalize_number(value: str) -> str:
    """归一化数值文本：去千分位/空白、整数化尾随小数（``5,520.00 万元`` → ``5520``）。"""
    raw = _NUM_CLEAN_PATTERN.sub("", str(value or "")).rstrip("%").lower()
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", raw)
    if not match:
        return raw
    integer, frac = match.group(1), (match.group(2) or "").rstrip("0")
    return f"{integer}.{frac}" if frac else integer


def numbers_in(text: str) -> list[str]:
    """答案/证据里的数值清单（原样保留，比较时再归一化）。"""
    return [str(n) for n in extract_numbers(str(text or ""))]


def strip_answer_labels(text: str) -> str:
    """去掉「答案：」「引用：…」等标签与引用标记，得到纯答案正文（用于可回溯判定）。"""
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = CITATION_PATTERN_ANY.sub("", raw_line).strip()
        if not line or _REF_LINE_PATTERN.match(line):
            continue
        while True:                                     # 「答案：答案：正文」多层标签也去掉
            match = _LABEL_PATTERN.match(line)
            if not match:
                break
            stripped = line[match.end():].strip()
            if stripped == line:
                break
            line = stripped
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def answer_body(answer_text: str) -> str:
    """答案正文（去标签后的文本）。"""
    return strip_answer_labels(answer_text)


def is_empty_answer(answer_text: str, *, min_chars: int = MIN_ANSWER_CHARS) -> bool:
    """答案是否为空/过短（只给引用、只给标签都算空答案）。"""
    body = answer_body(answer_text)
    if body.strip() in {"", UNKNOWN_TEXT, "unknown", "不知道"}:
        return body.strip() != UNKNOWN_TEXT
    return len(re.sub(r"[\s\W_]+", "", body)) < int(min_chars)


_FIELD_STATEMENT_PATTERN = re.compile(
    r"^(?P<field>[\u4e00-\u9fffA-Za-z0-9（）()]{2,20}?)(?:是|为)(?P<value>[^。！？；，；,]{1,40})$")
_FIELD_STATEMENT_TRAILING = "。！？；，,;."


def field_statement_supported(answer_text: str, evidence_text: str, *,
                              logger: Any = None) -> tuple[bool, str]:
    """**字段判断句**容忍判定（t16，§19.2）：整句形如「字段是/为取值」时，判断动词可由作答层补全。

    实测动机：字段型提问的规范答法是「法定代表人是程家明。」，而招股意向书原始表格是
    「法定代表人：程家明」（字段名 5 字 + 取值 3 字）。要求「≥6 字连续原话」时，判断动词「是」
    恰好把两段原话断开（实测最长连续 4 < 6）→ 引用被判「答案不在引用处」。
    容忍条件（**严到不能被用来编造**）：
        ① 整句只允许是「字段 + 是/为 + 取值」这一个判断句（正则全匹配，长答案塞不进来）；
        ② **字段名与取值都必须在引用处逐字出现**（去空白与标点后）；
        ③ 两者在引用处**先后顺序一致**（字段在前、取值在后）。
    任一条件不满足 → 不认可（返回 False），仍走原有「连续原话」判定。
    """
    log = _lazy_logger(logger)
    with log.enter("field_statement_supported", {"answer_chars": len(str(answer_text or "")),
                                                 "evidence_chars": len(str(evidence_text or ""))}) as span:
        body = answer_body(answer_text).strip().strip(_FIELD_STATEMENT_TRAILING)
        evidence = squash_text(evidence_text)
        if not body or not evidence:
            span.set_output({"supported": False, "why": "答案或证据为空"})
            return False, "答案或证据为空"
        match = _FIELD_STATEMENT_PATTERN.match(body)
        if not match:
            span.set_output({"supported": False, "why": "不是单一字段判断句"})
            return False, "不是单一字段判断句"
        field = squash_text(match.group("field"))
        value = squash_text(match.group("value"))
        if len(field) < 2 or not value:
            span.set_output({"supported": False, "why": "字段名或取值过短"})
            return False, "字段名或取值过短"
        field_at, value_at = evidence.find(field), evidence.find(value)
        ok = field_at >= 0 and value_at >= 0 and value_at > field_at
        why = (f"字段判断句：字段「{match.group('field')}」与取值「{match.group('value')}」"
               f"均在引用处逐字出现且先后一致（位置 {field_at}/{value_at}）" if ok else
               f"字段名或取值不在引用处（字段位置 {field_at}、取值位置 {value_at}）")
        log.log_event("citation.field_statement_check", level="" if ok else "DEBUG", supported=ok,
                      field=str(match.group("field")), file_field_at=field_at, value_at=value_at)
        span.set_output({"supported": ok, "why": why})
        return ok, why


def answer_support_check(answer_text: str, evidence_text: str, *,
                         threshold: float = SENTENCE_MIN_COVERAGE, logger: Any = None) -> tuple[bool, str]:
    """答案正文能否在证据文本里核实（数值命中率 + token 覆盖率，引用可回溯的唯一判据）。"""
    log = _lazy_logger(logger)
    with log.enter("answer_support_check", {"answer_chars": len(str(answer_text or "")),
                                            "evidence_chars": len(str(evidence_text or ""))}) as span:
        body = answer_body(answer_text)
        if not body:
            span.set_output({"supported": False, "why": "答案正文为空"})
            return False, "答案正文为空"
        evidence = squash_text(evidence_text)
        if not evidence:
            span.set_output({"supported": False, "why": "引用处无证据文本"})
            return False, "引用处无证据文本"
        if squash_text(body) in evidence:
            span.set_output({"supported": True, "why": "答案整句原文命中"})
            return True, "答案整句原文命中"
        numbers = numbers_in(body)
        hit_rate = 1.0
        if numbers:
            evidence_numbers = {normalize_number(n) for n in numbers_in(evidence_text)}
            hit = [n for n in numbers if normalize_number(n) in evidence_numbers]
            hit_rate = round(len(hit) / len(numbers), 4)
            if hit_rate < NUMBER_MIN_HIT:
                why = f"数值命中率 {hit_rate} < {NUMBER_MIN_HIT}（命中 {hit or '无'}）"
                span.set_output({"supported": False, "why": why})
                return False, why
        tokens = [t for t in tokenize(_NUMBER_LITERAL_PATTERN.sub(" ", body))
                  if len(t) >= 2 and t not in STOPWORDS]
        if not tokens:
            tokens = [t for t in re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", _NUMBER_LITERAL_PATTERN.sub(" ", body))]
        # 纯数值答案（「15,000 万元」这类）：文字 token 全被过滤掉时，只要数值**逐字命中**证据即认可
        # —— 这正是「数值保真 + 可回溯」的场景（T7/t12 实测题 207 的 `15,000 万元`）
        if not tokens and numbers and hit_rate >= NUMBER_MIN_HIT:
            why = f"仅数值命中（{len(numbers)} 个，命中率 {hit_rate}）"
            span.set_output({"supported": True, "why": why})
            return True, why
        evidence_tokens = set(tokenize(_NUMBER_LITERAL_PATTERN.sub(" ", str(evidence_text))))
        coverage = round(len([t for t in tokens if t in evidence_tokens]) / len(tokens), 4) if tokens else 0.0
        if coverage < threshold:
            why = f"片段覆盖率 {coverage} < {threshold}（token {len(tokens)} 个）"
            span.set_output({"supported": False, "why": why})
            return False, why
        # 原话片段：纯词面重合度会被「同域vocabulary + 页眉」抬高，必须要求一段连续原话
        need_phrase = PHRASE_WITH_NUMBER if numbers else PHRASE_NO_NUMBER
        phrase = longest_common_phrase_length(body, evidence_text,
                                              lengths=(8, 6, PHRASE_WITH_NUMBER))
        if phrase < need_phrase:
            # t16（§19.2）：字段判断句的「判断动词」由作答层补全时会断开连续原话，
            # 但字段名与取值都逐字来自引用处 → 按字段判断句容忍一次（条件见 field_statement_supported）
            field_ok, field_why = field_statement_supported(body, evidence_text, logger=log)
            if field_ok:
                why = f"{field_why}；最长连续原话 {phrase} < {need_phrase}（判断动词由作答层补全）"
                span.set_output({"supported": True, "why": why})
                return True, why
            why = (f"缺连续原话片段（最长 {phrase} < {need_phrase} 字）：词面覆盖 {coverage} 但无法逐字核对")
            span.set_output({"supported": False, "why": why})
            return False, why
        why = f"数值命中率 {hit_rate}、片段覆盖率 {coverage}、最长原话 {phrase} 字"
        span.set_output({"supported": True, "why": why})
        return True, why


def evidence_text_for(cite: Citation, *, chunk_lookup: Callable[[str], Chunk | None] | None = None,
                      page_text_lookup: Callable[[str, int], str] | None = None) -> str:
    """引用处的证据文本 = 引用块内容（表格块即 Markdown 表）∪ 引用页正文。"""
    parts: list[str] = []
    if chunk_lookup is not None and cite.chunk_id:
        chunk = chunk_lookup(str(cite.chunk_id))
        if chunk is not None:
            parts.append(str(getattr(chunk, "content", "") or ""))
    if page_text_lookup is not None and cite.file_name:
        parts.append(str(page_text_lookup(cite.file_name, int(cite.page)) or ""))
    return "\n".join(p for p in parts if p)


def retarget_citations(answer_text: str, citations: Sequence[Citation], chunks: Sequence[Any], *,
                       evidence_lookup: Callable[[Citation], str] | None = None,
                       logger: Any = None) -> tuple[list[Citation], list[dict[str, Any]]]:
    """回验并改正引用页码：引用处核不出答案 → 改指真正支撑答案的返回块；都核不出则保留待判非法。

    说明：小模型常照抄示例页码或用「排名第一的块」代替「真正含依据的块」。此处只在**已返回的检索块**
    内改指，绝不新造页码，因此仍是可回溯的（改指动作全部落日志与报告）。
    """
    log = _lazy_logger(logger)
    with log.enter("retarget_citations", {"citations": len(citations), "chunks": len(chunks)}) as span:
        actions: list[dict[str, Any]] = []
        out: list[Citation] = []
        for cite in citations:
            evidence = evidence_lookup(cite) if evidence_lookup is not None else (cite.quote or "")
            ok, why = answer_support_check(answer_text, evidence, logger=log)
            if ok:
                out.append(cite)
                continue
            replacement: Citation | None = None
            for candidate in chunks:
                content = str(getattr(candidate, "content", "") or "")
                candidate_ok, candidate_why = answer_support_check(answer_text, content, logger=log)
                if candidate_ok:
                    replacement = Citation(file_name=str(getattr(candidate, "file_name", "")),
                                           page=int(getattr(candidate, "page", 0)),
                                           chunk_id=str(getattr(candidate, "chunk_id", "")) or None,
                                           quote=content[:120])
                    actions.append({"action": "retarget", "from": cite.to_dict(), "to": replacement.to_dict(),
                                    "why": why, "matched_by": candidate_why})
                    log.log_event("citation.retarget", level="WARNING", from_file=cite.file_name,
                                  from_page=cite.page, to_file=replacement.file_name, to_page=replacement.page,
                                  why=why)
                    break
            if replacement is not None and replacement.key() != cite.key():
                out.append(replacement)
                continue
            actions.append({"action": "unsupported", "citation": cite.to_dict(), "why": why})
            log.log_event("citation.unsupported", level="WARNING", file_name=cite.file_name, page=cite.page,
                          why=why)
            out.append(cite)
        deduped = dedupe_citations(out)
        span.set_output({"citations": len(deduped), "retargeted": sum(1 for a in actions if a["action"] == "retarget"),
                         "unsupported": sum(1 for a in actions if a["action"] == "unsupported")})
        return deduped, actions


def best_supporting_chunk(answer_text: str, chunks: Sequence[Any], *, logger: Any = None) -> Any:
    """在返回块里找**最先能核实答案**的块（答案完全无引用时的落位依据；找不到返回 ``None``）。"""
    log = _lazy_logger(logger)
    with log.enter("best_supporting_chunk", {"chunks": len(chunks)}) as span:
        for candidate in chunks:
            ok, why = answer_support_check(answer_text, str(getattr(candidate, "content", "") or ""), logger=log)
            if ok:
                span.set_output({"chunk_id": str(getattr(candidate, "chunk_id", "")), "why": why})
                return candidate
        span.set_output({"chunk_id": None, "why": "无任何返回块能核实答案"})
        return None


def render_answer(answer_text: str, citations: Sequence[Citation], *, language: str = "zh") -> str:
    """用**已通过校验**的引用重建答案文本（正文不再残留任何未校验页码）。"""
    body = answer_body(answer_text)
    if not citations:
        return body
    label = "References" if str(language).lower().startswith("en") else "引用"
    return f"{body}\n{label}：{render_inline(citations, language=language)}"


def validate_citations(citations: Sequence[Citation], *, page_counts: Mapping[str, int],
                       chunk_lookup: Callable[[str], Chunk | None],
                       page_text_lookup: Callable[[str, int], str] | None = None,
                       evidence_lookup: Callable[[Citation], str] | None = None,
                       answer_text: str | None = None,
                       subject_gate: Any = None,
                       logger: Any = None) -> CitationReport:
    """按 §3.17 四规则校验引用，返回 ``CitationReport``（accuracy 参与验收 5）。"""
    log = _lazy_logger(logger)
    with log.enter("validate_citations", {"total": len(citations), "files": sorted(page_counts)}) as span:
        invalid: list[Citation] = []
        reasons: list[str] = []
        valid = 0
        for cite in citations:
            page = int(cite.page)
            if cite.file_name not in page_counts:
                invalid.append(cite)
                reasons.append(f"文件名不在语料内：{cite.file_name}")
                log.log_event("citation.invalid", level="WARNING", file_name=cite.file_name, page=page,
                              reason="file_not_found")
                continue
            page_count = int(page_counts[cite.file_name])
            if page < 1 or page > page_count:
                invalid.append(cite)
                reasons.append(f"页码越界：{cite.file_name} p{page}（1~{page_count}）")
                log.log_event("citation.invalid", level="WARNING", file_name=cite.file_name, page=page,
                              reason="page_out_of_range", page_count=page_count)
                continue
            if cite.chunk_id:
                chunk = chunk_lookup(cite.chunk_id)
                if chunk is None or int(getattr(chunk, "page", 0)) != page:
                    invalid.append(cite)
                    reasons.append(f"引用块不可查或页码不符：{cite.chunk_id} → p{page}")
                    log.log_event("citation.invalid", level="WARNING", file_name=cite.file_name, page=page,
                                  reason="chunk_mismatch", chunk_id=cite.chunk_id)
                    continue
            if evidence_lookup is not None:
                support_text = str(evidence_lookup(cite) or "")
            elif page_text_lookup is not None:
                support_text = str(page_text_lookup(cite.file_name, page) or "")
            else:
                support_text = ""
            if answer_text is not None and support_text:
                supported, why = answer_support_check(answer_text, support_text, logger=log)
                if not supported:
                    invalid.append(cite)
                    reasons.append(f"答案不在引用处：{cite.file_name} p{page}（{why}）")
                    log.log_event("citation.invalid", level="WARNING", file_name=cite.file_name, page=page,
                                  reason="answer_not_supported", why=why)
                    continue
            if support_text and cite.quote:
                if not (squash_text(cite.quote) in squash_text(support_text)
                        or evidence_contains(support_text, cite.quote, threshold=0.9)):
                    invalid.append(cite)
                    reasons.append(f"引用处无支撑原文：{cite.file_name} p{page}")
                    log.log_event("citation.invalid", level="WARNING", file_name=cite.file_name, page=page,
                                  reason="no_support", quote_digest=cite.quote[:40])
                    continue
            if page_text_lookup is not None and not support_text and cite.quote:
                invalid.append(cite)
                reasons.append(f"引用页无文本可核：{cite.file_name} p{page}")
                log.log_event("citation.invalid", level="WARNING", file_name=cite.file_name, page=page,
                              reason="page_text_missing")
                continue
            valid += 1
        if subject_gate is not None and getattr(subject_gate, "ok", True) is False:
            reasons.append(f"主体类型闸门未通过：{getattr(subject_gate, 'reason', 'unknown')}")
            log.log_event("citation.subject_gate_block", level="WARNING",
                          reason=getattr(subject_gate, "reason", "unknown"), valid_before=valid)
            valid = 0
        total = len(citations)
        accuracy = round(valid / total, 4) if total else 0.0
        report = CitationReport(total=total, valid=valid, invalid=invalid, accuracy=accuracy, reasons=reasons)
        log.log_event("citation.validate", total=total, valid=valid, invalid=len(invalid), accuracy=accuracy)
        span.set_output(report.to_dict())
        return report


def ensure_citation(answer_text: str, chunks: Sequence[Any], *, language: str = "zh") -> str:
    """答案无引用时补最后一条引用（保证「有答案必有引用」）；无块可引用则原样返回。"""
    text = str(answer_text or "").strip()
    if parse_citations(text) or not chunks:
        return text
    top = chunks[0]
    return f"{text}\n引用：{format_citation(str(getattr(top, 'file_name', '')), int(getattr(top, 'page', 0)))}"


def render_inline(citations: Sequence[Citation], *, language: str = "zh") -> str:
    """把引用列表渲染成一行内联文本。"""
    return " ".join(cite.render(language=language) for cite in dedupe_citations(citations))


def citation_summary(text: str) -> str:
    """答案文本里的引用摘要（日志用）。"""
    cites = parse_citations(text)
    return " ".join(f"{c.file_name or '?'}:{c.page}" for c in cites) or "无"


def raise_if_invalid(report: CitationReport) -> None:
    """校验报告全非法时抛 ``CitationError``（供需要硬失败的上层使用）。"""
    if report.total and report.valid == 0:
        raise CitationError(f"全部引用非法：{report.reasons[:3]}", detail=report.to_dict())
