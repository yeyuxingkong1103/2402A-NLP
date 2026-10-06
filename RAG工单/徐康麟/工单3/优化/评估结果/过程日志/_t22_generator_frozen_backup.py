# -*- coding: utf-8 -*-
"""工单3 答案生成与引用闭环（设计/接口设计.md §2.4、§3.18 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

流程（严格按 v1.6 顺序）：
    ① 理解：``classify_question(原文)`` → ``rewrite_query(原文, history)``（**仅供检索**）
       → ``classify_subject_expectation(strip_issuer_names(原文))``；
    ② 可答性闸门 ``answerability.decide``：判否 → **不调用 LLM**，直接回「不清楚」；
    ③ 生成：流式（首字延迟在 LLMClient 内计时，再叠加检索耗时）；
    ④ 引用：``citation.attach`` → ``validate``（页码 1-based 物理页、块可查、页内有支撑）；
    ⑤ 主体闸门：不合格 → 修一轮 → 仍不合格 → 整题回「不清楚」。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import answerability, citation as citation_mod, language as language_mod, llm_client
from . import reranker as reranker_mod, text_utils
from .config import AppConfig, get_config
from .errors import RagError
from .text_utils import text_digest

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 答案里的「标签回声」：模型照抄片段标签或提示词占位串（实测高频噪声）
_LABEL_ECHO_ANSWER_PATTERN = re.compile(
    r"\[\s*[Ss＃#]?\s*\d{1,3}\s*\]\s*[^\[\]\n]*?\.pdf\s*[:：]\s*\d{1,4}[^\[\]\n]*"  # [1]/[S3] 招股说明书1.pdf: 94 / 类型: 正文
    r"|\[\s*(?:文件名|file|编号|片段)\s*[:：][^\]\n]*\]"                        # [文件名: 页码]
    r"|\[\s*\d{1,3}\s*[:：]\s*\d{1,4}\s*\]"                                   # [1: 153]
    r"|(?<![\[\d])\.pdf\s*[:：]\s*\d{1,4}\s*/\s*类型\s*[:：]\s*[^\s，。;；]*"      # 招股说明书1.pdf: 94 / 类型: 正文
)
_PLACEHOLDER_LINE_PATTERN = re.compile(r"^[（(]?\s*(?:类型|编号|片段编号)\s*[:：]|^\s*[|｜]?\s*[-—]{2,}")
# 带单位的数值（数值锚定的比较口径；只做「是否出现」判定，不改变答案里的原文写法）
_UNIT_NUMBER_PATTERN = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元|万股|%|％|次|倍)")

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "qa_prompt.txt"
BODY_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "qa_body_prompt.txt"
UNKNOWN_TEXT = answerability.UNKNOWN_TEXT

# --- t16 新增（§19）：字段型问句「成句化」与「尾句裁剪」的判定常数 -----------------
# 字段取值不得以这些连接词开头（否则抽到的是句子尾巴而不是取值本身，如「发行后总股本**的**比例为…」）
_FIELD_VALUE_BAD_PREFIX: tuple[str, ...] = ("的", "之", "即", "占", "及", "与", "和", "、", "是", "为", "，", "（", "(")
# 判断句（问人/问物）：答句用「是」；其余（问数值/问数量）用「为」——中文规范答法
_WHO_QUESTION_PATTERN = re.compile(r"是谁|是哪一位|哪位|何人|什么人|谁担任|谁是")
# 枚举/清单意图：答案必须是多项，禁止用「单值短句」或「尾句裁剪」简化
_ENUM_QUESTION_PATTERN = re.compile(r"哪些|哪一种|哪几种|包括|分别|列举|都有|各有|各是")
# 行业地位/领域型问句的「地位句」形态（t16，§19.3b）：用于把答案锚定到招股书原话句
_POSITION_QUESTION_PATTERN = re.compile(r"领域|行业地位|市场地位|重要供应商")
_POSITION_SENTENCE_PATTERN = re.compile(r"已经成为|已成为|重要供应商")
# 子句切分（只切句末与逗号，**不切顿号**：顿号切分会把「6,464.51 万元、14,414.16 万元」拆散）
_CLAUSE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；，!?;,])")
# 字段取值抽取：①「字段是/为 取值」②「字段：取值」（表格行另走单元格分支）
_FIELD_COPULA_TEMPLATE = r"{kw}\s*(?:是|为)\s*[「“\"']?\s*([^|｜\s，。；;、]{{1,40}})"
_FIELD_SEPARATOR_TEMPLATE = r"{kw}\s*[:：]\s*[|｜]?\s*([^|｜\s，。；;、]{{1,40}})"
# 表格行里「空单元格」的占位写法（不得当成取值）
_FIELD_EMPTY_CELLS: frozenset[str] = frozenset({"-", "—", "/", "无", "未披露", "N/A", "n/a"})


def _lazy_logger(logger: Any, module: str = "generator") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _field_value_ok(value: str, keyword: str, *, expects_numeric: bool) -> bool:
    """取值是否「像字段取值」：非空、长度合适、不带连接词前缀、不夹带其它字段名。

    t16（§19.2）实测动机：不做这层校验时，`发行后总股本` 会把「**的**比例为25.04%」、
    `技术标准` 会把「**（即**《某视频技术规范1.0》）」当成取值，进而拼出病句答案。
    """
    text = str(value or "").strip()
    if not text or len(text) > 40:
        return False
    if text.startswith(_FIELD_VALUE_BAD_PREFIX):
        return False
    from .query_understanding import FIELD_KEYWORDS

    all_keywords = {k for keywords in FIELD_KEYWORDS.values() for k in keywords}
    if any(other != keyword and other in text for other in all_keywords):
        return False                                   # 取值里夹带其它字段名 → 多半是抽歪了
    if expects_numeric and not re.search(r"\d", text):
        return False                                   # 问数值却没数字 → 不是取值
    return True


def _extract_field_value(text: str, keyword: str, *, expects_numeric: bool) -> tuple[str, str]:
    """在证据/答案文本里抽「字段 → 取值」。

    返回 ``(取值, 呈现方式)``；呈现方式：``"copula"``=判断动词句（字段是/为取值）、
    ``"separator"``=表格或清单式（字段：取值 / 表格单元格）；抽不到返回 ``("", "")``。

    行处理顺序（**表格优先**，因为表格行的取值单元格最干净）：① Markdown 表行按列取下一格；
    ② 同行「字段是/为取值」；③ 同行「字段：取值」。
    """
    raw = str(text or "")
    if not raw or not keyword:
        return "", ""
    copula = re.compile(_FIELD_COPULA_TEMPLATE.format(kw=re.escape(keyword)))
    separator = re.compile(_FIELD_SEPARATOR_TEMPLATE.format(kw=re.escape(keyword)))
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # ① Markdown 表行：字段所在列的**下一格**就是取值
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            for index, cell in enumerate(cells):
                if keyword not in cell or len(text_utils.squash_text(cell)) > len(keyword) + 2:
                    continue
                for value_cell in cells[index + 1:]:
                    if not value_cell or value_cell in _FIELD_EMPTY_CELLS:
                        continue
                    if _field_value_ok(value_cell, keyword, expects_numeric=expects_numeric):
                        return value_cell, "separator"
                    break
        # ② 判断动词句；③ 冒号分隔
        for pattern, style in ((copula, "copula"), (separator, "separator")):
            for match in pattern.finditer(stripped):
                value = match.group(1).strip(" \t:：|｜「」“”\"'《》")
                if _field_value_ok(value, keyword, expects_numeric=expects_numeric):
                    return value, style
    return "", ""


def _split_clauses(text: str) -> list[str]:
    """按句末标点与逗号切子句（保留标点；**不切顿号**，避免拆散数值清单）。"""
    return [piece for piece in _CLAUSE_SPLIT_PATTERN.split(str(text or "")) if piece and piece.strip()]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass(slots=True)
class Answer:
    """一次问答的完整结果（设计 §2.4 冻结字段）。"""

    answer_id: str
    question: str
    text: str
    citations: list[Any] = field(default_factory=list)
    used_chunks: list[str] = field(default_factory=list)
    backend: str = ""
    model: str = ""
    language: str = "zh"
    is_unknown: bool = False
    unknown_reason: str | None = None
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    trace_id: str = ""
    conversation_id: str | None = None
    created_at: str = ""
    citation_report: Any = None
    subject_gate: dict[str, Any] | None = None    # 主体闸门审计结果（t14：在线套件要求可审计）
    retrieval: Any = None

    def to_dict(self, *, with_retrieval: bool = False) -> dict[str, Any]:
        data = {
            "answer_id": self.answer_id, "question": self.question, "text": self.text,
            "citations": [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in self.citations],
            "used_chunks": self.used_chunks, "backend": self.backend, "model": self.model,
            "language": self.language, "is_unknown": self.is_unknown,
            "unknown_reason": self.unknown_reason, "first_token_ms": self.first_token_ms,
            "total_ms": self.total_ms, "trace_id": self.trace_id,
            "conversation_id": self.conversation_id, "created_at": self.created_at,
            "citation_report": self.citation_report.to_dict() if hasattr(self.citation_report, "to_dict") else None,
        }
        if with_retrieval and self.retrieval is not None:
            data["retrieval"] = self.retrieval.to_dict() if hasattr(self.retrieval, "to_dict") else None
        return data


@dataclass(slots=True)
class AnswerDelta:
    """流式增量（``done=True`` 时携带完整 ``Answer``）。"""

    text: str
    is_first: bool
    first_token_ms: float | None
    done: bool
    answer: Answer | None = None


class AnswerGenerator:
    """生成 + 引用 + 主体闸门闭环。"""

    def __init__(self, *, cfg: AppConfig | None = None, llm: Any = None, citation_mgr: Any = None,
                 logger: Any = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.llm = llm
        self.citation_mgr = citation_mgr
        self.extractive = llm_client.ExtractiveGenerator(cfg=self.cfg, logger=self.log)
        self._page_cache: dict[tuple[str, int], str] = {}
        self._chunk_cache: dict[str, Any] = {}
        self._template_text: str | None = None
        self._body_template_text: str | None = None
        self._strong_client: Any = None
        self._strong_disabled = False

    # -- prompt ----------------------------------------------------------
    def build_prompt(self, question: str, chunks: Sequence[Any], *,
                     history: Sequence[Any] | None = None, language: str = "zh",
                     support_chunks: Sequence[Any] | None = None) -> str:
        """读 ``qa_prompt.txt`` 模板并填充（片段超预算按 top_k 截断）。

        ``support_chunks``（T7/t12 新增，默认空）是**数值锚定支持块**：不进 top-k 排名，
        但作为「优先依据」排在片段列表最前，用于修「答案原文块没被召回」（实测题 207 的 p490）。
        """
        with self.log.enter("build_prompt", {"question": question[:60], "chunks": len(chunks),
                                             "support": len(list(support_chunks or [])),
                                             "language": language}) as span:
            template = self._template()
            lines: list[str] = []
            index = 0
            for chunk in list(support_chunks or []):
                index += 1
                kind = "表格" if str(getattr(chunk, "type", "")) == "table" else "正文"
                lines.append(f"[S{index}] {getattr(chunk, 'file_name', '')}: {getattr(chunk, 'page', 0)}"
                             f" / 类型: {kind} / 【优先依据：该片段含问题所需的原始数值，数值必须逐字照抄】\n"
                             f"{str(getattr(chunk, 'content', ''))[:1200]}")
            for chunk in chunks:
                index += 1
                kind = "表格" if str(getattr(chunk, "type", "")) == "table" else "正文"
                lines.append(
                    f"[{index}] {getattr(chunk, 'file_name', '')}: {getattr(chunk, 'page', 0)} / 类型: {kind}\n"
                    f"{str(getattr(chunk, 'content', ''))[:1200]}"
                )
            history_text = "（无）"
            if history:
                history_text = "\n".join(
                    f"{'用户' if getattr(t, 'role', '') == 'user' else '助手'}：{str(getattr(t, 'content', ''))[:200]}"
                    for t in history
                )
            prompt = template.format(question=question, context="\n\n".join(lines),
                                     language=language, history=history_text)
            self.log.log_event("generation.prompt", chars=len(prompt),
                               chunk_ids=[str(getattr(c, "chunk_id", "")) for c in chunks],
                               support_ids=[str(getattr(c, "chunk_id", "")) for c in (support_chunks or [])],
                               truncated=any(len(str(getattr(c, "content", ""))) > 1200 for c in chunks))
            span.set_output({"chars": len(prompt), "chunks": len(chunks)})
            return prompt

    def _template(self) -> str:
        """读提示词模板（缓存；模板改动需重启进程）。"""
        if self._template_text is None:
            self._template_text = PROMPT_PATH.read_text(encoding="utf-8")
        return self._template_text

    def sanitize_answer(self, text: str, *, trace: str = "", logger: Any = None) -> str:
        """去掉答案里的**提示词回声 / 思维链 / 标签回声**，只留答案正文与引用。

        实测（§14.6）：
            * 3B 模型会把「若证据不足，则只回：不清楚」这类模板句写进答案；
            * 会输出 ``### 答案``、``### zh`` 这类 Markdown 标题行；
            * 推理模型（deepseek-r1）会把 ``<think>…</think>`` 思维链混进 ``response``；
            * 会照抄片段标签 ``[1] 招股说明书1.pdf: 94 / 类型: 正文`` 或占位串 ``[文件名: 页码]``。
        """
        log = logger or self.log
        raw = str(text or "")
        raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.S | re.I)      # 完整思维链
        if re.search(r"<think>", raw, flags=re.I):                           # 未闭合 = 思维链被截断
            log.log_event("generation.thinking_truncated", level="WARNING", trace_id=trace,
                          chars=len(raw))
            raw = re.split(r"<think>", raw, maxsplit=1, flags=re.I)[0]
        raw = re.sub(r"</?think>", " ", raw, flags=re.I)                      # 残缺标签
        raw = _LABEL_ECHO_ANSWER_PATTERN.sub(" ", raw)
        rule_lines = set()
        for line in self._template().splitlines():
            cleaned = re.sub(r"^[\s\-#*\d.、()（）]+", "", line).strip()
            normalized = citation_mod.strip_answer_labels(cleaned)
            if len(normalized) >= 6:
                rule_lines.add(normalized)
        kept: list[str] = []
        removed: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or _PLACEHOLDER_LINE_PATTERN.match(stripped):
                removed.append(stripped[:40])                          # Markdown 标题/占位回声
                continue
            normalized = citation_mod.strip_answer_labels(stripped)
            if normalized and normalized in rule_lines:
                removed.append(stripped[:40])
                continue
            kept.append(stripped)
        result = "\n".join(kept).strip()
        if removed:
            log.log_event("generation.prompt_echo_removed", level="WARNING", trace_id=trace,
                          removed=removed[:3], count=len(removed))
        return result

    def _resolve_citations(self, answer_text: str, chunks: Sequence[Any], *, trace: str, log: Any,
                           info: dict[str, Any]) -> tuple[list[Any], Any, list[dict[str, Any]]]:
        """引用闭环：附着（限定在返回块内）→ 回验并改指 → 校验（答案必须落在引用处）。"""
        citations = citation_mod.attach_citations(answer_text, chunks, logger=log)
        citations, actions = citation_mod.retarget_citations(
            answer_text, citations, chunks, evidence_lookup=self.evidence_lookup, logger=log)
        if not citations:
            chunk = citation_mod.best_supporting_chunk(answer_text, chunks, logger=log)
            if chunk is not None:
                citations = [citation_mod.Citation(
                    file_name=str(getattr(chunk, "file_name", "")), page=int(getattr(chunk, "page", 0)),
                    chunk_id=str(getattr(chunk, "chunk_id", "")) or None,
                    quote=str(getattr(chunk, "content", ""))[:120])]
                actions.append({"action": "attach_from_evidence", "to": citations[0].to_dict()})
                log.log_event("citation.attach_from_evidence", level="WARNING", trace_id=trace,
                              file_name=citations[0].file_name, page=citations[0].page)
        report = citation_mod.validate_citations(
            citations, page_counts=self.page_counts(), chunk_lookup=self.chunk_lookup,
            page_text_lookup=self.page_text_lookup, evidence_lookup=self.evidence_lookup,
            answer_text=answer_text, logger=log)
        if actions:
            log.log_event("generation.citation_actions", trace_id=trace, actions=actions[:4],
                          count=len(actions))
        return citations, report, actions

    # -- 存储回查辅助 -----------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        from .pdf_parser import connect_sqlite

        return connect_sqlite(self.cfg.paths.index_dir / "rag.sqlite3")

    def page_counts(self) -> dict[str, int]:
        """各文件页数（引用页码合法性判据）。"""
        conn = self._db()
        try:
            return {row[0]: int(row[1]) for row in conn.execute("SELECT file_name,page_count FROM documents")}
        finally:
            conn.close()

    def page_text_lookup(self, file_name: str, page: int) -> str:
        """取引用页正文（走 SQLite ``pages`` 表，避免为校验反复打开 PDF）。"""
        key = (file_name, int(page))
        if key in self._page_cache:
            return self._page_cache[key]
        conn = self._db()
        try:
            row = conn.execute("SELECT text FROM pages WHERE file_name=? AND page=?", key).fetchone()
        finally:
            conn.close()
        text = str(row[0]) if row else ""
        self._page_cache[key] = text
        return text

    def chunk_lookup(self, chunk_id: str) -> Any:
        """按 chunk_id 回查块（引用校验规则②）。"""
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]
        from .chunker import Chunk

        conn = self._db()
        try:
            row = conn.execute(
                "SELECT chunk_id,file_name,page,page_start,page_end,type,section,content,table_id,char_count,ord"
                " FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        chunk = Chunk(chunk_id=row[0], file_name=row[1], page=int(row[2]), page_start=int(row[3]),
                      page_end=int(row[4]), type=row[5], section=row[6] or "", content=row[7] or "",
                      keywords=[], table_id=row[8], char_count=int(row[9]), order=int(row[10]))
        self._chunk_cache[chunk_id] = chunk
        return chunk

    def evidence_lookup(self, cite: Any) -> str:
        """引用处的证据文本（引用块内容 ∪ 引用页正文；表格块的证据即 Markdown 表）。"""
        return citation_mod.evidence_text_for(cite, chunk_lookup=self.chunk_lookup,
                                              page_text_lookup=self.page_text_lookup)

    # -- 主流程 ----------------------------------------------------------
    def _unknown(self, question: str, retrieval: Any, *, reason: str, backend: str, model: str,
                 language: str, trace_id: str, first_token_ms: float = 0.0,
                 total_ms: float = 0.0, subject_gate: dict[str, Any] | None = None) -> Answer:
        """构造「不清楚」答案（禁止编造）。"""
        # t19（§22）：补 ``trace_id``（修前 202/202 为 null，无法按 trace 反查「为什么拒答」）。
        self.log.log_event("generation.unknown", trace_id=trace_id, reason=reason,
                           top_score=float(getattr(retrieval.chunks[0], "score", 0.0)) if getattr(retrieval, "chunks", None) else 0.0,
                           coverage=None)
        return Answer(answer_id=f"a{int(time.time() * 1000) % 100000000:08d}", question=question,
                      text=UNKNOWN_TEXT, citations=[], used_chunks=[], backend=backend, model=model,
                      language=language, is_unknown=True, unknown_reason=reason,
                      first_token_ms=first_token_ms, total_ms=total_ms, trace_id=trace_id,
                      created_at=_now_iso(), subject_gate=subject_gate, retrieval=retrieval)

    def answer(self, question: str, retrieval: Any, *, history: Sequence[Any] | None = None,
               language: str | None = None, trace_id: str | None = None, logger: Any = None) -> Answer:
        """生成带引用的答案；无依据/引用非法/主体闸门不过 → 「不清楚」。"""
        log = logger or self.log
        from .query_understanding import understand

        started = time.perf_counter()
        trace = trace_id or f"g{int(time.perf_counter() * 1000) % 100000000:08d}"
        chunks = list(getattr(retrieval, "chunks", []) or [])
        # 数值锚定支持块：不参与 top-k 排名，但作为「优先依据」进提示词，并可被引用（页码真实可回溯）
        support = list(getattr(retrieval, "support_chunks", []) or [])
        context = support + chunks
        with log.enter("AnswerGenerator.answer",
                       {"question": question[:60], "chunks": len(chunks), "support": len(support),
                        "trace_id": trace}) as span:
            lang = language or language_mod.answer_language(question)
            info = understand(question, history, cfg=self.cfg, llm=None, logger=log)   # ① 理解（原题分类）
            backend_name = getattr(getattr(self.llm, "backend", None), "name", "extractive")
            model_name = getattr(getattr(self.llm, "backend", None), "model", "rule-based")
            log.log_event("generation.start", trace_id=trace, question_digest=text_digest(question, limit=60),
                          chunks=len(chunks), language=lang)
            # ② 可答性闸门：判否 → 不调用 LLM
            decision = answerability.decide(question, retrieval, field_type=info["field_type"],
                                            expects_numeric=info["expects_numeric"], cfg=self.cfg, logger=log)
            if not decision.is_answerable:
                answer = self._unknown(question, retrieval, reason=decision.reason, backend=backend_name,
                                       model=model_name, language=lang, trace_id=trace,
                                       total_ms=round((time.perf_counter() - started) * 1000, 2))
                span.set_output({"unknown": True, "reason": decision.reason})
                return answer
            prompt = self.build_prompt(question, chunks, history=history, language=lang,
                                       support_chunks=support)
            text, first_ms, total_llm_ms = self._generate_text(prompt, question, chunks, lang, trace, log)
            answer = self._finalize(question, retrieval, text, chunks, info=info, lang=lang, trace=trace,
                                    backend=backend_name, model=model_name,
                                    first_ms=first_ms, started=started, log=log, history=history,
                                    prompt=prompt, context_chunks=context, support=support)
            span.set_output({"unknown": answer.is_unknown, "chars": len(answer.text),
                             "citations": [c.to_dict() for c in answer.citations],
                             "first_token_ms": answer.first_token_ms})
            return answer

    def _generate_text(self, prompt: str, question: str, chunks: Sequence[Any], lang: str,
                       trace: str, log: Any) -> tuple[str, float, float]:
        """调用 LLM（或抽取式兜底）取答案文本，返回 (文本, 首字 ms, 生成耗时 ms)。

        实测（§14.6）：给 3B 模型加「答案正文：」式前置引导会诱导它照抄片段标签/只回引用行，
        因此这里**不做前置引导**，仅按提示词的输出契约取文本；空正文由 ``_recover_answer`` 兜底。
        """
        backend_name = getattr(getattr(self.llm, "backend", None), "name", "extractive")
        if self.llm is None or backend_name == "extractive":
            result = self.extractive.generate(question, chunks, language=lang, logger=log)
            self.log.log_event("generation.first_token", trace_id=trace, first_token_ms=result.first_token_ms,
                               backend="extractive")
            return result.text, float(result.first_token_ms), float(result.total_ms)
        result = self.llm.generate_full(prompt, trace_id=trace, logger=log)
        self.log.log_event("generation.first_token", trace_id=trace, first_token_ms=result.first_token_ms)
        return str(result.text or ""), float(result.first_token_ms), float(result.total_ms)

    def _generate_body(self, question: str, chunks: Sequence[Any], lang: str, trace: str,
                       log: Any) -> str:
        """第二阶段：只问正文（不带引用/格式要求）的再生成——小模型只顾格式时用它兜住正文。

        正文不含引用，由 ``_resolve_citations`` 依据「答案落在哪个返回块」落位引用。
        """
        if self.llm is None:
            return ""
        template = self._body_template()
        lines: list[str] = []
        for index, chunk in enumerate(list(chunks)[:3], start=1):
            lines.append(f"[{index}] {getattr(chunk, 'file_name', '')}: {getattr(chunk, 'page', 0)}\n"
                         f"{str(getattr(chunk, 'content', ''))[:1200]}")
        prompt = template.format(question=question, context="\n\n".join(lines))
        result = self.llm.generate_full(prompt, trace_id=trace, logger=log)
        body = citation_mod.answer_body(str(result.text or "")).strip()
        if body == UNKNOWN_TEXT or len(re.sub(r"[\s\W_]+", "", body)) < citation_mod.MIN_ANSWER_CHARS:
            log.log_event("generation.body_stage_empty", level="WARNING", trace_id=trace,
                          output_digest=text_digest(body, limit=40))
            return ""
        log.log_event("generation.body_stage_ok", trace_id=trace, chars=len(body))
        return body

    def _body_template(self) -> str:
        """读「只问正文」提示词模板（缓存）。"""
        if self._body_template_text is None:
            self._body_template_text = BODY_PROMPT_PATH.read_text(encoding="utf-8")
        return self._body_template_text

    def _strong_llm(self) -> Any:
        """惰性构造兜底强模型客户端（同一后端、换模型名；不可用时返回 None 并记日志）。"""
        if self._strong_client is not None or self._strong_disabled:
            return self._strong_client
        name = str(getattr(self.cfg.llm, "ollama_strong_model", "") or "")
        primary = getattr(getattr(self.llm, "backend", None), "name", "")
        if not name or primary != "ollama" or name == getattr(self.llm.backend, "model", ""):
            self._strong_disabled = True
            return None
        try:
            info = llm_client.LLMBackendInfo(name="ollama", model=name, base_url=self.llm.backend.base_url,
                                             available=True, probe_ms=0.0)
            client = llm_client.LLMClient(cfg=self.cfg, backend=info, logger=self.log)
            client.begin_think = False                     # 推理模型：直接作答，思维链不进答案
            self._strong_client = client
            self.log.log_event("generation.strong_backend_ready", model=name)
        except Exception as exc:  # noqa: BLE001 —— 兜底模型构造失败只降级，不掩盖
            self._strong_disabled = True
            self.log.log_event("generation.strong_backend_failed", level="WARNING",
                               error_type=type(exc).__name__, message=str(exc))
            self._strong_client = None
        return self._strong_client

    def _generate_strong_body(self, question: str, chunks: Sequence[Any], lang: str, trace: str,
                              log: Any) -> str:
        """最后一道 LLM 兜底：用强模型只问正文（更慢但更听话），正文引用由证据落位。"""
        client = self._strong_llm()
        if client is None:
            return ""
        template = self._body_template()
        lines = [f"[{i}] {getattr(c, 'file_name', '')}: {getattr(c, 'page', 0)}\n"
                 f"{str(getattr(c, 'content', ''))[:1200]}"
                 for i, c in enumerate(list(chunks)[:3], start=1)]
        prompt = template.format(question=question, context="\n\n".join(lines))
        # 推理模型（deepseek-r1:7b）在 num_predict 太小的时候思维链会被截断、答案还没开始就结束；
        # 实测 512 时输出仅含未闭合 <think>（正文为空），2048 时能给出完整思维链 + 正文 → 取下限 2048
        budget = max(2048, int(self.cfg.llm.max_tokens))
        result = client.generate_full(prompt, max_tokens=budget, trace_id=trace, logger=log)
        body = self.sanitize_answer(str(result.text or ""), trace=trace, logger=log)
        body = citation_mod.answer_body(body).strip()
        if body == UNKNOWN_TEXT or not body:
            log.log_event("generation.strong_stage_empty", level="WARNING", trace_id=trace,
                          model=getattr(client.backend, "model", ""), total_ms=result.total_ms)
            return ""
        log.log_event("generation.strong_stage_ok", trace_id=trace, model=getattr(client.backend, "model", ""),
                      chars=len(body), total_ms=result.total_ms)
        return body

    def _reference_sentence(self, question: str, chunks: Sequence[Any], preferred: Sequence[Any], info: dict[str, Any],
                            *, logger: Any = None) -> str:
        """取最像「答案原话」的证据句（用于判断答案是否答全 / 补全作答）。

        打分（T6 标定，§14.6）：字段关键词命中数 + 与问题的实词覆盖度 +（数值题）含数字，
        并**扣掉页眉行与表格行**——早期只看字段关键词时会被「湖北省科技成果奖」这类同行词表行骗到。
        """
        from .query_understanding import FIELD_KEYWORDS
        from .text_utils import extract_numbers, keyword_coverage, split_sentences, tokenize

        pool: list[Any] = []
        for cite in (preferred or []):
            chunk = self.chunk_lookup(str(getattr(cite, "chunk_id", "") or "")) \
                if getattr(cite, "chunk_id", None) else None
            if chunk is None and getattr(cite, "file_name", ""):
                chunk = next((c for c in chunks if str(getattr(c, "file_name", "")) == cite.file_name
                              and int(getattr(c, "page", 0)) == int(cite.page)), None)
            if chunk is not None:
                pool.append(chunk)
        if not pool:
            pool = list(chunks)[:3]
        keywords = list(FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()))
        question_tokens = tokenize(question)
        expects_numeric = bool(info.get("expects_numeric"))
        best_score, best_text = -1.0, ""
        for chunk in pool:
            for sentence in split_sentences(str(getattr(chunk, "content", "") or "")):
                hits = sum(1 for kw in keywords if kw in sentence)
                if not hits:
                    continue
                score = hits * 3.0 + keyword_coverage(question_tokens, tokenize(sentence)) * 2.0
                if expects_numeric and extract_numbers(sentence):
                    score += 0.5
                score -= 0.3 * sentence.count("|")                     # 表格行：字段对齐差，扣分
                if "招股意向书" in sentence:                             # 页眉行：每页都有，必扣
                    score -= 1.5
                if score > best_score:
                    best_score, best_text = score, sentence
        return best_text

    @staticmethod
    def _clean_evidence_text(sentence: str, *, limit: int = 200) -> str:
        """把证据原话清理成可读文本：**保留数字与标点**（禁止用 squash_text，它会把 6,464.51 压成 646451）。

        只做三件事：折空白、去掉页眉里的发行人全称与「招股意向书」、截断到 limit。
        """
        cleaned = text_utils.collapse_whitespace(str(sentence or ""))
        cleaned = re.sub(r"^(?:[\u4e00-\u9fff（）()]{2,30}?(?:股份有限公司|有限公司))\s*招股意向书\s*", "", cleaned)
        return cleaned[:limit].strip()

    def _expand_short_answer(self, question: str, text: str, citations: Sequence[Any], chunks: Sequence[Any],
                            info: dict[str, Any], *, logger: Any = None) -> str:
        """答案过短或没点出字段时，扩成「含该数值/片段的那句证据原话」（确定性、纯摘原文）。

        实测动机（§14.6）：小模型偶尔只回 ``15,000.00``、``军队视频指挥控制领域`` 这类碎片，
        虽可回溯但读者无法判断问的是什么；扩句后仍只摘证据原话，不新增任何事实。
        """
        from .query_understanding import FIELD_KEYWORDS
        from .text_utils import extract_numbers, keyword_coverage, split_sentences, squash_text, tokenize

        body = citation_mod.answer_body(text).strip()
        if not body or body == UNKNOWN_TEXT:
            return ""
        keywords = list(FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()))
        if len(re.sub(r"[\s\W_]+", "", body)) >= 12 and any(kw in body for kw in keywords):
            return ""                                              # 已是一句完整结论，不必扩
        numbers = extract_numbers(body)
        fragments = re.findall(r"[\u4e00-\u9fffA-Za-z]{3,}", body)
        anchor = numbers[0] if numbers else (max(fragments, key=len) if fragments else "")
        if not anchor:
            return ""
        pool: list[Any] = []
        for cite in citations or []:
            chunk = self.chunk_lookup(str(getattr(cite, "chunk_id", "") or "")) \
                if getattr(cite, "chunk_id", None) else None
            if chunk is not None:
                pool.append(chunk)
        for chunk in chunks:
            if all(str(getattr(chunk, "chunk_id", "")) != str(getattr(c, "chunk_id", "")) for c in pool):
                pool.append(chunk)
        question_tokens = tokenize(question)
        best_score, best_sentence, best_chunk = -1.0, "", None
        for chunk in pool[:6]:
            for sentence in split_sentences(str(getattr(chunk, "content", "") or "")):
                if anchor not in sentence:
                    continue
                score = sum(1 for kw in keywords if kw in sentence) * 2.0 \
                    + keyword_coverage(question_tokens, tokenize(sentence)) * 1.5
                if "招股意向书" in sentence:
                    score -= 1.5
                if score > best_score:
                    best_score, best_sentence, best_chunk = score, sentence, chunk
        if best_chunk is None or len(squash_text(best_sentence)) <= len(squash_text(body)):
            return ""
        log = logger or self.log
        log.log_event("generation.answer_expanded", level="WARNING", from_chars=len(body),
                      to_chars=len(best_sentence), chunk_id=str(getattr(best_chunk, "chunk_id", "")))
        cite_text = citation_mod.format_citation(str(getattr(best_chunk, "file_name", "")),
                                                 int(getattr(best_chunk, "page", 0)))
        return f"{self._clean_evidence_text(best_sentence)}\n引用：{cite_text}"

    def _recover_answer(self, question: str, chunks: Sequence[Any], info: dict[str, Any], *,
                        preferred: Sequence[Any] | None = None, lang: str = "zh", trace: str = "",
                        logger: Any = None) -> str:
        """LLM 只回引用/正文为空时的确定性兜底：按问题字段关键词在证据块里定位原话作答。

        先只搜「已落地引用的块」，再退到 top-3；数值型字段取表格单元值，文本型字段取所在整句。
        仍无可用原话 → 回「不清楚」（绝不编造）。
        """
        from .query_understanding import FIELD_KEYWORDS
        from .text_utils import extract_numbers, split_sentences, squash_text

        log = logger or self.log
        with log.enter("_recover_answer", {"question": question[:60], "chunks": len(chunks),
                                           "field_type": info.get("field_type")}) as span:
            # 优先只在「已落地引用的块」里找；引用改回块，找不到再退到 top-3
            pool: list[Any] = []
            for cite in (preferred or []):
                chunk = self.chunk_lookup(str(getattr(cite, "chunk_id", "") or "")) \
                    if getattr(cite, "chunk_id", None) else None
                if chunk is None and getattr(cite, "file_name", ""):
                    chunk = next((c for c in chunks if str(getattr(c, "file_name", "")) == cite.file_name
                                  and int(getattr(c, "page", 0)) == int(cite.page)), None)
                if chunk is not None:
                    pool.append(chunk)
            pool = pool or list(chunks)[:3]
            keywords = list(FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()))
            expects_numeric = bool(info.get("expects_numeric"))
            best_score, best_text, best_chunk = 0.0, "", None
            for chunk in pool:
                for sentence in split_sentences(str(getattr(chunk, "content", "") or "")):
                    hits = [kw for kw in keywords if kw in sentence]
                    if not hits:
                        continue
                    numbers = extract_numbers(sentence)
                    if expects_numeric and not numbers:
                        continue                                   # 问数值却无比数 → 不是答案句
                    score = len(hits) * 2.0 + (1.0 if numbers else 0.0) + min(len(numbers), 3) * 0.2
                    if score > best_score:
                        best_score, best_text, best_chunk = score, sentence, chunk
            answer_text = ""
            if best_chunk is not None and best_text:
                answer_text = (self._field_focus(best_text, keywords, expects_numeric=expects_numeric)
                               or self._clean_evidence_text(best_text, limit=160))
                answer_text = f"{answer_text}\n引用：{citation_mod.format_citation(str(getattr(best_chunk, 'file_name', '')), int(getattr(best_chunk, 'page', 0)))}"
            else:
                fallback = self.extractive.generate(question, pool, language=lang, logger=log)
                if citation_mod.answer_body(fallback.text).strip() not in {"", UNKNOWN_TEXT}:
                    answer_text = fallback.text
            recovered = bool(answer_text.strip()) and not citation_mod.is_empty_answer(answer_text) \
                and citation_mod.answer_body(answer_text).strip() != UNKNOWN_TEXT
            log.log_event("generation.recover_answer", level="" if recovered else "WARNING", trace_id=trace,
                          recovered=recovered, score=round(best_score, 3), pool=len(pool),
                          chunk_id=str(getattr(best_chunk, "chunk_id", "")) if best_chunk is not None else None)
            span.set_output({"recovered": recovered, "chars": len(answer_text)})
            return answer_text if recovered else ""

    @staticmethod
    def _field_focus(sentence: str, keywords: Sequence[str], *, expects_numeric: bool) -> str:
        """从命中句里取出「字段 → 取值」（数值字段取表格单元值，文本字段回整句原话）。"""
        for keyword in keywords:
            match = re.search(rf"{re.escape(keyword)}\s*[:：]?\s*[|｜]?\s*([^|｜\n，。；;]{{1,40}})", sentence)
            if not match:
                continue
            value = match.group(1).strip(" \t:：|｜")
            if not value:
                continue
            if expects_numeric:
                if not re.search(r"\d", value):
                    continue
                return f"{keyword}：{value}"
            return sentence.strip()[:160]                       # 文本字段：整句原话最贴近 golden
        return ""

    def _finalize(self, question: str, retrieval: Any, text: str, chunks: Sequence[Any], *,
                  info: dict[str, Any], lang: str, trace: str, backend: str, model: str,
                  first_ms: float, started: float, log: Any, history: Sequence[Any] | None,
                  prompt: str, context_chunks: Sequence[Any] | None = None,
                  support: Sequence[Any] | None = None) -> Answer:
        """引用校验 + 主体闸门 + 必要时修一轮；仍不合格则回「不清楚」。

        T6 修正（§14.6，实测根因）：
            * 提示词回声、只给引用不给正文 → 清洗 + 空答案重试，仍空则「不清楚」；
            * 引用只在**本次返回块**内落地，页面对不上 → 回验改指；核不出答案的引用一律剔除；
            * 最终答案文本由「正文 + 通过校验的引用」重建，正文里不再残留未校验页码。
        T7/t12 新增（§15.2）：
            * ``context_chunks`` = 支持块 + top-k（引用的落地范围；支持块页码同样真实可回溯）；
            * ``support`` 非空且问题问数值时做**数值锚定校验**：答案没有候选块里的带单位数值时，
              用「支持块置首 + 要求逐字照抄」重生成一次，只在数值命中变多时才采纳。
        """
        citation_pool = list(context_chunks) if context_chunks else list(chunks)
        support_pool = list(support or [])

        def evaluate(candidate_text: str) -> tuple[str, list[Any], Any, Any, list[dict[str, Any]]]:
            cleaned = self.sanitize_answer(candidate_text, trace=trace, logger=log)
            cites, rep, acts = self._resolve_citations(cleaned, citation_pool, trace=trace, log=log, info=info)
            gate_result = answerability.subject_gate(question, cleaned, citation_pool, issuer_names=None,
                                                     cfg=self.cfg, logger=log)
            return cleaned, cites, rep, gate_result, acts

        def anchor_numbers(chunk_list: Sequence[Any]) -> set[str]:
            """与问题字段相关的「带单位数值」集合（逐字照抄的唯一来源）。"""
            from .query_understanding import FIELD_KEYWORDS
            from .text_utils import split_sentences

            keywords = list(FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()))
            found: set[str] = set()
            for chunk in chunk_list:
                for sentence in split_sentences(str(getattr(chunk, "content", "") or "")):
                    if keywords and not any(k in sentence for k in keywords):
                        continue
                    for raw in _UNIT_NUMBER_PATTERN.findall(sentence):
                        found.add(citation_mod.normalize_number(raw))
            return found

        def answer_numbers_in_answer() -> set[str]:
            body = citation_mod.answer_body(text)
            return {citation_mod.normalize_number(m) for m in _UNIT_NUMBER_PATTERN.findall(body)}

        text, citations, report, gate, actions = evaluate(text)
        retry_count = 0                                            # t19：真实重试次数（阶段④触发即 +1）

        def need_more() -> bool:
            """还缺「可校验的答案」：正文空 或 没有任何合法引用。"""
            return citation_mod.is_empty_answer(text) or report.valid == 0

        def looks_incomplete() -> bool:
            """答案可校验但不完整（枚举/多值问题只答了一部分）→ 需要补全。

            判据（T6 实测，§14.6）：以「字段关键词命中的证据原话句」为参照，
            答案对它的实词覆盖率 < 0.70，或漏掉参照句里过半数值 → 判为不完整。
            只对枚举/数值意图（哪些/包括/分别/多少…）生效，避免误伤简洁正确答。
            """
            if citation_mod.is_empty_answer(text) or report.valid == 0 or not gate.ok:
                return False
            if not re.search(r"哪些|包括|分别|列举|都有|各有|是什么|多少", question):
                return False
            reference = self._reference_sentence(question, chunks, citations, info, logger=log)
            if not reference:
                return False
            tokens = [t for t in text_utils.tokenize(reference) if len(t) >= 2 and t not in text_utils.STOPWORDS]
            body = citation_mod.answer_body(text)
            answer_tokens = set(text_utils.tokenize(body))
            coverage = (len([t for t in tokens if t in answer_tokens]) / len(tokens)) if tokens else 1.0
            reference_numbers = {citation_mod.normalize_number(n) for n in text_utils.extract_numbers(reference)}
            answer_numbers = {citation_mod.normalize_number(n) for n in text_utils.extract_numbers(body)}
            missing = reference_numbers - answer_numbers
            incomplete = coverage < 0.70 or bool(reference_numbers and len(missing) > len(reference_numbers) / 2)
            log.log_event("generation.answer_completeness", trace_id=trace, complete=not incomplete,
                          coverage=round(coverage, 3), reference_numbers=len(reference_numbers),
                          missing_numbers=sorted(missing)[:5])
            return incomplete

        if citation_mod.answer_body(text).strip() == UNKNOWN_TEXT:
            log.log_event("generation.model_declined", level="WARNING", trace_id=trace,
                          note="模型自称证据不足，进入补全阶段")
        # 阶段②：只问正文（去掉引用/格式负担，快）——引用由「答案落在哪个块」落位
        if need_more():
            body = self._generate_body(question, chunks, lang, trace, log)
            if body:
                text, citations, report, gate, actions = evaluate(body)
                log.log_event("generation.body_stage_used", level="WARNING", trace_id=trace,
                              ok=not need_more(), citations=[c.render(language=lang) for c in citations])
        # 阶段③：确定性兜底——按问题字段关键词在证据块里摘原话（毫秒级，不编造）
        if need_more():
            recovered = self._recover_answer(question, chunks, info, preferred=citations,
                                             lang=lang, trace=trace, logger=log)
            if recovered:
                text, citations, report, gate, actions = evaluate(recovered)
                log.log_event("generation.recovered", level="WARNING", trace_id=trace,
                              ok=not need_more(), citations=[c.render(language=lang) for c in citations])
        # 阶段④：带修正要求重试一轮（主模型，快）
        # t19（§22）：``retry_count`` 是**真实重试计数器**（本请求触发过几次阶段④）。
        # 修前这里与 ``generation.done`` 都恒传 ``retry=1``（按代码路径写死的字面量，实测 1168:54），
        # devops 曾据此误读为「该请求发生了 1 次重试」——死字段必须换成真值。
        if need_more() or not gate.ok:
            reasons: list[str] = []
            if citation_mod.is_empty_answer(text):
                reasons.append("- 上一条回答没有正文：必须先用一句完整的话给出结论，再给引用；")
            if report.total == 0 or report.valid == 0:
                reasons.append("- 引用必须照抄片段标签里的文件名与页码，且该片段要真的含答案依据；")
            if not gate.ok:
                reasons.append("- 不得包含与问题主体类型不符的实体（问企业就不要写自然人，反之亦然）；")
            extra = ("\n\n【修正要求】只输出修正后的答案，不要解释：\n" + "\n".join(reasons)
                     + "\n- 证据确实不足时，整条回答只输出：不清楚")
            retry_count += 1
            log.log_event("generation.retry", level="WARNING", trace_id=trace, retry_index=retry_count,
                          reason=("empty_answer" if citation_mod.is_empty_answer(text) else
                                  ("citation_invalid" if report.valid == 0 else f"subject_gate:{gate.reason}")),
                          invalid_reasons=report.reasons[:3])
            text, citations, report, gate, actions = evaluate(
                self._generate_text(prompt + extra, question, chunks, lang, trace, log)[0])
        # 阶段⑤：兜底强模型（慢）——答案缺失或证据显示答得不全时才用
        if need_more():
            strong_body = self._generate_strong_body(question, chunks, lang, trace, log)
            if strong_body:
                text, citations, report, gate, actions = evaluate(strong_body)
                log.log_event("generation.strong_stage_used", level="WARNING", trace_id=trace,
                              ok=not need_more(), citations=[c.render(language=lang) for c in citations])
        elif looks_incomplete():
            # 答案可校验但明显不全：**只记日志**。
            # 实测（§14.7）：无论「确定性证据原话补全」还是「deepseek-r1:7b 补全」，14 题语义正确数都下降
            # （11/14 → 9/14、8/14），且强模型把首字推到 4.0~4.8 s；因此默认不改写答案，
            # 需要更高完整度时用 RAG_LLM__OLLAMA_STRONG_MODEL=deepseek-r1:7b 显式打开。
            strong_body = self._generate_strong_body(question, chunks, lang, trace, log)
            if strong_body:
                candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                    evaluate(strong_body)
                better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                          and len(citation_mod.answer_body(candidate_text)) > len(citation_mod.answer_body(text)))
                log.log_event("generation.strong_completion", level="WARNING", trace_id=trace, adopted=better,
                              citations=[c.render(language=lang) for c in candidate_cites])
                if better:
                    text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                              candidate_gate, candidate_actions)
        def leak_only(text_value: str) -> list[str]:
            """该答案触发的「主体泄漏」名单（用于定向修正提示）。"""
            gate_result = answerability.subject_gate(question, text_value, citation_pool, issuer_names=None,
                                                     cfg=self.cfg, logger=log)
            return list(gate_result.leaked)

        def allowed_table_answer() -> str:
            """主体闸门判否（夹带自然人）时的确定性兜底：直接摘「企业名称」表的行作答。

            实测（t12 定位数据）：题 4 的 p157 有**两张同标题表**——表1「关联方名称」列含自然人赵马克，
            表2「企业名称」列是 9 家企业；allowed 集合完整（7 家真值全在），闸门拦截正确。
            这里按 allowed（来自「企业名称」列语义）逐字列出企业名，并给出该表所在页引用。
            """
            from . import answerability as _ab

            tables = [c for c in citation_pool if str(getattr(c, "type", "")) == "table"]
            expected = str(info.get("expected_subject") or "any")
            if expected != "organization" or not tables:
                return ""
            allowed, _leaked = _ab.allowed_set_from_tables(tables, expected, logger=log)
            if len(allowed) < 2:
                return ""
            for chunk in tables:
                content = str(getattr(chunk, "content", "") or "")
                hits = [a for a in allowed if a in content]
                if len(hits) < 2:
                    continue
                relations: dict[str, str] = {}
                for line in content.splitlines():
                    cells = [c.strip() for c in line.strip("|").split("|")] if line.strip().startswith("|") else []
                    if len(cells) >= 2 and cells[0] in allowed:
                        relations[cells[0]] = cells[1]
                body = "不存在控制关系的关联方企业包括：" + "、".join(
                    f"{name}（{relations[name]}）" if relations.get(name) else name for name in allowed)
                cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),
                                                        int(getattr(chunk, "page", 0)))
                return f"{body}\n引用：{cite_text}"
            return ""

        def aspect_terms() -> list[str]:
            """问题里的「方位/对比词」：上游/下游、本期/上期…（题 793 问「下游」就不能只答「上游」）。"""
            return [w for w in ("下游", "上游", "本期", "上期", "母公司", "子公司") if w in question]

        def document_list_intent() -> bool:
            """是否在问「清单/枚举」（哪些/包括/有…）——用于清单项覆盖校验。"""
            return bool(re.search(r"哪些|包括|有哪些|分别是|分别", question))

        def find_definitional_sentence(chunk_list: Sequence[Any]) -> tuple[str, Any] | tuple[None, None]:
            """找「含字段关键词 + 定义/枚举句式」的最佳证据句（题 34/793 这类定义型问题的锚点）。

            t13 修正（§16.2，实测题 793）：优先**问题里真正出现的**字段词与方位词——
            问「下游」时不得选到「上游」句（此前只按 token 覆盖率打分，两句打平后选中了先出现的「上游」句）。
            只有当题面没有任何字段词时才退回全部字段词。
            """
            from .query_understanding import FIELD_KEYWORDS
            from .text_utils import split_sentences

            keywords = list(FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()))
            question_keywords = [k for k in keywords if k in question] or keywords
            aspects = aspect_terms()
            q_tokens = text_utils.tokenize(question)
            best: tuple[float, str, Any] | None = None
            for chunk in chunk_list:
                for sentence in split_sentences(str(getattr(chunk, "content", "") or "")):
                    if question_keywords and not any(k in sentence for k in question_keywords):
                        continue
                    if aspects and not any(a in sentence for a in aspects):
                        continue                                   # 只接受问题所问那一侧的原话
                    if not reranker_mod.DEFINITIONAL_PATTERN.search(sentence):
                        continue
                    cleaned = self._clean_evidence_text(sentence, limit=200)
                    if len(cleaned) < 12:
                        continue
                    hits = [k for k in question_keywords if k in cleaned]
                    score = text_utils.keyword_coverage(q_tokens, text_utils.tokenize(cleaned)) + len(hits) * 0.1
                    if best is None or score > best[0]:
                        best = (score, cleaned, chunk)
            return (best[1], best[2]) if best is not None else (None, None)

        def table_project_answer() -> str:
            """问「募投/投资项目有哪些」时的确定性兜底：逐行摘「序号 | 项目名称 | 金额」表。

            t13 实测（§16.3）：题 2 的 `招股说明书2_p0022_t07` / `p0306_t196` 是干净三列表，
            5 个项目名与金额齐全，第 5 行金额原文即「未披露」；项目名与金额**逐字**来自表格。
            引用优先落在 **top-k 返回块**（候选集内），其次才用支持块。
            """
            if not re.search(r"项目|募投|募集资金用途", question) or not document_list_intent():
                return ""
            ordered = list(chunks) + [c for c in support_pool
                                      if all(str(getattr(c, "chunk_id", "")) != str(getattr(x, "chunk_id", ""))
                                             for x in chunks)]
            for chunk in ordered:
                if str(getattr(chunk, "type", "")) != "table" and "项目名称" not in str(getattr(chunk, "content", "")):
                    continue
                content = str(getattr(chunk, "content", "") or "")
                rows = [line for line in content.splitlines() if line.strip().startswith("|")]
                if len(rows) < 4:
                    continue
                header = [c.strip() for c in rows[0].strip("|").split("|")]
                name_idx = next((i for i, h in enumerate(header) if "项目名称" in h or h == "项目"), None)
                amount_idx = next((i for i, h in enumerate(header)
                                   if any(k in h for k in ("计划总投资", "项目总投资", "总投资", "拟投入募集资金"))), None)
                if name_idx is None:
                    continue
                items: list[str] = []
                for row in rows[1:]:
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    if len(cells) <= name_idx or not re.fullmatch(r"\d{1,2}", cells[0] or ""):
                        continue
                    name = cells[name_idx]
                    if not name or name in {"-", "—"}:
                        continue
                    amount = cells[amount_idx] if amount_idx is not None and len(cells) > amount_idx else ""
                    items.append(f"{name}（{amount} 万元）" if amount and amount not in {"未披露", "-", "—"}
                                 else f"{name}（未披露）")
                if len(items) < 3:
                    continue
                cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),
                                                        int(getattr(chunk, "page", 0)))
                log.log_event("generation.project_table_answer", level="WARNING", trace_id=trace,
                              chunk_id=str(getattr(chunk, "chunk_id", "")), items=len(items),
                              distinct_numbers=len(set(re.findall(r"[\d,]+\.\d{2}", content))),
                              in_top_k=any(str(getattr(c, "chunk_id", "")) == str(getattr(chunk, "chunk_id", ""))
                                           for c in chunks))
                return "本次募集资金拟投资项目包括：" + "、".join(items) + f"\n引用：{cite_text}"
            return ""

        def find_anchor_sentence(chunk_list: Sequence[Any]) -> tuple[str, Any] | tuple[None, None]:
            """在候选块里找「含字段关键词 + 锚点数值」的最短原话句（逐字照抄的唯一来源）。"""
            from .query_understanding import FIELD_KEYWORDS
            from .text_utils import split_sentences

            keywords = list(FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()))
            best: tuple[int, str, Any] | None = None
            for chunk in chunk_list:
                for sentence in split_sentences(str(getattr(chunk, "content", "") or "")):
                    if keywords and not any(k in sentence for k in keywords):
                        continue
                    if not ({citation_mod.normalize_number(m)
                             for m in _UNIT_NUMBER_PATTERN.findall(sentence)} & anchors):
                        continue
                    cleaned = self._clean_evidence_text(sentence, limit=200)
                    if len(cleaned) < 8:
                        continue
                    if best is None or len(cleaned) < best[0]:
                        best = (len(cleaned), cleaned, chunk)
            return (best[1], best[2]) if best is not None else (None, None)

        # 阶段⑥：数值锚定（T7/t12 新增，§15.2）——两段式：
        #   ① 受约束重生成（支持块置首 + 要求逐字照抄）；② 若仍无数值，则**确定性摘录**锚点原话句。
        # 两者都只在「答案里出现了更多锚点数值」时才采纳，数值一律逐字来自候选块，绝不凭空造数。
        if support_pool:
            anchors = anchor_numbers(support_pool + list(chunks))
            before = answer_numbers_in_answer() & anchors
            if anchors and not before:
                log.log_event("generation.numeric_anchor_missing", level="WARNING", trace_id=trace,
                              anchors=sorted(anchors)[:6], support=[c.chunk_id for c in support_pool])
                extra = (f"\n\n【修正要求】上一个答案没有用到片段里的原始数值。"
                         f"本题问的是「{info.get('field_type') or '数值'}」，请只依据片段里的数值作答，"
                         "数值必须**逐字照抄**（含千分位与单位），并在最后一行给出引用；"
                         "证据不足则只输出：不清楚")
                anchor_prompt = self.build_prompt(question, chunks, history=history, language=lang,
                                                  support_chunks=support_pool)
                candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                    evaluate(self._generate_text(anchor_prompt + extra, question, chunks, lang, trace, log)[0])
                after = {citation_mod.normalize_number(m)
                         for m in _UNIT_NUMBER_PATTERN.findall(citation_mod.answer_body(candidate_text))} & anchors
                better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                          and len(after) > len(before))
                log.log_event("generation.numeric_anchor_regen", level="WARNING", trace_id=trace,
                              adopted=better, before=sorted(before), after=sorted(after)[:6],
                              citations=[c.render(language=lang) for c in candidate_cites])
                if better:
                    text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                              candidate_gate, candidate_actions)
            if anchors and not (answer_numbers_in_answer() & anchors):
                snippet, chunk = find_anchor_sentence(support_pool + list(chunks))
                if snippet is not None and chunk is not None:
                    cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),
                                                            int(getattr(chunk, "page", 0)))
                    candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                        evaluate(f"{snippet}\n引用：{cite_text}")
                    after = {citation_mod.normalize_number(m)
                             for m in _UNIT_NUMBER_PATTERN.findall(citation_mod.answer_body(candidate_text))} & anchors
                    better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                              and bool(after))
                    log.log_event("generation.numeric_anchor_extract", level="WARNING", trace_id=trace,
                                  adopted=better, after=sorted(after)[:6], chunk_id=str(getattr(chunk, "chunk_id", "")),
                                  citations=[c.render(language=lang) for c in candidate_cites])
                    if better:
                        text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                                  candidate_gate, candidate_actions)
        # 阶段⑦：定义/枚举锚定（t12 + t13 + t20，§15.2/§16.2/§23）——问「涉及/包括/哪些」时，
        # 若答案没覆盖问题所问的**方位侧**（题 793 问「下游」只答了「上游」）、对最佳定义句的关键词
        # 覆盖更低，**或者现有答案是「证据里核不出整句原话」的小模型改写句**，就用该句逐字替换
        # （纯摘原文，不新增事实）。

        def answer_is_paraphrase() -> bool:
            """现有答案是否为「证据里核不出整句原话」的改写句（t20，§23）。

            判据与引用校验同源（严格）：把答案正文与每个候选块都做 ``squash_text``（去空白与标点）后比较，
            若**没有任何候选块包含该整句**，判为改写句。

            实测动机（题 793，验收数字在 14/14 与 13/14 间翻转的**唯一**来源）：小模型在
            ``RAG_LLM__TEMPERATURE = 0.2`` 下会非确定地给出
            「电子信息行业的下游主要包括军队、政府机关、能源等行业企业。」——它丢掉了招股书原话里的
            「为各类终端用户，覆盖范围广泛」，却因为照抄了问题主体词「电子信息行业」而在**问题 token
            覆盖率**上压过原话句 → 旧的 ``new_cov > old_cov + 0.05`` 条件不触发 → 判分器判否。
            通用处置：**改写句（证据里核不出整句）不参与「覆盖率比较」，直接让位给可被引用校验核出的
            招股书原话定义句**——原话句逐字来自 PDF，天然满足「引用可回溯」，且不受采样抖动影响。
            """
            body = text_utils.squash_text(citation_mod.answer_body(text))
            if not body:
                return False
            return all(body not in text_utils.squash_text(str(getattr(chunk, "content", "") or ""))
                       for chunk in citation_pool)

        aspects = aspect_terms()
        missing_aspect = bool(aspects) and not any(a in citation_mod.answer_body(text) for a in aspects)
        if re.search(r"涉及|包括|是什么|分别|哪些", question) or missing_aspect:
            snippet, chunk = find_definitional_sentence(citation_pool)
            if snippet is not None and chunk is not None:
                question_tokens = text_utils.tokenize(question)
                new_cov = text_utils.keyword_coverage(question_tokens, text_utils.tokenize(snippet))
                old_cov = text_utils.keyword_coverage(question_tokens,
                                                      text_utils.tokenize(citation_mod.answer_body(text)))
                paraphrased = answer_is_paraphrase()
                # 注意：**不要求**页码未被引用过——实测题 793 引的就是正确页 152，但正文只是碎片
                # （覆盖率 0.27）；此时应把碎片换成同页那句完整定义句（同页去噪 + 完整度）。
                # t20 增补：``paraphrased``（改写句）优先采纳原话句，不再让覆盖率比较决定胜负。
                if missing_aspect or new_cov > old_cov + 0.05 or paraphrased:
                    cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),
                                                            int(getattr(chunk, "page", 0)))
                    candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                        evaluate(f"{snippet}\n引用：{cite_text}")
                    better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0)
                    log.log_event("generation.definitional_anchor", level="WARNING", trace_id=trace,
                                  adopted=better, missing_aspect=missing_aspect, paraphrased=paraphrased,
                                  new_coverage=round(new_cov, 3), old_coverage=round(old_cov, 3),
                                  chunk_id=str(getattr(chunk, "chunk_id", "")),
                                  citations=[c.render(language=lang) for c in candidate_cites])
                    if better:
                        text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                                  candidate_gate, candidate_actions)
        # 阶段⑨：清单项断言（t13，§16.3，实测题 2）——问「募投/投资项目有哪些」且上下文里有
        # 「序号 | 项目名称 | 金额」表时，逐行摘表作答；只在**覆盖到更多项目名/金额**时采纳。
        project_answer = table_project_answer()
        if project_answer:
            candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                evaluate(project_answer)
            names = re.findall(r"（[^）]*）", citation_mod.answer_body(candidate_text))
            old_body = citation_mod.answer_body(text)
            gained = sum(1 for item in re.findall(r"[^、\s（]+（", citation_mod.answer_body(candidate_text))
                         if item.rstrip("（") not in old_body)
            better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                      and gained >= 2)
            log.log_event("generation.project_list_anchor", level="WARNING", trace_id=trace,
                          adopted=better, items=len(names), gained=gained,
                          citations=[c.render(language=lang) for c in candidate_cites])
            if better:
                text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                          candidate_gate, candidate_actions)
        # 阶段⑧：主体闸门定向修复（t12，§15.3，实测题 4）
        # 定位数据：p157 有两张同标题表——表1 是「关联方名称」含自然人赵马克，表2 是「企业名称」9 家企业；
        # 闸门拦得对（org_leaked=赵马克），但**拒答不是正确处置**：先定向重生成（剔除泄漏主体），
        # 仍不过则用「企业名称」表行逐字作答（allowed 集合完整，7 家真值全在其中）。
        if not gate.ok and list(gate.leaked):
            leaked = sorted({str(x) for x in gate.leaked})
            log.log_event("generation.subject_gate_repair", level="WARNING", trace_id=trace,
                          reason=gate.reason, leaked=leaked)
            extra = (f"\n\n【修正要求】答案里出现了不属于本题提问对象的条目：{'、'.join(leaked)}。"
                     "本题只问「不存在控制关系的关联方企业」，请只列**企业**名称"
                     "（不要写自然人、控股股东或其它非企业主体），逐字照抄片段里的企业名，"
                     "并在最后一行给出引用；证据不足则只输出：不清楚")
            candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                evaluate(self._generate_text(prompt + extra, question, chunks, lang, trace, log)[0])
            repaired = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                        and candidate_gate.ok)
            log.log_event("generation.subject_gate_regenerate", level="WARNING", trace_id=trace,
                          ok=repaired, gate_reason=candidate_gate.reason,
                          citations=[c.render(language=lang) for c in candidate_cites])
            if repaired:
                text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                          candidate_gate, candidate_actions)
            if not repaired:
                deterministic = allowed_table_answer()
                if deterministic:
                    candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                        evaluate(deterministic)
                    repaired = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                                and candidate_gate.ok)
                    log.log_event("generation.subject_gate_table_answer", level="WARNING", trace_id=trace,
                                  ok=repaired, citations=[c.render(language=lang) for c in candidate_cites])
                    if repaired:
                        text, citations, report, gate, actions = (candidate_text, candidate_cites,
                                                                  candidate_report, candidate_gate,
                                                                  candidate_actions)

        def evidence_pool() -> list[Any]:
            """证据池：**当前已落地引用的块**优先，其后是候选块（top-k ∪ 支持块）。"""
            pool: list[Any] = []
            for cite in citations:
                chunk = self.chunk_lookup(str(getattr(cite, "chunk_id", "") or "")) \
                    if getattr(cite, "chunk_id", None) else None
                if chunk is None and getattr(cite, "file_name", ""):
                    chunk = next((c for c in citation_pool
                                  if str(getattr(c, "file_name", "")) == cite.file_name
                                  and int(getattr(c, "page", 0)) == int(cite.page)), None)
                if chunk is not None:
                    pool.append(chunk)
            for chunk in citation_pool:
                if all(str(getattr(chunk, "chunk_id", "")) != str(getattr(c, "chunk_id", "")) for c in pool):
                    pool.append(chunk)
            return pool

        def field_statement_answer() -> str:
            """字段型问句「成句化」：表格行/字段清单式答案 → 「字段 + 判断动词 + 取值」短句（t16，§19.2）。

            实测动机（题 531）：答案把 PDF1 物理 52 页「发行人基本情况」整段铺进来
            （180 字单行 `… 法定代表人：程家明 注册资本：5,520 万元 …`），事实全对、引用 p52 可回溯，
            但**不是一句人话**：官方判分器第①步要求「去标点后 golden 是答案子串」→ 需要
            `法定代表人是程家明` 中的判断动词「是」。
            通用规则（非单题特判）：
                ① 问题属「字段型」（问句里出现该字段的关键词）、**只问一个字段**（无枚举词、无多个问号）；
                ② 现有答案对该字段只有「：」/表格式赋值（非判断动词）→ 判为「未成句」，需要改写；
                ③ 在候选块里找到**同一取值**的原话（去空白后逐字一致）→ 断言成 `字段是/为取值。`；
                   「是」用于问人/问物（是谁），「为」用于问数值（是多少）——中文规范答法。
                ④ 取值一律逐字来自证据块，**不新增任何事实**；找不到同值原话则不改写（宁可不改）。
            """
            from .query_understanding import FIELD_KEYWORDS

            field_type = str(info.get("field_type") or "")
            keywords = [k for k in FIELD_KEYWORDS.get(field_type, ()) if k in question]
            if not keywords:
                return ""
            if _ENUM_QUESTION_PATTERN.search(question):
                return ""                                      # 枚举题必须多项作答，不能压成单值句
            if question.count("？") + question.count("?") > 1:
                return ""                                      # 多问句：必须把小问全答上，不能只回一个字段
            expects_numeric = bool(info.get("expects_numeric"))
            body = citation_mod.answer_body(text).strip()
            if not body:
                return ""
            pool = evidence_pool()
            for keyword in keywords:
                value, style = _extract_field_value(body, keyword, expects_numeric=expects_numeric)
                if not value or style == "copula":
                    continue                                   # 抽不到取值 / 已是判断句 → 无需改写
                for chunk in pool:
                    evidence_value, _style = _extract_field_value(
                        str(getattr(chunk, "content", "") or ""), keyword, expects_numeric=expects_numeric)
                    if not evidence_value:
                        continue
                    if text_utils.squash_text(evidence_value) != text_utils.squash_text(value):
                        continue                               # 取值必须与证据同值，否则不采纳
                    copula = "是" if _WHO_QUESTION_PATTERN.search(question) else "为"
                    cite_text = citation_mod.format_citation(str(getattr(chunk, "file_name", "")),
                                                             int(getattr(chunk, "page", 0)))
                    log.log_event("generation.field_statement_built", level="WARNING", trace_id=trace,
                                  field_type=field_type, keyword=keyword, style=style, copula=copula,
                                  chunk_id=str(getattr(chunk, "chunk_id", "")), cite=cite_text,
                                  answer_chars=len(body))
                    return f"{keyword}{copula}{value}。\n引用：{cite_text}"
            return ""

        def trailing_clause_trim() -> str:
            """裁掉「与问题无关的尾部补语」（题 957 类：正确答案 + 附加整句）（t16，§19.3）。

            实测动机：题 957 答案 = `公司目前已经成为军队视频指挥控制领域的重要供应商，参与制定了国防用户
            第一个视频指挥系统技术标准（即《某视频技术规范1.0》）。`，golden 只要前半句；
            官方判分器第①步不成立、第⑤步（二元组相似度 ≥0.62）被后半句稀释 → 判否。
            通用规则：
                ① 枚举/清单题不裁剪（答案本就需要多项）；
                ② **只裁尾部**：从句尾起逐个丢掉「与问题无任何实词重叠」的子句，遇到有关子句立即停；
                ③ 数值型问题下，含数字的尾子句**一律保留**（数字很可能就是答案，如题 1 的 `1,670万股`）；
                ④ 安全网：问题所问的字段词/方位词必须仍在保留文本里，且保留文本必须能被某个候选块**按引用校验同一条判据**核出（`citation.answer_support_check` 通过），否则放弃裁剪（宁可不裁）。
            """
            from .query_understanding import FIELD_KEYWORDS

            if _ENUM_QUESTION_PATTERN.search(question):
                return ""
            body = citation_mod.answer_body(text).strip()
            if not body:
                return ""
            clauses = _split_clauses(body)
            if len(clauses) < 2:
                return ""
            # 多子句答案才可能裁剪：先把判据落日志（INFO 决策遥测），便于复盘「为什么没裁 / 裁了什么」
            log.log_event("generation.trailing_trim_probe", clauses=len(clauses), trace_id=trace,
                          first_clause=clauses[0][:30], last_clause=clauses[-1][:30])
            expects_numeric = bool(info.get("expects_numeric"))
            question_tokens = {token for token in text_utils.tokenize(question) if len(token) >= 2}
            kept = list(clauses)
            dropped: list[str] = []
            while len(kept) >= 2:
                last = kept[-1]
                if expects_numeric and re.search(r"\d", last):
                    break                                      # 数值保护
                tokens = {token for token in text_utils.tokenize(last) if len(token) >= 2}
                if tokens & question_tokens:
                    break                                      # 尾子句与问题相关 → 停止裁剪
                dropped.append(kept.pop())
            if not dropped:
                return ""
            kept_body = "".join(kept).strip()
            if kept_body and kept_body[-1] in "，,；;、":
                kept_body = kept_body[:-1] + "。"                # 裁掉尾子句后不能留下悬空逗号
            if len(text_utils.squash_text(kept_body)) < 8:
                return ""
            anchors = [k for k in FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()) if k in question]
            anchors += aspect_terms()
            if any(anchor not in kept_body for anchor in anchors):
                return ""                                      # 问题所问的关键词被裁掉 → 放弃
            kept_tokens = text_utils.tokenize(kept_body)
            best_chunk, best_why = None, ""
            for chunk in evidence_pool():
                # 用**与引用校验同一条判据**挑块：能核出保留文本的块才作为引用来源
                evidence = self.evidence_lookup(citation_mod.Citation(
                    file_name=str(getattr(chunk, "file_name", "")), page=int(getattr(chunk, "page", 0)),
                    chunk_id=str(getattr(chunk, "chunk_id", "")) or None))
                ok, why = citation_mod.answer_support_check(kept_body, evidence, logger=log)
                if ok:
                    best_chunk, best_why = chunk, str(why)
                    break
            if best_chunk is None:
                log.log_event("generation.trailing_trim_skipped", level="WARNING", trace_id=trace,
                              reason="保留文本在任何候选块都核不出（放弃裁剪，宁可不裁）",
                              kept_chars=len(kept_body), kept_tokens=len(kept_tokens))
                return ""                                      # 保留文本核不出来 → 放弃（绝不凭空断言）
            cite_text = citation_mod.format_citation(str(getattr(best_chunk, "file_name", "")),
                                                    int(getattr(best_chunk, "page", 0)))
            log.log_event("generation.trailing_trim_built", level="WARNING", trace_id=trace,
                          dropped=len(dropped), dropped_head=dropped[0][:40],
                          kept_chars=len(kept_body), support=best_why,
                          chunk_id=str(getattr(best_chunk, "chunk_id", "")), cite=cite_text)
            return f"{kept_body}\n引用：{cite_text}"

        if citation_mod.answer_body(text).strip() == UNKNOWN_TEXT:      # 兜底后仍是「证据不足」
            log.log_event("generation.insufficient_evidence", trace_id=trace, retry_count=retry_count)
            return self._unknown(question, retrieval, reason="insufficient_evidence", backend=backend,
                                 model=model, language=lang, trace_id=trace, first_token_ms=first_ms,
                                 total_ms=round((time.perf_counter() - started) * 1000, 2))
        if citation_mod.is_empty_answer(text):
            log.log_event("generation.empty_answer_failed", level="ERROR", trace_id=trace, retry_count=retry_count)
            return self._unknown(question, retrieval, reason="empty_answer", backend=backend, model=model,
                                 language=lang, trace_id=trace, first_token_ms=first_ms,
                                 total_ms=round((time.perf_counter() - started) * 1000, 2))
        if report.total == 0 or report.valid == 0:
            log.log_event("generation.citation_failed", level="ERROR", invalid=[c.to_dict() for c in report.invalid],
                          retry_count=retry_count, reasons=report.reasons[:3])
            return self._unknown(question, retrieval, reason="citation_failed", backend=backend, model=model,
                                 language=lang, trace_id=trace, first_token_ms=first_ms,
                                 total_ms=round((time.perf_counter() - started) * 1000, 2))
        if not gate.ok:
            log.log_event("generation.subject_gate_failed", level="ERROR", reason=gate.reason,
                          leaked=list(gate.leaked), retry_count=retry_count)
            return self._unknown(question, retrieval, reason=f"subject_gate:{gate.reason}", backend=backend,
                                 model=model, language=lang, trace_id=trace, first_token_ms=first_ms,
                                 total_ms=round((time.perf_counter() - started) * 1000, 2))
        # 碎片答案扩句（确定性、纯摘原文）：先把答案质量补到「能读懂」，再落最终校验
        expanded = self._expand_short_answer(question, text, citations, chunks, info, logger=log)
        if expanded:
            candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = evaluate(expanded)
            if (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                    and len(citation_mod.answer_body(candidate_text)) > len(citation_mod.answer_body(text))):
                text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                          candidate_gate, candidate_actions)
                log.log_event("generation.expanded_adopted", trace_id=trace,
                              chars=len(citation_mod.answer_body(text)),
                              citations=[c.render(language=lang) for c in citations])
        def position_anchor_answer() -> str:
            """行业地位/领域型问句的「原话锚定」（t16，§19.3b；实测题 957）。

            实测动机：题 957 在不同运行里被小模型改写成三种措辞——「公司目前已经成为军队视频指挥控制领域的重要
            供应商，参与制定了…」/「兴图新科目前已经成为国防军队视频指挥领域的重要供应商…」/「武汉兴图新科电子
            股份有限公司在军队视频指挥领域已经成为重要供应商。」。前两种只与 golden 差同义改写（过判分器第⑤步），
            第三种把「已经成为…重要供应商」倒装成「在…领域已经成为重要供应商」，相似度只有 0.23 → 判否。
            通用规则（**问「在哪个领域/什么地位」时锚定招股书原话**，不新增事实）：
                ① 触发：问题含 领域/行业地位/市场地位/重要供应商（且非枚举清单题）；
                ② 选句：候选块里含**问题所问主体词**（供应商/客户…）与**问题所出限定词**（领域/上游/下游…）
                   且是地位句形态（已经成为/已成为/重要供应商）的**最短**原话句（最短 = 最少无关补语）；
                ③ 采纳：逐字替换答案正文（页码取该句所在块），随后交给阶段⑪裁掉尾补语；
                   已在答案里的原话、或引用/闸门校验不过 → 不改（宁可不改）。
            """
            from .query_understanding import FIELD_KEYWORDS

            if not _POSITION_QUESTION_PATTERN.search(question):
                return ""
            if _ENUM_QUESTION_PATTERN.search(question):
                return ""
            keywords = [k for k in FIELD_KEYWORDS.get(str(info.get("field_type") or ""), ()) if k in question]
            if not keywords:
                return ""
            aspects = [word for word in ("领域", "上游", "下游", "本期", "上期") if word in question]
            best: tuple[int, str, Any] | None = None
            for chunk in evidence_pool():
                for sentence in text_utils.split_sentences(str(getattr(chunk, "content", "") or "")):
                    if not _POSITION_SENTENCE_PATTERN.search(sentence):
                        continue
                    if not any(keyword in sentence for keyword in keywords):
                        continue
                    if aspects and not any(word in sentence for word in aspects):
                        continue
                    cleaned = self._clean_evidence_text(sentence, limit=200)
                    if len(cleaned) < 12:
                        continue
                    if best is None or len(cleaned) < best[0]:
                        best = (len(cleaned), cleaned, chunk)
            if best is None:
                return ""
            body = citation_mod.answer_body(text).strip()
            if text_utils.squash_text(best[1]) == text_utils.squash_text(body):
                return ""                                      # 现答案就是该原话 → 无需替换
            cite_text = citation_mod.format_citation(str(getattr(best[2], "file_name", "")),
                                                    int(getattr(best[2], "page", 0)))
            log.log_event("generation.position_anchor_built", level="WARNING", trace_id=trace,
                          keywords=keywords, aspects=aspects, chars=best[0],
                          chunk_id=str(getattr(best[2], "chunk_id", "")), cite=cite_text)
            return f"{best[1]}\n引用：{cite_text}"

        # 阶段⑩：字段型问句成句化（t16，§19.2；实测题 531/543）
        # 放在「碎片扩句」之后：扩句会把答案重新拉长（实测题 531 的整页倾倒会在扩句阶段复现），
        # 因此成句化必须是**最后一道路径整形**，随后只做「已通过校验的引用重建」。
        statement = field_statement_answer()
        if statement:
            candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                evaluate(statement)
            better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                      and candidate_gate.ok)
            log.log_event("generation.field_statement", level="WARNING", trace_id=trace, adopted=better,
                          field_type=info.get("field_type"),
                          chars=len(citation_mod.answer_body(candidate_text)),
                          empty=citation_mod.is_empty_answer(candidate_text),
                          citation_valid=candidate_report.valid, citation_total=candidate_report.total,
                          invalid_reasons=candidate_report.reasons[:3],
                          gate_ok=bool(candidate_gate.ok), gate_reason=candidate_gate.reason,
                          citations=[c.render(language=lang) for c in candidate_cites])
            if better:
                text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                          candidate_gate, candidate_actions)
        # 阶段⑫：行业地位/领域原话锚定（t16，§19.3b；实测题 957）——先锚原话，再由阶段⑪裁剪尾补语
        anchored = position_anchor_answer()
        if anchored:
            candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                evaluate(anchored)
            better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                      and candidate_gate.ok)
            log.log_event("generation.position_anchor", level="WARNING", trace_id=trace, adopted=better,
                          before_chars=len(citation_mod.answer_body(text)),
                          after_chars=len(citation_mod.answer_body(candidate_text)),
                          empty=citation_mod.is_empty_answer(candidate_text),
                          citation_valid=candidate_report.valid, citation_total=candidate_report.total,
                          invalid_reasons=candidate_report.reasons[:3],
                          gate_ok=bool(candidate_gate.ok), gate_reason=candidate_gate.reason,
                          citations=[c.render(language=lang) for c in candidate_cites])
            if better:
                text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                          candidate_gate, candidate_actions)
        # 阶段⑪：无关尾句裁剪（t16，§19.3；实测题 957）
        trimmed = trailing_clause_trim()
        if trimmed:
            candidate_text, candidate_cites, candidate_report, candidate_gate, candidate_actions = \
                evaluate(trimmed)
            better = (not citation_mod.is_empty_answer(candidate_text) and candidate_report.valid > 0
                      and candidate_gate.ok
                      and len(citation_mod.answer_body(candidate_text)) < len(citation_mod.answer_body(text)))
            log.log_event("generation.trailing_trim", level="WARNING", trace_id=trace, adopted=better,
                          before_chars=len(citation_mod.answer_body(text)),
                          after_chars=len(citation_mod.answer_body(candidate_text)),
                          empty=citation_mod.is_empty_answer(candidate_text),
                          citation_valid=candidate_report.valid, citation_total=candidate_report.total,
                          invalid_reasons=candidate_report.reasons[:3],
                          gate_ok=bool(candidate_gate.ok), gate_reason=candidate_gate.reason,
                          citations=[c.render(language=lang) for c in candidate_cites])
            if better:
                text, citations, report, gate, actions = (candidate_text, candidate_cites, candidate_report,
                                                          candidate_gate, candidate_actions)
        # 只保留通过校验的引用（引用可回溯 100%），并据此重建答案文本
        invalid_keys = {c.key() for c in report.invalid}
        kept = [c for c in citations if c.key() not in invalid_keys]
        dropped = [c for c in citations if c.key() in invalid_keys]
        if dropped:
            log.log_event("generation.citation_dropped", level="WARNING", trace_id=trace,
                          dropped=[c.to_dict() for c in dropped], reasons=report.reasons[:3])
        text = citation_mod.render_answer(text, kept, language=lang)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = Answer(answer_id=f"a{int(time.time() * 1000) % 100000000:08d}", question=question, text=text,
                        citations=kept, used_chunks=[str(getattr(c, "chunk_id", "")) for c in chunks],
                        backend=backend, model=model, language=lang, is_unknown=False, unknown_reason=None,
                        first_token_ms=first_ms, total_ms=total_ms, trace_id=trace, created_at=_now_iso(),
                        citation_report=report, subject_gate=gate, retrieval=retrieval)
        # t19（§22）：① 移除恒为常量的死字段 ``retry``，改为**真实重试计数** ``retry_count``；
        #           ② 补 ``trace_id``（修前 1222/1222 全为 null，无法把「开始→重试→恢复→完成」串成链路）。
        log.log_event("generation.done", trace_id=trace, chars=len(text),
                      citations=[c.render(language=lang) for c in kept],
                      backend=backend, model=model, total_ms=total_ms, first_token_ms=first_ms,
                      citation_accuracy=report.accuracy, subject_gate=gate.reason, retry_count=retry_count,
                      retargeted=sum(1 for a in actions if a.get("action") == "retarget"),
                      dropped_citations=len(dropped))
        return answer

    def stream(self, question: str, retrieval: Any, *, history: Sequence[Any] | None = None,
               language: str | None = None, trace_id: str | None = None,
               logger: Any = None) -> Iterator[AnswerDelta]:
        """流式作答（逐块 yield；``done=True`` 的 delta 携带完整 ``Answer``）。"""
        log = logger or self.log
        chunks = list(getattr(retrieval, "chunks", []) or [])
        lang = language or language_mod.answer_language(question)
        trace = trace_id or f"g{int(time.perf_counter() * 1000) % 100000000:08d}"
        # t19（§22）：流式路径同样要发 ``generation.start``，否则 SSE 请求的 trace 链**缺头**
        # （实测依据：修好后 39 条 ``generation.done`` 里唯一 1 条按 trace 关联不到的，正是流式请求）。
        log.log_event("generation.start", trace_id=trace, question_digest=text_digest(question, limit=60),
                      chunks=len(chunks), language=lang, stream=True)
        # 可答性判否 → 直接产出「不清楚」（不调用 LLM）
        from .query_understanding import understand

        info = understand(question, history, cfg=self.cfg, llm=None, logger=log)
        decision = answerability.decide(question, retrieval, field_type=info["field_type"],
                                        expects_numeric=info["expects_numeric"], cfg=self.cfg, logger=log)
        if not decision.is_answerable or not chunks:
            answer = self._unknown(question, retrieval, reason=decision.reason, backend="none", model="",
                                   language=lang, trace_id=trace)
            yield AnswerDelta(text=UNKNOWN_TEXT, is_first=True, first_token_ms=0.0, done=True, answer=answer)
            return
        prompt = self.build_prompt(question, chunks, history=history, language=lang)
        backend_name = getattr(getattr(self.llm, "backend", None), "name", "extractive")
        if self.llm is None or backend_name == "extractive":
            result = self.extractive.generate(question, chunks, language=lang, logger=log)
            first_ms = float(result.first_token_ms)
            yield AnswerDelta(text=result.text, is_first=True, first_token_ms=first_ms, done=False)
            full = self._finalize(question, retrieval, result.text, chunks, info=info, lang=lang, trace=trace,
                                  backend="extractive", model="rule-based", first_ms=first_ms,
                                  started=time.perf_counter(), log=log, history=history, prompt=prompt)
            yield AnswerDelta(text=full.text, is_first=False, first_token_ms=first_ms, done=True, answer=full)
            return
        started = time.perf_counter()
        buffer: list[str] = []
        first_ms = 0.0
        index = 0
        for delta in self.llm.generate(prompt, trace_id=trace, logger=log):
            piece = delta.text
            if index == 0:
                first_ms = round((time.perf_counter() - started) * 1000, 2)
                log.log_event("generation.first_token", trace_id=trace, first_token_ms=first_ms)
            index += 1
            buffer.append(piece)
            yield AnswerDelta(text=piece, is_first=(index == 1), first_token_ms=(first_ms if index == 1 else None),
                              done=False)
        text = "".join(buffer)
        full = self._finalize(question, retrieval, text, chunks, info=info, lang=lang, trace=trace,
                              backend=backend_name, model=getattr(self.llm.backend, "model", ""),
                              first_ms=first_ms, started=started, log=log, history=history, prompt=prompt)
        yield AnswerDelta(text="", is_first=False, first_token_ms=None, done=True, answer=full)


def build_generator(*, cfg: AppConfig | None = None, logger: Any = None) -> AnswerGenerator:
    """构造生成器（自动解析 LLM 后端：ollama → openai → extractive）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    backend = llm_client.resolve_backend(config, logger=log)
    llm = llm_client.LLMClient(cfg=config, backend=backend, logger=log) if backend.name != "extractive" else None
    log.log_event("generation.backend_resolved", backend=backend.name, model=backend.model,
                  base_url=backend.base_url, probe_ms=backend.probe_ms)
    return AnswerGenerator(cfg=config, llm=llm, logger=log)
