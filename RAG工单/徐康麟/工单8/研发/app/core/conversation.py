# -*- coding: utf-8 -*-
"""工单4 多轮会话（``session_id`` 维度）与指代消解辅助。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

设计依据：设计/接口设计.md §3.3（``session_id`` 多轮）与 §2.2（``--interactive`` 交互式 CLI）。

职责：
    * 记录每个会话的问答轮次（内存 + 可选落盘 ``SqliteManager``）；
    * **指代消解**：问句出现「它 / 其 / 该公司 / 这个项目 / 上述」等指代词且缺少主体时，
      把上一轮的问句主体拼到当前问句前面（**只补主体，不改写事实**，且拼接过的事实必须留痕）；
    * 会话轮数上限由 ``RAG_ANSWER__HISTORY_TURNS`` 控制。

纪律：本模块**不做任何答案生成**；它只决定「检索与生成看到的问题文本是什么」。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 指代词 / 省略主体标记（出现且问句本身没有主体名时触发上文补全）
#:
#: 工单7（人工智能NLP-RAG-功能测试及评估）扩充：本工单的多轮问法里，
#: 除了「它/该行/该公司」，高频的还有**口语化省略追问**：
#:     「那净利润呢？」「那么拨备覆盖率呢？」「营业收入呢？」
#: 它们既没有显式代词也没有主体名，原正则一律不匹配 ⇒ 指代链断裂
#: （实测回归用例 TC-M-A2 因此引到中国平安年报）。现补两条**位置受限**的标记：
#:   * 句首的「那 / 那么 / 还有 / 然后」——只在句首才算承接上文；
#:   * 句尾的「呢 / 如何 / 怎么样」——只在句尾才算省略追问。
#: 位置受限是为了避免把「那么，公司的不良贷款率是多少？」这类**自带主体**的正常问句误判
#: （那时 `extract_subject` 已能抽出主体，resolve 会直接短路返回）。
_ANAPHORA = re.compile(
    r"(它|其|该|这个|那个|上述|前述|该公司|本项目|这家|此项)"
    r"|^(那|那么|还有|然后)"
    r"|(呢|如何|怎么样)[？?]?$"
)

#: 主体名词（招股书/年报主体，用于判断上一轮问句里哪一段可以拿来补全）
#:
#: 工单7（人工智能NLP-RAG-功能测试及评估）修正：
#:     原正则**只认全称** ``…股份有限公司|有限公司``，而本工单语料（9 份 A 股年报）的问句几乎都用
#:     **简称**（「平安银行」「招商银行」「中国太保」…）⇒ 主体抽不到 ⇒ ``pending_subject=None`` ⇒
#:     多轮指代完全未消解（实测「那净利润呢？」「它」「该行」分别落到 中国平安 p169 /
#:     招商银行 p17 / 邮储银行 p83，全部答非所问）。
#:     现补一条**简称形态**：2~10 个汉字 + 机构通用后缀。仍然只认机构名，
#:     不会把「资产负债表」「不良贷款率」这类普通名词当主体。
_SUBJECT_FULL = re.compile(r"[\u4e00-\u9fff]{4,20}(?:股份有限公司|有限公司)")
_SUBJECT_SHORT = re.compile(
    r"[\u4e00-\u9fff]{2,10}(?:银行|证券|保险|集团|基金|信托|人寿|财险|寿险|资管|控股)")


def extract_subject(question: str, *, known_subjects: Any = None) -> str | None:
    """从问句里抽「主体」：优先**语料已知主体名**，其次全称形态，最后机构简称形态。

    ``known_subjects`` 可传入 ``IndexBundle.corpus_issuers`` 里的全部名字
    （由检索层从文件名/正文统计而来）——这是最可靠的一档：语料的真实主体名，
    不依赖任何中文形态规则，因此像「国泰君安」（不带机构通用后缀）也能识别。

    返回命中的**最长**主体名，避免「招商」抢走「招商证券」。
    """
    flat = re.sub(r"\s+", "", str(question or ""))
    if not flat:
        return None
    hits: list[str] = []
    if known_subjects:
        names: list[str] = []
        for group in (known_subjects.values() if hasattr(known_subjects, "values")
                      else [known_subjects]):
            for name in group or []:
                clean = re.sub(r"\s+", "", str(name))
                if len(clean) >= 4:
                    names.append(clean)
        # 去重并按长度降序，长名优先
        for name in sorted(dict.fromkeys(names), key=len, reverse=True):
            if name in flat:
                hits.append(name)
                break
    if not hits:
        hits = _SUBJECT_FULL.findall(flat) or _SUBJECT_SHORT.findall(flat)
    if not hits:
        return None
    return max(hits, key=len)


#: 兼容旧调用点（只做形态匹配、不接语料主体表）
def _match_subject_by_shape(question: str) -> str | None:
    return extract_subject(question)


# 兼容：保留模块级常量名，供既有测试/调用方直接 import
_SUBJECT = _SUBJECT_FULL


def _lazy_logger(logger: Any, module: str = "conversation") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class Turn:
    """一轮问答。"""

    question: str
    answer: str
    ts: float
    trace_id: str = ""
    cited_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "ts": self.ts,
                "trace_id": self.trace_id, "cited_pages": self.cited_pages}


class ConversationStore:
    """会话存储（线程安全；容量上限 + 可选落盘）。"""

    def __init__(self, *, max_turns: int = 5, max_sessions: int = 200,
                 storage: Any = None, logger: Any = None,
                 known_subjects: Any = None) -> None:
        self.max_turns = max(1, int(max_turns))
        self.max_sessions = max(1, int(max_sessions))
        self.storage = storage
        self.log = _lazy_logger(logger)
        self._sessions: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()
        # 工单7：指代消解用的「语料已知主体名」表（= IndexBundle.corpus_issuers）。
        # 由引擎在加载索引后注入；缺省为 None 时退回纯形态匹配（行为与工单4/5 一致）。
        self._known_subjects = known_subjects

    def set_known_subjects(self, known_subjects: Any) -> None:
        """注入/更新语料主体名表（引擎在建好 bundle 后调用）。"""
        self._known_subjects = known_subjects

    def add(self, session_id: str, turn: Turn) -> None:
        """追加一轮（超上限丢弃最旧）。"""
        if not session_id:
            return
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(turn)
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]
            if len(self._sessions) > self.max_sessions:
                oldest = min(self._sessions, key=lambda sid: self._sessions[sid][-1].ts)
                if oldest != session_id:
                    self._sessions.pop(oldest, None)
        self.log.log_event("conversation.turn_added", session_id=session_id,
                           turns=len(self._sessions.get(session_id, [])),
                           question_digest=turn.question[:60])
        if self.storage is not None:
            try:
                self.storage.touch_session(session_id, title=turn.question)
            except Exception as exc:  # noqa: BLE001 —— 落盘失败不影响会话
                self.log.log_event("conversation.persist_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id or "", []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id or "", None)

    def pending_subject(self, session_id: str) -> str | None:
        """取本会话的**焦点主体**（找不到返回 ``None``）。

        工单7 修正（实测缺陷）：原实现**只看最后一轮**问句 —— 一旦中间夹了一轮省略主体的追问
        （A1「平安银行…」→ A2「它的拨备覆盖率是多少？」），A2 的问句本身没有主体名，
        到 A3「该行的不良贷款率是多少？」时 ``pending_subject`` 就变成 None，
        指代链在第 3 轮断掉（实测 A3/B3 分别落到 邮储银行 p83）。
        现改为**从最近一轮往前回溯**第一个能抽出主体的问句 —— 这才是"焦点实体"的语义，
        也符合多轮对话的真实用法（焦点不会因为用户省略了一轮就消失）。
        """
        turns = self.history(session_id)
        for turn in reversed(turns):
            subject = extract_subject(turn.question, known_subjects=self._known_subjects)
            if subject:
                return subject
        return None

    def resolve(self, question: str, session_id: str | None) -> tuple[str, bool]:
        """指代消解：返回 ``(用于检索/生成的问题, 是否拼接了上文主体)``。"""
        log = self.log
        with log.enter("resolve", {"session_id": session_id, "question_digest": (question or "")[:60]}) as span:
            if not session_id:
                span.set_output({"resolved": False, "reason": "无 session_id（单轮问答）"})
                return question, False
            # 工单7：当前问句若已含主体（含简称/语料已知名）→ 无需消解
            if not _ANAPHORA.search(question or "") or extract_subject(
                    question, known_subjects=self._known_subjects):
                span.set_output({"resolved": False, "reason": "无指代词或问句已含主体"})
                return question, False
            subject = self.pending_subject(session_id)
            if not subject:
                span.set_output({"resolved": False, "reason": "上文未找到主体名词"})
                return question, False
            resolved = f"{subject}{question}"
            log.log_event("conversation.anaphora_resolved", session_id=session_id, subject=subject,
                          resolved_digest=resolved[:80])
            span.set_output({"resolved": True, "subject": subject, "chars": len(resolved)})
            return resolved, True


def now_turn(question: str, answer: str, *, trace_id: str = "", cited_pages: list[int] | None = None) -> Turn:
    """构造一轮记录。"""
    return Turn(question=question, answer=answer, ts=time.time(), trace_id=trace_id,
                cited_pages=list(cited_pages or []))
