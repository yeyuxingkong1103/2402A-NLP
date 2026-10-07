"""
多轮会话状态
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
工单编号：人工智能NLP-RAG-Query理解优化任务

工单5 的交付物之一「多轮对话」需要一层**会话状态**：把前面几轮问过什么、
系统把它理解成了什么存下来，供本轮消解指代与省略使用。

--------------------------------------------------------------------------
为什么是「内存态 + 限长 + 线程安全」这三件事
--------------------------------------------------------------------------
1. **内存态**：工单要的是「多轮对话能力」，不是「对话历史管理平台」。
   本系统是单机演示/评测环境，回答完即弃，落盘反而带来越权与隐私问题。
2. **限长**：会话必须**有上限**，两个原因：
   * 性能——拼进 prompt 的历史越长，LLM 往返越慢，工单要求端到端 ≤3s；
   * 内存——会话数不设上限就是内存泄漏，而性能验收明确要求「高并发稳定」。
   所以同时限制「每会话保留轮数」与「全局会话数」。
3. **线程安全**：FastAPI 的默认线程池会并发处理请求，同一 session_id 的
   两次请求可能同时读写。用一把 `RLock` 把整个存储圈起来 ——
   临界区极短（只做列表增删与字典查找），不会成为性能瓶颈。

--------------------------------------------------------------------------
每轮存什么（这是设计的关键）
--------------------------------------------------------------------------
只存「原始问题 + 系统答案」是不够的。真正要传承给下一轮的是**语义槽位**：

  * `subject`  —— 本轮问的是**哪家公司**（doc_key + 人类可读名）。
                  这是指代消解的答案来源：「他」「这个公司」都指向它。
  * `intent`   —— 本轮问的是**哪一类事**（法定代表人 / 注册资本 / 收入 …）。
                  这是省略补全的答案来源：「那力源呢？」需要把上一轮的
                  intent 搬到新一轮的主体上。
  * `resolved` —— 本轮**消解后**的自包含问题（「他参与的哪个工程…」→
                  「武汉兴图新科电子股份有限公司参与的哪个工程…」）。
                  下一轮做省略补全时，用它可以拿到完整的句型模板。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .config import (
    DOCS_BY_KEY,
    DOC_ALIASES,
    MULTITURN_MAX_TURNS,
    WORK_ORDER_NO_QU,  # noqa: F401  工单编号：人工智能NLP-RAG-Query理解优化任务
)

logger = logging.getLogger(__name__)

# 全局会话数上限。超出后按「最久未使用」淘汰。
# 取 200：单会话最多 6 轮 × ~200 字节，200 个会话也只有几百 KB，
# 但足以覆盖演示 + 压测场景，且杜绝了「session_id 无限增长撑爆内存」。
MAX_SESSIONS = 200


@dataclass
class Turn:
    """一轮问答。字段名刻意与 QueryUnderstanding 对齐，减少转换。"""

    q: str                      # 用户原始问题
    a: str = ""                 # 系统答案（截断后保存，仅供回看/演示）
    resolved: str = ""          # 消解后的自包含问题（送检索的那个）
    subject: str = ""           # 本轮主体（人类可读全称，如「武汉力源信息技术股份有限公司」）
    doc_key: str = ""           # 主体对应的文档键（xingtu / liyuan），用于文档消歧先验
    intent: str = "其他"        # 本轮意图（事实查询 / 财务数据查询 / …）
    topic: str = ""             # 本轮问的「事」的短标签（如「法定代表人」），省略补全要用
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "q": self.q,
            "a": self.a,
            "resolved": self.resolved,
            "subject": self.subject,
            "doc_key": self.doc_key,
            "intent": self.intent,
            "topic": self.topic,
        }


def detect_subject(text: str) -> tuple[str, str]:
    """
    从一句话里识别**主体公司**，返回 (doc_key, 公司全称)。

    用 `config.DOC_ALIASES` 做匹配，按别名长度**降序**扫描 ——
    否则「兴图」会先命中「兴图新科电子股份有限公司」的前缀，拿到半截名字。
    识别不到返回 ("", "")。

    注意别名里的「力源」两个字在语料里也大量出现在「武汉力源」之外的语境，
    但招股书语料里它就是指这家公司，这里不做更细的歧义处理（如实记录在文档里）。
    """
    t = text or ""
    best: tuple[int, str, str] = (0, "", "")   # (命中长度, doc_key, 全称)
    for key, aliases in DOC_ALIASES.items():
        company = DOCS_BY_KEY.get(key, {}).get("company", aliases[0])
        for alias in sorted(aliases, key=len, reverse=True):
            if alias and alias in t and len(alias) > best[0]:
                best = (len(alias), key, company)
    return (best[1], best[2]) if best[0] else ("", "")


def detect_topic(text: str) -> str:
    """
    给一句话打一个「问的是什么事」的短标签 —— 省略补全时要用它。

    这不是意图分类，而是给「那力源呢？」找一个可搬运的谓词：
    上一轮问的是「法定代表人」，本轮就要接着问「法定代表人」。

    做法：按一组**有序**的关键词表去匹配，先命中者胜（顺序＝优先级）。
    为什么不直接用 LLM 的分类结果？因为这条路径要能在**规则模式下**独立工作
    （消融的 `rule` 臂不调 LLM），必须有一份确定性的兜底。
    """
    t = text or ""
    # 顺序有意义：越具体的越靠前，避免被更泛的词先抢走。
    # 例：「法定代表人」必须排在「人」类泛词之前。
    for topic, pats in _TOPIC_TABLE:
        for p in pats:
            if p in t:
                return topic
    return ""


# (短标签, 命中关键词) —— 顺序＝优先级
_TOPIC_TABLE: list[tuple[str, tuple[str, ...]]] = [
    ("法定代表人", ("法定代表人", "法人代表", "法人")),
    ("注册资本", ("注册资本", "股本总额")),
    ("发行股数", ("发行股数", "发行数量", "发行多少股")),
    ("募集资金", ("募集资金", "募投项目", "募集资金投向")),
    ("关联方", ("关联方", "关联企业", "控制关系")),
    ("组织结构", ("组织结构", "部门构成", "销售处", "组织架构")),
    ("技术标准", ("技术标准", "标准制定", "参与制定")),
    ("荣誉奖项", ("科技进步奖", "荣获", "获奖", "荣誉称号")),
    ("军用收入", ("军用领域", "军品", "军用")),
    ("上下游", ("上游", "下游", "供应商", "客户")),
    ("收入", ("收入", "营业收入", "营收")),
    ("占比", ("占比", "比重", "比例")),
    ("供应商", ("供应商",)),
]


@dataclass
class Session:
    """一个会话（同一 session_id 下的多轮）。"""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -------------------------------------------------- 写入
    def add(self, turn: Turn) -> None:
        """追加一轮，并把会话裁到 `MULTITURN_MAX_TURNS`。"""
        self.turns.append(turn)
        if len(self.turns) > MULTITURN_MAX_TURNS:
            # 只丢最早的：最近几轮才是消解的上下文来源
            self.turns = self.turns[-MULTITURN_MAX_TURNS:]
        self.updated_at = time.time()

    def update_last_answer(self, answer: str, limit: int = 600) -> None:
        """回填最新一轮的答案（生成是异步的，轮次先落、答案后补）。"""
        if self.turns:
            self.turns[-1].a = (answer or "")[:limit]
            self.updated_at = time.time()

    # -------------------------------------------------- 读取
    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def recent(self, n: int) -> list[Turn]:
        """最近 n 轮，按时间正序返回。"""
        return self.turns[-n:] if n > 0 else []

    def last_subject(self) -> tuple[str, str]:
        """向上回溯找最近一个「有主体」的轮次。

        为什么要回溯而不是只看上一轮：用户可能连问两次省略句
        （「那力源呢？」→「注册资本呢？」），中间那轮没有新主体，
        此时主体应当继续沿用更早那轮，而不是丢掉。
        """
        for t in reversed(self.turns):
            if t.subject:
                return (t.doc_key, t.subject)
        return ("", "")

    def last_topic(self) -> str:
        """向上回溯找最近一个「有事」的轮次（省略补全的谓词来源）。"""
        for t in reversed(self.turns):
            if t.topic:
                return t.topic
        return ""

    def last_intent(self) -> str:
        for t in reversed(self.turns):
            if t.intent and t.intent != "其他":
                return t.intent
        return "其他"

    def last_resolved(self) -> str:
        return self.turns[-1].resolved if self.turns else ""

    def to_dict(self, include_answers: bool = True) -> dict:
        return {
            "session_id": self.session_id,
            "turns": len(self.turns),
            "items": [t.to_dict() if include_answers
                      else {k: v for k, v in t.to_dict().items() if k != "a"}
                      for t in self.turns],
        }


class SessionStore:
    """会话存储。进程内、限长、线程安全。"""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._max = max_sessions

    # -------------------------------------------------- 基本操作
    def get(self, session_id: str) -> Session:
        """取会话；不存在则创建。"""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                self._evict_if_needed()
                s = Session(session_id=session_id)
                self._sessions[session_id] = s
            return s

    def peek(self, session_id: str) -> Session | None:
        """只读查看，不存在返回 None（不创建）。"""
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> bool:
        """清空某会话的内容（但保留 session_id）。返回是否命中已有会话。"""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return False
            s.turns.clear()
            s.updated_at = time.time()
            return True

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    # -------------------------------------------------- 运维
    def _evict_if_needed(self) -> None:
        """超过上限时淘汰「最久未更新」的会话。调用方已持有锁。"""
        while len(self._sessions) >= self._max:
            oldest = min(self._sessions.items(), key=lambda kv: kv[1].updated_at)[0]
            self._sessions.pop(oldest, None)
            logger.info("会话数达上限 %d，淘汰最久未用：%s", self._max, oldest)

    def stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "max_sessions": self._max,
                "max_turns_per_session": MULTITURN_MAX_TURNS,
                "turns_total": sum(len(s.turns) for s in self._sessions.values()),
            }


# 模块级单例：与 index_store / embedder 同一套约定（进程内共享）
_STORE: SessionStore | None = None
_STORE_LOCK = threading.Lock()


def store() -> SessionStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = SessionStore()
    return _STORE


def get_session(session_id: str) -> Session:
    return store().get(session_id)
