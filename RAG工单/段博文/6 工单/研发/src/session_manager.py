# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
会话管理模块：管理多轮对话历史，支持指代消解与主语继承/切换。

核心功能：
    1. 会话存储：session_id → 对话历史列表 [{role, content}]；
    2. 历史窗口：默认保留最近 N 轮，避免上下文过长；
    3. 实体追踪：记录最近一次明确提到的"主语实体"（公司名/人名等），
       用于"这个公司/他/那XX呢"等指代的快速消解；
    4. 会话过期：超过 TTL 未活跃的会话自动清理；
    5. 会话清空：支持显式 reset。
"""

import re
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from logger import get_logger

logger = get_logger(__name__)


# ==================== 配置 ====================
SESSION_MAX_HISTORY = int(__import__('os').getenv("SESSION_MAX_HISTORY", "10"))  # 每个会话最多保留的轮数
SESSION_TTL_SEC = int(__import__('os').getenv("SESSION_TTL_SEC", "3600"))        # 会话过期时间
SESSION_MAX_COUNT = int(__import__('os').getenv("SESSION_MAX_COUNT", "1024"))    # 最大会话数（LRU）


class Session:
    """单条会话。"""

    __slots__ = ("session_id", "history", "last_entity", "updated_at", "created_at")

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []     # [{role: 'user'|'assistant', content: str}]
        self.last_entity: Optional[str] = None       # 最近一个明确实体（公司名等）
        self.created_at = time.time()
        self.updated_at = time.time()

    def append(self, role: str, content: str):
        """追加一轮对话。"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > SESSION_MAX_HISTORY * 2:  # user+assistant 两条 = 一轮
            self.history = self.history[-SESSION_MAX_HISTORY * 2:]
        self.updated_at = time.time()

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """获取历史，默认全部。"""
        if last_n is None:
            return list(self.history)
        return list(self.history[-last_n * 2:])  # 一轮 = user + assistant

    def clear(self):
        self.history.clear()
        self.last_entity = None
        self.updated_at = time.time()


class SessionManager:
    """会话管理器（线程安全，LRU）。"""

    def __init__(self):
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        """获取或创建会话。"""
        with self._lock:
            if session_id in self._sessions:
                sess = self._sessions.pop(session_id)
                self._sessions[session_id] = sess  # LRU：移到末尾
                return sess
            sess = Session(session_id)
            self._sessions[session_id] = sess
            # 超容量则淘汰最旧
            while len(self._sessions) > SESSION_MAX_COUNT:
                self._sessions.popitem(last=False)
            return sess

    def reset(self, session_id: str) -> bool:
        """清空某会话历史。"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                return True
            return False

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self):
        """清理过期会话。"""
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items()
                       if now - s.updated_at > SESSION_TTL_SEC]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                logger.info(f"[SessionManager] 清理过期会话 {len(expired)} 个")

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": SESSION_MAX_COUNT,
                "max_history_per_session": SESSION_MAX_HISTORY,
                "ttl_sec": SESSION_TTL_SEC,
            }


# 全局单例
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager


# ==================== 实体识别 ====================
# 公司名识别正则（招股说明书场景）
# 匹配 "XX股份有限公司"、"XX有限公司"、"XX集团" 等
_COMPANY_PATTERN = re.compile(
    r"([一-龥]{2,30}?(?:股份有限公司|有限责任公司|有限公司|集团|公司))"
)


def extract_entities(text: str) -> List[str]:
    """从文本中抽取公司/机构实体。返回去重后的实体列表。"""
    if not text:
        return []
    entities = _COMPANY_PATTERN.findall(text)
    # 去重保序
    seen = set()
    out = []
    for e in entities:
        e = e.strip()
        if e and e not in seen and len(e) >= 4:
            seen.add(e)
            out.append(e)
    return out


def detect_query_type(query: str) -> str:
    """粗粒度识别问题类型，辅助改写策略。

    Returns:
        'standalone'   - 完整独立问题（含明确实体），无需改写
        'coreference'  - 含指代词（他/这个/那/其/该），需要消解
        'switch'       - 主语切换（"那XX呢"），需要替换主语
        'followup'     - 省略主语的追问
    """
    if not query:
        return 'standalone'

    # 主语切换标志：那/那么 + 实体 + 呢/？
    if re.search(r"^(那|那么|而)\s*[一-龥]{2,30}?(?:股份有限公司|有限责任公司|有限公司|集团|公司)?\s*呢", query):
        return 'switch'

    # 指代词
    coref_words = ["他", "她", "它", "这个", "那个", "该", "其", "这家", "那家",
                   "该公司", "这家公司", "这家公司", "上述", "前面提到"]
    if any(w in query for w in coref_words):
        return 'coreference'

    # 已经有明确公司实体，判为独立问题
    if _COMPANY_PATTERN.search(query):
        return 'standalone'

    # 短句且缺主语，视为追问
    if len(query) <= 25 and not re.search(r"^(谁|什么|哪|何|怎么|为什么|请问)", query):
        return 'followup'

    return 'standalone'


def update_session_entity(session: Session, text: str):
    """从最新文本中更新会话的当前实体。"""
    entities = extract_entities(text)
    if entities:
        session.last_entity = entities[0]
        logger.debug(f"[Session] 更新当前实体：{session.last_entity}")


def get_history_for_rewrite(session: Session, max_turns: int = 3) -> List[Tuple[str, str]]:
    """提取最近 N 轮对话，格式 [(user_q, assistant_a), ...] 供改写使用。"""
    history = session.get_history(last_n=max_turns)
    pairs: List[Tuple[str, str]] = []
    cur_q: Optional[str] = None
    for msg in history:
        if msg["role"] == "user":
            cur_q = msg["content"]
        elif msg["role"] == "assistant" and cur_q is not None:
            pairs.append((cur_q, msg["content"]))
            cur_q = None
    return pairs
