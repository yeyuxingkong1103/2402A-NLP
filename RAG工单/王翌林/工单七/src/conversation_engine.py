# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
src/conversation_engine.py —— 工单五 多轮对话引擎（新增文件）

核心能力：
  1. 会话管理（session_id 维度的对话历史 + 当前实体追踪）
  2. 指代消解（他/这个公司/其 → 当前公司；那X呢 → 切换实体并复用问法）
  3. 追问改写（将依赖上下文的追问改写为独立可检索的完整问句）
  4. 委托 RAGEngineV4 执行检索问答

设计原则：指代消解采用规则+实体追踪的确定性方案（低延迟、可解释），
不引入额外 LLM 调用，保障单轮 ≤3s 响应指标。
"""
import re
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional

from loguru import logger

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化任务"

# 工单五：已知公司实体库（招股说明书1/2）
_COMPANIES = {
    "武汉兴图新科电子股份有限公司": "招股说明书1",
    "兴图新科": "招股说明书1",
    "兴图": "招股说明书1",
    "武汉力源信息技术股份有限公司": "招股说明书2",
    "力源信息": "招股说明书2",
    "力源": "招股说明书2",
}

# 工单五：指代消解关键词（第三人称代词 + 指代词，长词优先匹配）
_COREF_PRONOUNS = ["这个公司", "那个公司", "这家公司", "那家公司",
                   "本公司", "该公司", "此公司",
                   "他", "她", "它", "其", "该", "这个", "那个", "这家", "那家"]

# 工单五："那X呢" 切换实体模式
_SWITCH_PATTERN = re.compile(
    r"^(?:那|那么|那)?\s*([\u4e00-\u9fa5A-Za-z0-9]+?)(?:股份有限公司|有限公司|公司|信息|新科|力源)?\s*(?:呢|吗|怎么样|如何|是什么|有哪些)\s*[？?]?$")


def _detect_company(text: str) -> Optional[str]:
    """工单五：从文本中识别公司全称（最长匹配优先）"""
    found = []
    for name in _COMPANIES:
        if name in text:
            found.append(name)
    if not found:
        return None
    # 工单五：取最长匹配（避免"兴图"先于"武汉兴图新科电子股份有限公司"命中）
    return max(found, key=len)


class ConversationSession:
    """工单五：单会话状态（历史消息 + 当前实体 + 最后问法）"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []     # [{role, content, answer, ...}]
        self.current_entity: Optional[str] = None    # 当前讨论的公司全称
        self.current_doc: Optional[str] = None       # 当前实体对应文档
        self.last_question: Optional[str] = None     # 上一轮原始问题（用于"那X呢"复用问法）
        self.last_intent: Optional[str] = None       # 上一轮问法关键词（如"法定代表人"）
        self.created_at = time.time()
        self.updated_at = time.time()

    def add_turn(self, question: str, answer: str,
                 entity: Optional[str] = None, doc: Optional[str] = None,
                 metadata: Optional[Dict] = None) -> None:
        """工单五：追加一轮对话并更新实体追踪"""
        self.history.append({
            "role": "user", "content": question,
            "ts": time.time()})
        self.history.append({
            "role": "assistant", "content": answer,
            "ts": time.time(), **(metadata or {})})
        if entity:
            self.current_entity = entity
        if doc:
            self.current_doc = doc
        self.last_question = question
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "history": self.history,
            "current_entity": self.current_entity,
            "current_doc": self.current_doc,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ConversationEngine:
    """工单五：多轮对话引擎（会话管理 + 指代消解 + 追问改写 + RAG 委托）"""

    def __init__(self, rag_engine: Optional[Any] = None):
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = Lock()
        self._rag = rag_engine

    # ------------------------------------------------------------------
    def _get_rag(self):
        """工单五：懒加载 RAGEngineV4（复用工单四图文表融合引擎）"""
        if self._rag is None:
            from src.rag_engine_v4 import RAGEngineV4
            self._rag = RAGEngineV4(top_k=5)
        return self._rag

    # ------------------------------------------------------------------
    def get_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """工单五：获取或创建会话"""
        with self._lock:
            if not session_id:
                session_id = uuid.uuid4().hex[:12]
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSession(session_id)
            return self._sessions[session_id]

    # ------------------------------------------------------------------
    def resolve_coref(self, query: str, session: ConversationSession) -> Dict[str, Any]:
        """工单五：指代消解 + 追问改写

        返回：{
          resolved_query: 改写后的独立问句,
          entity: 当前实体（公司全称）,
          doc_id: 实体对应文档,
          is_followup: 是否为追问,
          strategy: 消解策略描述
        }
        """
        q = query.strip()
        detected_entity = _detect_company(q)
        entity = detected_entity
        is_followup = False
        strategy = "direct"

        # 工单五：策略1 —— "那X呢？" 切换实体并复用问法（优先检测，即使问句含完整公司名）
        # 例："那武汉力源信息技术股份有限公司呢？" → 切换到力源信息，问法同前一轮（法定代表人）
        if session.last_intent:
            m = _SWITCH_PATTERN.match(q)
            if m:
                candidate = m.group(1)
                # 工单五：优先用已检测到的实体，否则模糊匹配公司名
                switch_entity = detected_entity
                if not switch_entity:
                    for full_name in _COMPANIES:
                        if candidate in full_name or full_name in candidate:
                            switch_entity = full_name
                            break
                if switch_entity:
                    entity = switch_entity
                    q = f"{switch_entity}{session.last_intent}"
                    is_followup = True
                    strategy = "switch_entity_reuse_intent"

        # 工单五：策略2 —— 指代消解（他/这个公司/其 → 当前实体）
        if not entity:
            has_pronoun = any(p in q for p in _COREF_PRONOUNS)
            if has_pronoun and session.current_entity:
                entity = session.current_entity
                # 工单五：将代词替换为实体全称
                replaced = q
                for p in sorted(_COREF_PRONOUNS, key=len, reverse=True):
                    replaced = replaced.replace(p, entity)
                q = replaced
                is_followup = True
                strategy = "pronoun_resolution"

        # 工单五：策略3 —— 无实体且无代词但有上一轮实体，继承实体
        if not entity and session.current_entity:
            # 工单五：仅当问句不含新实体时继承（避免误覆盖显式指定的公司）
            entity = session.current_entity
            is_followup = True
            strategy = "inherit_entity"

        doc_id = _COMPANIES.get(entity) or session.current_doc

        # 工单五：记录本轮问法关键词（供下一轮"那X呢"复用）
        intent_kw = self._extract_intent(query)
        if intent_kw:
            session.last_intent = intent_kw

        return {
            "resolved_query": q,
            "entity": entity,
            "doc_id": doc_id,
            "is_followup": is_followup,
            "strategy": strategy,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_intent(query: str) -> Optional[str]:
        """工单五：提取问法关键词（用于"那X呢"时复用问法）

        例："这个公司的法定代表人是谁？" → "的法定代表人是谁"
            "他参与的哪个工程荣获了国家科技进步一等奖？" → "参与的哪个工程荣获了国家科技进步一等奖"
        """
        q = query.strip().rstrip("？?。.")
        # 工单五：去掉主语部分（公司名/代词），保留问法
        for p in sorted(_COREF_PRONOUNS + list(_COMPANIES.keys()),
                        key=len, reverse=True):
            if q.startswith(p):
                q = q[len(p):]
                break
        q = q.lstrip("的，,")
        return q if len(q) >= 3 else None

    # ------------------------------------------------------------------
    def chat(self, query: str, session_id: Optional[str] = None,
             doc_id: Optional[str] = None, use_image: bool = True,
             **kw) -> Dict[str, Any]:
        """工单五：多轮对话主入口

        流程：指代消解 → 问句改写 → RAGEngineV4.ask → 历史更新
        """
        t0 = time.perf_counter()
        session = self.get_session(session_id)

        # 工单五：指代消解
        coref = self.resolve_coref(query, session)
        resolved = coref["resolved_query"]
        entity = coref["entity"]
        target_doc = doc_id or coref["doc_id"]

        logger.info(f"[conv_v5] sid={session.session_id} "
                    f"raw='{query[:40]}' → resolved='{resolved[:40]}' "
                    f"entity={entity} doc={target_doc} strategy={coref['strategy']}")

        # 工单五：委托 RAGEngineV4（图文表融合检索）
        rag = self._get_rag()
        result = rag.ask(resolved, doc_id=target_doc, use_image=use_image, **kw)

        # 工单五：更新会话历史与实体追踪
        session.add_turn(
            question=query, answer=result.get("answer", ""),
            entity=entity, doc=target_doc,
            metadata={"resolved_query": resolved,
                      "strategy": coref["strategy"],
                      "latency_ms": result.get("latency_ms", 0)})

        result.update({
            "session_id": session.session_id,
            "resolved_query": resolved,
            "entity": entity,
            "doc_id": target_doc,
            "is_followup": coref["is_followup"],
            "coref_strategy": coref["strategy"],
            "conversation_latency_ms": round(
                (time.perf_counter() - t0) * 1000, 1),
        })
        return result

    # ------------------------------------------------------------------
    def get_history(self, session_id: str) -> Dict[str, Any]:
        """工单五：获取会话历史"""
        session = self.get_session(session_id)
        return session.to_dict()

    # ------------------------------------------------------------------
    def reset_session(self, session_id: str) -> None:
        """工单五：重置会话"""
        with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """工单五：会话数 + RAG 健康"""
        rag_rows = 0
        try:
            rag_rows = self._get_rag().health().get("milvus_images_rows", 0)
        except Exception:
            pass
        return {
            "sessions": len(self._sessions),
            "milvus_images_rows": rag_rows,
        }
