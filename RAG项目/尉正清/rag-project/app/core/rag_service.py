# app/core/rag_service.py
"""RAG 主流程编排。

一次问答的完整链路：
    角色装载 -> 双层记忆召回 -> 追问改写 -> 混合检索 -> 重排
    -> 提示词组装 -> 大模型生成 -> 免责声明 -> 记忆与消息落库
"""
from datetime import datetime
from typing import Dict, Iterator, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.core.chain_service import build_answer_chain
from app.core.memory_service import get_memory
from app.core.query_service import get_query_service
from app.db import redis_conn
from app.core.prompt import build_system_prompt
from app.core.retrieve_service import get_retriever
from app.core.role_service import RoleService
from app.core.session_service import SessionService
import logging

logger = logging.getLogger(__name__)


class RagService:

    def __init__(self):
        self.retriever = get_retriever()
        self.memory = get_memory()
        # 生成改走 LangChain 的 LCEL 链（见 chain_service），
        # 这里不再持有 LLM 实例

    # ---------------- 内部：组装一轮所需的上下文 ----------------
    def _prepare(self, db: Session, user_id: int, role_key: str,
                 question: str, session_id: Optional[str]):
        role = RoleService.get_by_key(db, role_key)
        if role is None:
            raise ValueError("角色不存在或已停用: %s" % role_key)
        if not role.status:
            raise ValueError("角色已停用: %s" % role_key)

        session = SessionService.get_or_create(db, user_id, role_key, session_id,
                                               title=question)
        sid = session.session_id

        # 1) 双层记忆
        history, memories = self.memory.get_context(user_id, role_key, sid, question)

        # 2) 查询理解：先按上下文改写追问，再扩写成多个检索查询
        queries = get_query_service().prepare(
            question, history, expand_n=settings.QUERY_EXPAND_N)
        search_query = queries[0]           # 对外展示改写后的问题
        if search_query != question:
            logger.info("追问改写: %s -> %s", question, search_query)
        if len(queries) > 1:
            logger.debug("扩写查询: %s", queries[1:])

        # 3) 多查询 × 多路召回 → RRF 融合 → 精排
        docs = self.retriever.search(
            queries, role_key,
            db=db if settings.ENABLE_METADATA_ROUTE else None)

        # 4) 组装提示词
        system_prompt = build_system_prompt(
            persona=role.persona or "你是一位专业的智能助手。",
            rules=role.rules,
            fallback=role.fallback,
            memories=memories,
            contexts=docs)

        return role, session, system_prompt, history, docs, search_query

    @staticmethod
    def _history_messages(history_lines):
        """把 Redis 里的文本行还原成 LangChain 消息对象。"""
        from app.core.memory_service import _line_to_message
        return [_line_to_message(l) for l in (history_lines or []) if l and l.strip()]

    def _chain_input(self, system_prompt: str, history, question: str) -> dict:
        return {
            "system_prompt": system_prompt,
            "history": self._history_messages(history),
            "question": question,
        }

    def _persist_turn(self, db: Session, user_id: int, role_key: str, session,
                      question: str, answer: str, final: str, docs: list) -> None:
        """一轮结束后统一落库。

        Redis 短期记忆 + Redis 会话热状态 + MySQL 归档，三者各司其职：
        热状态放 Redis Hash，避免每次读会话概况都要打 MySQL。
        """
        sid = session.session_id
        # 记忆存不带免责声明的正文，避免摘要被套话占满
        self.memory.save_turn(user_id, role_key, sid, question, answer)
        SessionService.touch(db, session, question)
        SessionService.add_message(db, sid, "user", question)
        SessionService.add_message(
            db, sid, "assistant", final,
            sources=[{"title": d["title"], "source": d["source"],
                      "score": d["score"]} for d in docs])
        redis_conn.set_session_meta(
            sid, role_key=role_key, user_id=str(user_id),
            turn_count=str(session.turn_count or 0),
            last_active=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def _with_disclaimer(role, answer: str) -> str:
        text = (answer or "").strip()
        # 模型偶发返回空正文（敏感话题上尤其容易只产出推理就停止），
        # 此时不能让用户只看到一句免责声明，用角色兜底话术顶上
        if not text:
            text = (role.fallback or "").strip() or \
                "抱歉，我暂时没能组织好这个回答，可以换个说法再问一次吗？"
        d = (role.disclaimer or "").strip()
        if d and d not in text:
            text = "%s\n\n%s" % (text, d)
        return text

    # ---------------- 同步问答 ----------------
    def chat(self, db: Session, user_id: int, role_key: str, question: str,
             session_id: Optional[str] = None) -> Dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        role, session, system_prompt, history, docs, search_query = self._prepare(
            db, user_id, role_key, question, session_id)

        # 走 LangChain 的 LCEL 链：提示词 → 模型 → 取正文
        answer = build_answer_chain().invoke(
            self._chain_input(system_prompt, history, question))

        # 后处理：核对外呼的法条引用，防止编造条文号
        answer, citation = self._validate(answer, db)
        final = self._with_disclaimer(role, answer)
        self._persist_turn(db, user_id, role_key, session, question,
                           answer, final, docs)

        return {
            "answer": final,
            "session_id": session.session_id,
            "role": role.to_dict(),
            "search_query": search_query,
            "sources": docs,
            "citation_check": citation,
        }

    @staticmethod
    def _validate(answer: str, db: Session) -> tuple:
        """法条引用校验。关掉或失败时原样返回，不影响主流程。"""
        if not settings.ENABLE_CITATION_CHECK:
            return answer, None
        try:
            from app.core.validate_service import get_validate_service
            return get_validate_service().postprocess(answer, db)
        except Exception as e:                              # pragma: no cover
            # 本模块用的是标准库 logging，占位符必须是 %s。
            # 写成 loguru 风格的 {} 会让这行在 emit 时抛 TypeError，
            # 被 logging 的 handleError 吞掉——日志打不出来，还多一段 stderr 噪声。
            logger.warning("引用校验失败，跳过: %s", e)
            return answer, None

    # ---------------- 流式问答 ----------------
    def chat_stream(self, db: Session, user_id: int, role_key: str, question: str,
                    session_id: Optional[str] = None) -> Iterator[Dict]:
        """逐段产出。先给 meta（含来源），再给若干 delta，最后 done。"""
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        role, session, system_prompt, history, docs, search_query = self._prepare(
            db, user_id, role_key, question, session_id)

        yield {"type": "meta", "session_id": session.session_id,
               "role": role.to_dict(), "search_query": search_query,
               "sources": docs}

        pieces: List[str] = []
        chain = build_answer_chain()
        for piece in chain.stream(self._chain_input(system_prompt, history, question)):
            pieces.append(piece)
            yield {"type": "delta", "content": piece}

        answer = "".join(pieces).strip()
        answer, citation = self._validate(answer, db)
        final = self._with_disclaimer(role, answer)

        # 免责声明在流式里单独补发一段
        if final != answer:
            tail = final[len(answer):]
            if tail:
                yield {"type": "delta", "content": tail}

        self._persist_turn(db, user_id, role_key, session, question,
                           answer, final, docs)
        yield {"type": "done", "session_id": session.session_id,
               "citation_check": citation}


_service: Optional[RagService] = None


def get_rag_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service
