"""RAG 主链路编排。

answer / answer_stream：
    检索(混合) -> 重排 -> 提示词(人设+上下文+短期记忆) -> LLM -> 后处理 -> 写回短期记忆
"""
from __future__ import annotations

import threading

from typing import Iterator

from .config import Settings
from .embedding import EmbeddingClient
from .llm import LLMClient
from .logging_config import get_logger
from .postprocess import postprocess
from .prompt.role_presets import ROLE_PRESETS
from .prompt.templates import build_messages
from .reranker import Reranker, create_reranker
from .errors import EmptyAnswerError
from .retrieve.hybrid_retriever import HybridRetriever
from .store.memory import MemoryStore, create_memory_store
from .store.milvus_store import MilvusStore
from .store.sql_store import SQLStore

log = get_logger("pipeline")

# EmptyAnswerError 定义已挪到 app/core/errors.py（llm.py 也要抛它，放这里会循环导入）；
# 上面那行 import 同时把它再导出一遍，`from app.core.pipeline import EmptyAnswerError` 照旧可用。


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedding: EmbeddingClient,
        milvus: MilvusStore,
        sql: SQLStore,
        memory: MemoryStore,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.embedding = embedding
        self.milvus = milvus
        self.sql = sql
        self.memory = memory
        self.retriever = retriever or HybridRetriever(settings, embedding, milvus)
        self.reranker = reranker or create_reranker(settings)
        # 同一会话的并发请求要串行（见 _session_lock）：前端只在单个标签页内禁并发，
        # 多标签页/脚本随时能同时打同一个 session_id。
        self._session_locks: dict[str, threading.Lock] = {}
        self._lock_guard = threading.Lock()

    # 会话锁最多留这么多把；超过只清"当前没人持有"的，持有中的不能删——
    # 删了同一会话会拿到两把不同的锁，串行化就失效了。
    _MAX_SESSION_LOCKS = 512

    def _session_lock(self, mem_key: str) -> threading.Lock:
        with self._lock_guard:
            lock = self._session_locks.get(mem_key)
            if lock is None:
                if len(self._session_locks) >= self._MAX_SESSION_LOCKS:
                    for k in [k for k, lk in self._session_locks.items() if not lk.locked()]:
                        del self._session_locks[k]
                        if len(self._session_locks) < self._MAX_SESSION_LOCKS:
                            break
                lock = self._session_locks[mem_key] = threading.Lock()
            return lock

    @classmethod
    def build(cls, settings: Settings) -> "RAGPipeline":
        """按配置组装所有依赖。"""
        return cls(
            settings=settings,
            llm=LLMClient(settings),
            embedding=EmbeddingClient(settings),
            milvus=MilvusStore(settings),
            sql=SQLStore(settings),
            memory=create_memory_store(settings),
        )

    # ---- 角色 ----
    def ensure_role(self, role_id: str):
        """获取角色；若为内置预设则自动登记入库。"""
        self.sql.connect()
        role = self.sql.get_role(role_id)
        if role is not None:
            # 老库没有 avatar 列时的兜底：预设角色头像为空就照预设补上，已有值不覆盖
            preset = ROLE_PRESETS.get(role_id)
            if preset and not role.avatar and preset.get("avatar"):
                role = self.sql.upsert_role(
                    role_id, role.name, role.description, role.system_prompt, preset["avatar"]
                )
            return role
        preset = ROLE_PRESETS.get(role_id)
        if preset is None:
            raise ValueError(f"未知角色: {role_id}")
        return self.sql.upsert_role(
            role_id,
            preset["name"],
            preset["description"],
            preset["system_prompt"],
            preset.get("avatar", ""),
        )

    # ---- 检索（独立暴露，供流式接口取 sources）----
    def retrieve(self, question: str, role_id: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.settings.top_k
        # 开精排时先按 rerank_pool 宽召回再精排截断——池子必须有富余，精排才挑得出东西。
        # score_fusion 时池宽就是 top_k，与加精排池之前的召回行为一致。
        pool = max(top_k, self.settings.rerank_pool) if self.settings.reranker == "bge" else top_k
        chunks = self.retriever.retrieve(question, role_id, pool)
        # 池宽必须是**精排输入的上界**，而不只是"每条 query 的召回宽度"。开了 query 改写时
        # 上游会把 ≤MAX_QUERIES 条 query 的召回结果去重合并，条数可达 pool 的数倍（实测
        # rerank_pool=20 + 4 条 query → 80 条，每查询还要再算一路 BM25），于是 20 条的池宽
        # 承诺失效：本机 CrossEncoder 逐对推理，一次 80 对就意味着十几秒到 60s 超时，超时即
        # 整体降级回 score_fusion，精排等于白开。融合分数（RRF）已排好序，截断取最好的 pool 条。
        # 只在开精排（pool > top_k）时截断：score_fusion 下 pool == top_k，本来就要取前
        # top_k 条，行为与截断前完全一致，不动它。
        if pool > top_k and len(chunks) > pool:
            log.debug("精排输入按池宽截断：%d -> %d 条", len(chunks), pool)
            chunks = chunks[:pool]
        chunks = self.reranker.rerank(question, chunks)
        return chunks[:top_k]

    @staticmethod
    def to_sources(chunks: list[dict]) -> list[dict]:
        return [
            {
                "text": c.get("text", ""),
                "title": c.get("title", ""),
                "source": c.get("source", ""),
                "score": round(float(c.get("score", 0.0)), 4),
            }
            for c in chunks
        ]

    # ---- 对话 ----
    def answer(self, question: str, role_id: str, session_id: str = "default") -> dict:
        # ① 取/建角色（内置预设首次会自动登记入库）
        role = self.ensure_role(role_id)
        # ② 检索：混合召回（稠密+BM25）→ RRF 融合 → 重排精排
        chunks = self.retrieve(question, role_id)
        # ③ 取短期记忆（多轮上下文，按「会话+角色」隔离，避免切角色串上下文）
        mem_key = self._memory_key(session_id, role_id)
        # 串行化同一会话：两个并发请求交错写记忆会把历史配错对——实测出现过
        # [user Q_A][user Q_B, assistant A_B, assistant A_A]，下一轮模型读到的就是错位对话史。
        # 检索（上面）不碰记忆、不进锁，锁只覆盖「读历史 → 生成 → 写历史」。
        with self._session_lock(mem_key):
            history = self.memory.get_history(mem_key)
            # ④ 拼提示词：角色人设 + 知识片段 + 历史 + 当前问题
            messages = build_messages(
                {"name": role.name, "system_prompt": role.system_prompt},
                chunks,
                history,
                question,
                max_history_chars=self.settings.history_max_chars,
                max_prompt_chars=self.settings.prompt_max_chars,
            )
            # ⑤ 大模型生成
            raw = self.llm.chat(messages)
            # ⑥ 后处理清洗（去 think、正则替换、去幻觉引用）
            answer = postprocess(raw)
            # ⑥' 空答案必须报错，不能静默返回空串（见 EmptyAnswerError）
            if not answer:
                log.warning("模型未产出回答 role=%s session=%s", role_id, session_id)
                raise EmptyAnswerError
            # ⑦ 写回短期记忆，支撑下一轮对话
            self._remember(mem_key, question, answer)
        return {
            "answer": answer,
            "sources": self.to_sources(chunks),
            "session_id": session_id,
        }

    def answer_stream(
        self,
        question: str,
        role_id: str,
        session_id: str = "default",
        chunks: list[dict] | None = None,
        final: list[str] | None = None,
    ) -> Iterator[str]:
        """流式输出文本增量；结束时把问答写入短期记忆。

        注意：yield 出去的是模型原始增量，没有过后处理（正则清洗天然需要完整文本，
        逐块做会破坏「行内多空格折叠」这类跨块语义）。清洗后的完整答案通过 `final`
        回传给调用方，供 /chat/stream 的 done 事件覆盖显示，保证流式与非流式口径一致。
        """
        role = self.ensure_role(role_id)
        if chunks is None:
            chunks = self.retrieve(question, role_id)
        mem_key = self._memory_key(session_id, role_id)
        # 锁覆盖整个流的生命周期：生成器被关闭（客户端断开）时 with 退出、锁自动释放。
        with self._session_lock(mem_key):
            history = self.memory.get_history(mem_key)
            messages = build_messages(
                {"name": role.name, "system_prompt": role.system_prompt},
                chunks,
                history,
                question,
                max_history_chars=self.settings.history_max_chars,
                max_prompt_chars=self.settings.prompt_max_chars,
            )

            collected: list[str] = []
            for delta in self.llm.stream(messages):
                collected.append(delta)
                yield delta
            full = postprocess("".join(collected))
            # 空答案：不写记忆、也不回传 final，抛出交给接口层发 error 事件。
            # 注意抛在 _remember 之前——否则会在历史里留一条空的 assistant 消息，
            # 下一轮模型读到的上下文就是「用户问了一句，自己什么都没答」。
            if not full:
                log.warning("模型未产出回答（流式）role=%s session=%s", role_id, session_id)
                raise EmptyAnswerError
            self._remember(mem_key, question, full)
            if final is not None:
                final.append(full)

    @staticmethod
    def _memory_key(session_id: str, role_id: str) -> str:
        """短期记忆按「会话 + 角色」隔离，避免切换角色时串入上一角色的上下文。"""
        return f"{session_id}:{role_id}"

    def _remember(self, mem_key: str, question: str, answer: str) -> None:
        self.memory.add(mem_key, "user", question)
        self.memory.add(mem_key, "assistant", answer)
