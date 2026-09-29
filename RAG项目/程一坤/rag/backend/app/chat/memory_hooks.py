"""记忆读取钩子（长期记忆 + 会话前情，记忆职责，自 chat/service.py 拆出）。

以 Mixin 形式挂到 ChatService 上：各方法都只依赖 self 上的记忆配置字段
（long_term_memory / memory_gate / memory_top_k / memory_write_async /
session_summary_enabled / llm_client / retrieval_service），
不触碰检索与 LLM 主流程。设计原则（批次 14）：记忆只是个性化背景，
**任何失败都不能影响问答主流程**——读取失败返回 None，写入失败只记日志。

两类记忆的分工：
- 长期记忆（跨会话）：`_memory_block` 读、`_maybe_write_memory` 写
- 会话前情（本会话）：`_session_summary` 读（批次 21；写入侧在
  app/memory/summary_service.py，由 API 持久化层在回答完成后触发）
"""
import threading

from app.memory.memory_summarize import summarize_memory_fact
from app.memory.summary_policy import SUMMARY_MAX_CHARS


class ChatMemoryMixin:
    """ChatService 的长期记忆读写钩子（检索前读、回答后写）。"""

    def _memory_enabled(self, user_id: str | None) -> bool:
        """长期记忆读写前置校验：有存储、有用户、用户未关闭。"""
        if not self.long_term_memory or not user_id:
            return False
        if self.memory_gate is None:
            return True
        try:
            return bool(self.memory_gate(user_id))
        except Exception:  # noqa: BLE001
            # 开关校验失败按"未开启"处理：宁可少个性化，不可错读
            return False

    def _memory_block(self, user_id: str | None, question: str) -> str | None:
        """检索法律知识之前取该用户记忆，格式化成提示词段落。

        任何失败都返回 None（记忆只是个性化背景，不能影响回答主流程）。
        """
        if not self._memory_enabled(user_id):
            return None
        try:
            records = self.long_term_memory.search(
                user_id, question, top_k=self.memory_top_k
            )
        except Exception as error:  # noqa: BLE001
            # 检索失败不吞问题，但也不阻断回答：记日志、返回 None
            import logging

            logging.getLogger("app.chat.memory").warning(
                "长期记忆检索失败：%s: %s", type(error).__name__, error
            )
            return None
        if not records:
            return None
        lines = [f"{i}. {record.summary}" for i, record in enumerate(records, start=1)]
        return "\n".join(lines)

    def _session_summary(self, user_id: str | None, session_id: str | None) -> str | None:
        """读取本次会话的前情摘要，供提示词注入（批次 21）。

        与长期记忆的分工：长期记忆是**跨会话**的用户事实，会话前情是**本会话**
        已被滑动窗口淘汰轮次的压缩摘要；两者同属"背景"而不是"依据"。

        返回 None 的所有情形都代表"本轮不注入前情"，与引入摘要之前的行为一致：
        - 开关关闭（session_summary_enabled=False）→ 连 Redis 都不读
        - 缺 user_id/session_id、检索侧没有短期记忆存储
        - 读取失败（Redis 抖动）、摘要为空
        - 摘要超过 SUMMARY_MAX_CHARS → 截断后注入并记一次 warning

        为什么从 retrieval_service 上取存储：短期记忆实例由检索装配持有
        （retrieval/assembly.py），chat 层复用同一个 store，避免两套连接看到不同窗口。
        """
        if not getattr(self, "session_summary_enabled", False):
            return None
        if not user_id or not session_id or not self.retrieval_service:
            return None
        store = getattr(self.retrieval_service, "short_term_memory", None)
        if store is None:
            return None
        try:
            summary = store.read_summary(user_id, session_id)
        except Exception as error:  # noqa: BLE001
            import logging

            logging.getLogger("app.chat.memory").warning(
                "会话摘要读取失败（本轮不注入前情）：%s: %s", type(error).__name__, error
            )
            return None
        if not summary or not summary.strip():
            return None
        text = summary.strip()
        if len(text) > SUMMARY_MAX_CHARS:
            # 提示词预算硬约束：摘要必须给法源清单让位，超长就截断
            import logging

            logging.getLogger("app.chat.memory").warning(
                "会话摘要超长，已截断注入：%d -> %d 字", len(text), SUMMARY_MAX_CHARS
            )
            text = text[:SUMMARY_MAX_CHARS]
        return text

    def _maybe_write_memory(
        self,
        user_id: str | None,
        session_id: str | None,
        question: str,
        answer: str,
        refused: bool,
    ) -> None:
        """一次问答结束后写入长期记忆（摘要 → 去重更新/新增）。

        - 拒答/无用户/开关关闭：不写（拒答没有产生新的用户事实）
        - 摘要失败：跳过本条（summarize_memory_fact 内部兜底）
        - memory_write_async=True 时放后台线程，**不阻塞回答返回**
        - 任何异常只记日志，绝不影响问答主流程
        """
        if refused or not self._memory_enabled(user_id):
            return
        if not self.llm_client:
            return

        def _write() -> None:
            try:
                fact = summarize_memory_fact(self.llm_client, question, answer)
                if fact is None:
                    return
                memory_id, updated = self.long_term_memory.add_memory(
                    user_id=user_id,
                    content=f"问：{question[:1500]}\n答：{answer[:2000]}",
                    summary=fact["summary"],
                    character_id="",
                    importance=fact["importance"],
                    source_session_id=session_id or "",
                )
                import logging

                logging.getLogger("app.chat.memory").info(
                    "长期记忆写入：%s（%s）", memory_id, "更新" if updated else "新增"
                )
            except Exception as error:  # noqa: BLE001
                import logging

                logging.getLogger("app.chat.memory").warning(
                    "长期记忆写入失败（已忽略）：%s: %s", type(error).__name__, error
                )

        if self.memory_write_async:
            threading.Thread(target=_write, daemon=True).start()
        else:
            _write()
