"""问答持久化编排：短期记忆获取与一问一答落库回写（任务 5.4）。

职责边界：
- 本模块承载 /chat/stream 的"持久化编排"：短期记忆实例的懒建、
  问答完成后的 MySQL 落库与 Redis 短期记忆回写
- 会话摘要（批次 21）的**触发点在 persist_turn 末尾**：一轮对话已完整收敛
  （MySQL 已落、Redis 已回写）才判断是否需要压缩更早轮次；执行放后台线程，
  不阻塞回答返回。摘要本身的逻辑在 app/memory/summary_service.py
- SSE 协议与端点逻辑留在 chat.py；本拆分只为遵守单文件 300 行上限，
  函数逻辑与注释与拆分前完全一致
- 存储实现本身在 chat_store.py（MySQL 写）与 memory/short_term.py（Redis）
"""

# 导入日志（持久化失败只记警告，不中断回答流）
import logging

# 导入线程（会话摘要放后台执行，不阻塞回答返回）
import threading

# 导入请求对象类型注解（懒建实例存放在 app.state 上）
from fastapi import Request
# 导入 Any（问答结果类型由注入方决定，注解层面不依赖具体类）
from typing import Any

# 导入 Redis 短期记忆存储（写回结构与 query_rewrite 的读取结构对齐）
from app.memory.short_term import ShortTermMemoryStore
# 导入写入/读取存储类型与生产装配（懒建入口的类型注解与默认构建用）
from app.chat.chat_history import ChatHistoryReader
from app.chat.chat_store import ChatSessionStore
from app.chat.chat_runtime import build_default_chat_history, build_default_chat_store

# 模块级日志器（额外字段 request_id 由调用方注入）
logger = logging.getLogger(__name__)

# 短期记忆窗口（消息条数）：本模块写回与 retrieval/assembly.py 装配检索服务时必须一致，
# 窗口差异会让查询改写读到的上下文与写回的不对称。取值的一致性由
# `tests/test_short_term_window_binding.py` 强制（覆盖本模块 / assembly / 评测）。
SHORT_TERM_MAX_MESSAGES = 20


def get_short_term_memory(request: Request) -> ShortTermMemoryStore | None:
    """从 app.state 懒建/复用 Redis 短期记忆存储（任务 5.4）。

    参数与 retrieval/assembly.py 的装配完全一致（ttl 用 session_ttl_seconds、
    窗口 20 条），保证"写回"与"查询改写时读取"看到同一个 key 和窗口。

    测试注入点：直接给 app.state.short_term_memory 赋替身即可。

    参数：
    - request: FastAPI 请求对象（通过它访问 app.state）

    返回：
    - ShortTermMemoryStore: 短期记忆存储实例
    """
    # 优先复用已存在的实例（含测试注入的替身）
    memory = getattr(request.app.state, "short_term_memory", None)
    # 尚未创建时按装配侧同款参数构建并缓存（create_redis_client 幂等单例）
    if memory is None:
        from app.core.config import settings
        from app.db.redis_client import create_redis_client

        memory = ShortTermMemoryStore(
            create_redis_client(settings.redis_url),
            ttl_seconds=settings.session_ttl_seconds,
            # 取值见模块常量（须与 assembly.py 的装配一致，由绑定测试强制）
            max_messages=SHORT_TERM_MAX_MESSAGES,
        )
        request.app.state.short_term_memory = memory
    # 返回存储实例
    return memory


def _run_session_summary(
    short_term_memory: ShortTermMemoryStore,
    user_id: str,
    session_key: str,
    request_id: str,
) -> None:
    """后台执行"判断是否需要更新会话摘要"（批次 21）。

    为什么放在本模块：这里是"一轮对话已完成"的唯一收敛点——消息已落 MySQL、
    已回写 Redis。放在能力层（ChatService）会让同步 chat() 与流式 chat_stream()
    两条路径各写一遍，且摘要与检索/生成耦合。

    失败策略：summary_service 内部已全包 try/except 并保留旧摘要；这里再兜一层，
    保证后台线程绝不把异常抛到无人接管的地方（daemon 线程抛错只会污染日志）。
    """
    try:
        from app.memory.summary_service import maybe_update_session_summary

        maybe_update_session_summary(
            short_term_memory=short_term_memory,
            user_id=user_id,
            session_id=session_key,
            request_id=request_id,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "会话摘要后台执行失败（已忽略）：%s",
            type(error).__name__,
            extra={"request_id": request_id},
        )


def persist_turn(
    *,
    chat_store: Any | None,
    short_term_memory: ShortTermMemoryStore | None,
    user_id: str,
    session_key: str,
    question: str,
    result: Any,
    request_id: str,
    message_id: str | None = None,
) -> None:
    """把一次完整问答写入 MySQL 并回写 Redis 短期记忆（任务 5.4）。

    失败策略：两个动作都"尽力而为"——落库失败在存储层已降级为 warning，
    Redis 失败在这里捕获为 warning；任何持久化问题都不允许抛出中断回答流
    （归属校验早已在 stream_chat 端点、流开始前完成并通过）。

    参数：
    - chat_store: MySQL 写入存储（None 时跳过落库，仅直调测试场景）
    - short_term_memory: Redis 短期记忆存储（None 时跳过写回）
    - user_id: 归属用户标识
    - session_key: 对外会话 ID
    - question: 清洗后的提问文本（与发给检索/LLM 的完全一致）
    - result: ChatService 返回的问答结果（answer / sources）
    - request_id: 请求追踪 ID（日志用）
    - message_id: assistant 消息对外 ID（透传落库，与 SSE message_start 一致）
    """
    # MySQL 落库：一问一答两条消息 + 会话活跃时间刷新；内部已吞非归属异常
    if chat_store is not None:
        chat_store.append_message_pair(
            session_key=session_key,
            user_id=user_id,
            question=question,
            answer=result.answer,
            sources=list(result.sources),
            # 模型名取当轮生效配置（批次 31）：记真实值，模型变更才可追溯（此前写死 NULL，
            # 导致历史消息无法回溯"这条回答是哪个模型生成的"）
            model=current_llm_model(),
            # 对外消息 ID 与 SSE message_start 保持一致（契约 7.6 一致性要求）
            message_id=message_id,
        )
    # Redis 短期记忆写回：与 query_rewrite 的读取结构对齐（role + content）
    if short_term_memory is not None:
        try:
            # 写回提问（user 角色），query_rewrite._latest_user_topic 依赖该结构
            short_term_memory.append_message(
                user_id, session_key, {"role": "user", "content": question}
            )
            # 写回回答（assistant 角色），构成下一轮改写用的短期上下文
            short_term_memory.append_message(
                user_id, session_key, {"role": "assistant", "content": result.answer}
            )
        except Exception as error:
            # Redis 抖动只记警告，不影响本次已完成的回答，也不影响下一轮降级改写
            logger.warning(
                "短期记忆写回失败：%s session_key=%s",
                type(error).__name__,
                session_key,
                extra={"request_id": request_id},
            )

    # 会话摘要（批次 21）：窗口满后按节流把更早轮次压成前情。
    # 开关默认关（SESSION_SUMMARY_ENABLED）；开启时放后台线程——摘要是额外的 LLM
    # 调用，绝不能拖慢首字延迟或让回答等待。判断条件在 summary_service 内。
    if short_term_memory is not None and _session_summary_enabled():
        threading.Thread(
            target=_run_session_summary,
            args=(short_term_memory, user_id, session_key, request_id),
            daemon=True,
        ).start()


def current_llm_model() -> str | None:
    """读当轮生效的 LLM 模型名（懒读配置，测试可通过环境变量覆盖）。

    为什么由持久化层读而不是让 LLM 客户端回传：模型名是"当轮配置"的事实，
    落库只需要它，为此把模型名一路穿过检索/生成链路会把配置耦合进能力层。
    取值失败时落 None（禁止伪造），与 _session_summary_enabled 的降级策略一致。
    """
    try:
        from app.core.config import settings

        # 空串按"未配置"处理（配置缺省可能是 ""，不该当作有效模型名入库）
        return settings.llm_model or None
    except Exception:  # noqa: BLE001
        # 配置不可读时不阻断回答：模型名只是溯源信息，缺失由后续补录
        return None


def _session_summary_enabled() -> bool:
    """读会话摘要开关（懒读配置，测试可通过环境变量覆盖）。"""
    try:
        from app.core.config import settings

        return bool(settings.session_summary_enabled)
    except Exception:  # noqa: BLE001
        # 配置不可读时按"关闭"处理：宁可少一段前情，不可让持久化层报错
        return False


def get_chat_store(request: Request) -> ChatSessionStore:
    """从 app.state 懒建/复用 MySQL 写入存储实例（自 session_routes 迁入）。

    迁移原因：session_routes 加分页参数后逼近 300 行上限，
    两个懒建入口与"持久化编排"同属装配职责，收拢到本模块。

    测试注入点：直接给 app.state.chat_session_store 赋替身即可，
    与 chat.py 的 _get_chat_service 注入模式同构。

    参数：
    - request: FastAPI 请求对象（通过它访问 app.state）

    返回：
    - ChatSessionStore: 写入存储实例
    """
    # 优先复用已存在的实例（懒建模式，避免每个请求重建连接工厂）
    store = getattr(request.app.state, "chat_session_store", None)
    # 尚未创建时连接真实 MySQL 构建默认实例并缓存
    if store is None:
        store = build_default_chat_store()
        request.app.state.chat_session_store = store
    # 返回存储实例
    return store


def get_chat_history(request: Request) -> ChatHistoryReader:
    """从 app.state 懒建/复用 MySQL 读取存储实例（自 session_routes 迁入）。

    测试注入点：直接给 app.state.chat_history_reader 赋替身即可。

    参数：
    - request: FastAPI 请求对象（通过它访问 app.state）

    返回：
    - ChatHistoryReader: 读取存储实例
    """
    # 优先复用已存在的实例（含测试注入的替身）
    reader = getattr(request.app.state, "chat_history_reader", None)
    # 尚未创建时连接真实 MySQL 构建默认实例并缓存
    if reader is None:
        reader = build_default_chat_history()
        request.app.state.chat_history_reader = reader
    # 返回读取实例
    return reader
