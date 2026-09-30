"""聊天会话与消息的持久化存储：写入侧与归属校验（任务 5.4）。

职责边界：
- 本模块负责"写"：自动建档、一问一答落库，以及跨模块共用的归属校验
- "读"（会话列表 / 消息历史）在 chat_history.py；标题规则在 chat_title.py
- 归属校验失败抛 SessionAccessDenied（能力层异常），由 API 层统一转成
  404 响应，遵循"不暴露会话是否存在"的原则
- Redis 短期记忆的写回由 API 层另行调用 ShortTermMemoryStore，
  与本模块无关（key 结构 short_memory:{user_id}:{session_id}:messages）
"""

# 导入 JSON 序列化（citations 字段存储用）
import json
# 导入日志（落库失败时的警告记录）
import logging

# 导入 SQLAlchemy 查询构造器（delete 用于删除会话消息）
from sqlalchemy import delete, select
# 导入数据库连接异常类型（并发建档时的唯一键冲突捕获）
from sqlalchemy.exc import IntegrityError
# 导入 ORM 会话类型注解
from sqlalchemy.orm import Session, sessionmaker

# 导入聊天会话与消息的 ORM 模型（utc_now 供手动刷新活跃时间复用）
from app.db.chat_models import ChatMessage, ChatSession
from app.db.sql_models import utc_now

# 模块级日志器（额外字段 request_id 由调用方注入）
logger = logging.getLogger(__name__)


class SessionAccessDenied(Exception):
    """会话不存在或不属于当前用户。

    两种情况对外统一表现为 404（不区分"不存在"与"存在但不属于你"），
    避免攻击者通过状态码差异探测他人会话 ID 是否有效。
    """


def assert_session_owned(
    session: Session, user_id: str, session_key: str
) -> ChatSession:
    """在给定数据库会话内校验会话归属；查不到即抛 SessionAccessDenied。

    设计要点：查询条件同时携带 session_key 与 user_id，
    SQL 层直接过滤他人会话——即使猜中 ID 也读不到任何数据。

    参数：
    - session: 已开启的 SQLAlchemy 数据库会话（由调用方管理事务）
    - user_id: 认证上下文给出的用户标识
    - session_key: 对外会话 ID

    返回：
    - ChatSession: 归属校验通过的会话记录

    异常：
    - SessionAccessDenied: 会话不存在或不属于本人（对外同一句话，不暴露存在性）
    """
    # 查询条件同时带 session_key 与 user_id：SQL 层直接过滤掉他人会话
    record = session.scalar(
        select(ChatSession).where(
            ChatSession.session_key == session_key,
            ChatSession.user_id == user_id,
        )
    )
    # 查不到 → 不存在或不属于你，对外同一句话，不暴露存在性
    if record is None:
        raise SessionAccessDenied("会话不存在")
    # 校验通过，返回会话记录
    return record


class ChatSessionStore:
    """聊天会话与消息的 MySQL 写入存储；所有操作都携带 user_id 过滤。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化写入存储层。

        参数：
        - session_factory: SQLAlchemy 会话工厂（事务边界由本类方法内部管理）
        """
        # 保存会话工厂，供每个方法独立开启短事务
        self._session_factory = session_factory

    def get_or_create_session(
        self,
        user_id: str,
        session_key: str,
        character_id: str = "legal-assistant",
        title: str | None = None,
    ) -> ChatSession:
        """获取本人会话；不存在则自动建档（首问自动建档设计）。

        归属规则：
        - session_key 存在且属于 user_id → 原样返回
        - session_key 存在但属于他人 → 抛 SessionAccessDenied（API 层转 404）
        - session_key 不存在 → 以当前用户身份创建

        参数：
        - user_id: 认证上下文给出的用户标识
        - session_key: 调用方自带的会话 ID
        - character_id: 助手角色 ID（首期固定 legal-assistant）
        - title: 显式标题（创建会话接口可指定）；自动建档时由 API 层先算好传入

        返回：
        - ChatSession: 归属校验通过（或新建）的会话记录

        异常：
        - SessionAccessDenied: 会话存在但属于他人
        """
        # 开启独立事务完成"查 → 判 → 建"
        with self._session_factory() as session:
            # 按对外会话 ID 查找（session_key 全局唯一）
            existing = session.scalar(
                select(ChatSession).where(ChatSession.session_key == session_key)
            )

            # 会话已存在：先做归属校验，再决定放行还是拒绝
            if existing is not None:
                # 归属人不匹配 → 拒绝；404 语义由 API 层转换
                if existing.user_id != user_id:
                    raise SessionAccessDenied("会话不存在")
                # 归属人匹配 → 原样返回
                return existing

            # 会话不存在：以当前用户身份自动建档
            record = ChatSession(
                # 对外会话 ID（调用方自带）
                session_key=session_key,
                # 归属人（认证上下文，不接受请求体伪造）
                user_id=user_id,
                # 助手角色 ID
                character_id=character_id,
                # 标题：显式指定优先，否则用"新会话 + 时间"兜底；
                # 首问场景下 API 层会先用 derive_title 算好标题再传入
                title=title or derive_title(None),
            )
            # 加入当前事务
            session.add(record)

            try:
                # 提交事务，会话记录落库
                session.commit()
            except IntegrityError:
                # 并发兜底：两个请求同时为同一 session_key 建档，
                # 唯一键冲突后回滚并重查（对方事务已建好的那条）
                session.rollback()
                # 重新开启事务查询同一会话
                existing = session.scalar(
                    select(ChatSession).where(ChatSession.session_key == session_key)
                )
                # 二次归属校验（重查到的记录可能是别人的）
                if existing is None or existing.user_id != user_id:
                    # 仍不存在或归属不符 → 统一拒绝
                    raise SessionAccessDenied("会话不存在") from None
                # 归属匹配 → 返回重查结果
                return existing

            # 建档成功，返回新记录
            return record

    def append_message_pair(
        self,
        session_key: str,
        user_id: str,
        question: str,
        answer: str,
        sources: list[dict],
        model: str | None = None,
        message_id: str | None = None,
    ) -> None:
        """把一次问答落库为"user 提问 + assistant 回答"两条消息，并刷新会话活跃时间。

        失败策略：落库失败只记 warning 不抛出（持久化是尽力而为，
        不能因为存储抖动中断正在返回的回答流；归属校验在流开始前已单独完成）。

        参数：
        - session_key: 对外会话 ID
        - user_id: 归属用户标识（用于把活跃时间刷在本人会话上）
        - question: 提问原文
        - answer: 回答最终文本（含护栏处理）
        - sources: 引用法源列表（非空时 JSON 序列化存入 citations）
        - model: 生成模型名（由调用方从当轮配置取，取不到时为 None，禁止伪造）
        - message_id: assistant 消息的对外 ID（与 SSE message_start 一致；
          user 消息不写入该列，未传场景为 NULL）
        """
        try:
            # 开启独立短事务，失败时整体回滚
            with self._session_factory() as session:
                # 按会话 ID + 归属人定位会话（复用 SQL 层归属过滤）
                record = assert_session_owned(session, user_id, session_key)

                # 写入 user 消息（提问原文）
                session.add(
                    ChatMessage(
                        # 关联内部会话主键
                        session_id=record.id,
                        # 发言角色：提问方
                        role="user",
                        # 提问原文
                        content=question,
                        # user 消息没有引用与模型信息
                        citations=None,
                        model=None,
                    )
                )

                # 引用法源：非空才序列化存储，空列表存 NULL（语义更干净）
                citations_json = (
                    json.dumps(sources, ensure_ascii=False) if sources else None
                )

                # 写入 assistant 消息（回答 + 引用 + 模型名）
                session.add(
                    ChatMessage(
                        # 关联内部会话主键
                        session_id=record.id,
                        # 发言角色：助手
                        role="assistant",
                        # 回答最终文本
                        content=answer,
                        # 引用法源 JSON（无引用时为 NULL）
                        citations=citations_json,
                        # 生成模型名（可为 None）
                        model=model,
                        # 对外消息 ID：与 SSE message_start 事件一致（契约 7.6 一致性要求）
                        message_id=message_id,
                    )
                )

                # 手动刷新会话活跃时间（无其他字段变更时 onupdate 不会触发）
                record.updated_at = utc_now()

                # 提交事务：两条消息 + 会话活跃时间一次性落库
                session.commit()
        except SessionAccessDenied:
            # 归属异常原样上抛（调用方在流开始前已校验过，这里属于防御路径）
            raise
        except Exception as error:
            # 任何落库异常都不允许影响回答流，只记警告日志（含异常类型，不记正文）
            logger.warning(
                "问答落库失败：%s session_key=%s",
                type(error).__name__,
                session_key,
            )


    def delete_session(self, user_id: str, session_key: str) -> None:
        """删除本人会话及其全部消息（同一事务先删消息再删会话）。

        归属规则与会话获取一致：session_key 不存在或不属于 user_id
        统一抛 SessionAccessDenied（API 层转 404，不暴露存在性）。

        参数：
        - user_id: 认证上下文给出的用户标识
        - session_key: 对外会话 ID

        异常：
        - SessionAccessDenied: 会话不存在或不属于本人
        """
        # 开启独立事务：消息与会话记录必须同生共死，避免留下孤儿消息
        with self._session_factory() as session:
            # 复用 SQL 层归属过滤（session_key + user_id 双条件查会话）
            record = assert_session_owned(session, user_id, session_key)

            # 先删该会话的全部消息（表上没有级联删除配置，手动保证完整性）
            session.execute(
                delete(ChatMessage).where(ChatMessage.session_id == record.id)
            )

            # 再删会话记录本身
            session.delete(record)

            # 提交事务：两条删除一起生效，任何一步失败整体回滚
            session.commit()


# 延迟导入标题规则（模块底部导入避免循环依赖；chat_title 不依赖本模块）
from app.chat.chat_title import derive_title  # noqa: E402
