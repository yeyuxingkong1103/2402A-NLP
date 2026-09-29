"""聊天会话与消息的读取侧：会话列表与历史消息（任务 5.4）。

职责边界：
- 本模块负责"读"：本人会话列表（含消息条数）、会话内历史消息
- "写"（自动建档 / 问答落库）与归属校验实现在 chat_store.py
- 标题规则在 chat_title.py；三者拆分只为遵守单文件 300 行上限
- 读取同样走 SQL 层归属过滤（复用 assert_session_owned），
  归属失败抛 SessionAccessDenied，由 API 层统一转 404
"""

# 导入 JSON 反序列化（citations 字段在库里存的是 JSON 字符串）
import json

# 导入 SQLAlchemy 聚合函数（统计会话消息条数）与查询构造器
from sqlalchemy import func, select
# 导入 ORM 会话类型注解
from sqlalchemy.orm import Session, sessionmaker

# 导入聊天会话与消息的 ORM 模型
from app.db.chat_models import ChatMessage, ChatSession
# 导入写入侧的归属校验（读写共用同一套 SQL 层过滤，避免两处实现漂移）
from app.chat.chat_store import assert_session_owned


class ChatHistoryReader:
    """聊天会话与消息的 MySQL 读取存储；所有查询都携带 user_id 过滤。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化读取存储层。

        参数：
        - session_factory: SQLAlchemy 会话工厂（事务边界由本类方法内部管理）
        """
        # 保存会话工厂，供每个方法独立开启短事务
        self._session_factory = session_factory

    def list_sessions(
        self, user_id: str, page: int = 1, page_size: int = 20
    ) -> list[dict]:
        """返回本人会话的分页列表（按最近活跃倒序），附带每个会话的消息条数。

        参数：
        - user_id: 认证上下文给出的用户标识（SQL 层只返回本人会话）
        - page: 页码（1 起始；下限由 API 层 Query 校验保证）
        - page_size: 每页条数（1~100，由 API 层 Query 校验保证）

        返回：
        - list[dict]: 每项含 session_key / character_id / title /
          message_count / created_at / updated_at；时间字段为 datetime，
          由 API 层统一序列化
        """
        # 开启只读事务（纯查询无需显式提交）
        with self._session_factory() as session:
            # 一条 SQL 同时取会话记录与消息条数：左连接消息表 + 按会话分组计数
            # outerjoin 保证一条消息都没有的会话也出现在列表里（计数为 0）
            rows = session.execute(
                select(
                    # 会话整行记录
                    ChatSession,
                    # 该会话下的消息条数（聚合列）
                    func.count(ChatMessage.id),
                )
                # 连接条件：消息表的 session_id 外键 = 会话主键
                .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
                # 归属过滤：SQL 层直接排除他人会话
                .where(ChatSession.user_id == user_id)
                # 按会话主键分组（主键函数依赖使其他列合法出现在 select 中）
                .group_by(ChatSession.id)
                # 最近活跃的会话排最前；主键作次序键保证同秒记录排序稳定
                .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
                # 分页：offset 从 0 计而 page 从 1 计，故偏移 = (page - 1) * page_size
                .limit(page_size)
                .offset((page - 1) * page_size)
            ).all()

        # 把 (会话记录, 消息条数) 元组列表整理为对外字典列表
        return [
            {
                # 对外会话 ID（调用方在请求里使用的就是这个值）
                "session_key": record.session_key,
                # 助手角色 ID
                "character_id": record.character_id,
                # 会话标题（首条提问前 20 字或"新会话 + 时间"兜底）
                "title": record.title,
                # 该会话的消息条数（含 user 与 assistant 两侧）
                "message_count": message_count,
                # 建档时间
                "created_at": record.created_at,
                # 最近活跃时间（append_message_pair 每次问答都会刷新）
                "updated_at": record.updated_at,
            }
            # 逐行解包：会话 ORM 记录 + 聚合出的消息条数
            for record, message_count in rows
        ]

    def list_messages(
        self, user_id: str, session_key: str, page: int = 1, page_size: int = 20
    ) -> list[dict]:
        """返回本人指定会话的分页消息（按时间正序），citations 已反序列化。

        参数：
        - user_id: 认证上下文给出的用户标识
        - session_key: 对外会话 ID
        - page: 页码（1 起始；下限由 API 层 Query 校验保证）
        - page_size: 每页条数（1~100，由 API 层 Query 校验保证）

        返回：
        - list[dict]: 每项含 message_id / role / content / citations / model /
          created_at；citations 为反序列化后的法源列表（无引用或脏数据为 None）；
          message_id 仅 assistant 消息有值（与 SSE message_start 一致），
          user 消息与历史旧行为 None

        异常：
        - SessionAccessDenied: 会话不存在或不属于本人（API 层转 404，
          不区分"不存在"与"存在但不属于你"）
        """
        # 开启独立事务：归属校验与消息查询在同一快照内完成，避免竞态
        with self._session_factory() as session:
            # 复用写入侧的归属校验：查询条件同时带 session_key 与 user_id，
            # 查不到（不存在或属于他人）统一抛 SessionAccessDenied
            record = assert_session_owned(session, user_id, session_key)

            # 按消息主键正序取本会话全部消息（id 自增，等价于历史时间先后）
            rows = session.scalars(
                select(ChatMessage)
                # 只取归属校验通过的这个会话的消息
                .where(ChatMessage.session_id == record.id)
                # 正序：先问后答，与对话发生的真实顺序一致
                .order_by(ChatMessage.id.asc())
                # 分页：offset 从 0 计而 page 从 1 计，故偏移 = (page - 1) * page_size
                .limit(page_size)
                .offset((page - 1) * page_size)
            ).all()

        # 逐条整理为对外字典结构（不再暴露 ORM 对象）
        return [
            {
                # 对外消息 ID（仅 assistant 有值，与 SSE message_start 一致；
                # user 消息与历史旧行为 None）
                "message_id": message.message_id,
                # 发言角色：user / assistant
                "role": message.role,
                # 消息正文（提问原文或回答最终文本）
                "content": message.content,
                # 引用法源列表（已反序列化；user 消息与无引用回答为 None）
                "citations": self._deserialize_citations(message.citations),
                # 生成模型名（LLM 客户端未暴露时为 None）
                "model": message.model,
                # 消息落库时间
                "created_at": message.created_at,
            }
            for message in rows
        ]

    @staticmethod
    def _deserialize_citations(raw: str | None) -> list[dict] | None:
        """把 citations 字段的 JSON 字符串还原为法源列表。

        参数：
        - raw: 数据库中的原始值（None 或 JSON 字符串）

        返回：
        - list[dict] | None: 反序列化结果；空值或解析失败一律返回 None
        """
        # 空值直接返回 None（与写入侧"空列表存 NULL"的语义对应）
        if not raw:
            return None
        try:
            # 正常路径：写入侧由 json.dumps 生成，必然可解析
            return json.loads(raw)
        except ValueError:
            # 防御路径：历史脏数据（非 JSON 文本）不允许炸掉历史接口，
            # 降级为"无引用"展示
            return None
