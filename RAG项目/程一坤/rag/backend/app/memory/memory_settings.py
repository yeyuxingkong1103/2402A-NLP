"""长期记忆的用户级开关存储（MySQL 侧，自 long_term.py 拆出）。

接口 8.3：PUT /api/v1/users/me/memory-settings。
与 Milvus 存储分离的动机：开关落 MySQL users 表（随用户注销/审计一起管理），
记忆本体在 Milvus 独立集合——两种存储、两种生命周期，不该挤在一个文件。
关闭后既不写入也不读取（由调用方先查 is_enabled）。
"""
from __future__ import annotations


class MemorySettingsStore:
    """用户级长期记忆开关（接口 8.3：PUT /api/v1/users/me/memory-settings）。

    关闭后既不写入也不读取（由调用方先查 is_enabled）。
    设置落 MySQL users.long_term_memory_enabled（默认开），随用户注销/审计一起管理。
    """

    def __init__(self, session_factory) -> None:
        """注入会话工厂。

        参数：session_factory —— 可调用对象，`with session_factory() as session` 用法，
              返回一个 SQLAlchemy Session。
        返回：None。

        只存工厂而不是 Session 本体：每个方法各自开一次短会话，
        避免把长事务/失效连接持在对象里（该类实例会被装配成常驻单例）。
        """
        self.session_factory = session_factory

    def is_enabled(self, user_id: str) -> bool:
        """读开关；用户不存在按关闭处理（不做无归属判断）。

        参数：user_id —— 用户标识（对应表字段 User.user_key）。
        返回：True = 允许读写长期记忆；False = 关闭（含 user_id 为空 / 用户不存在）。

        为什么"查不到用户"要返回 False 而不是抛错：本方法会被装在 ChatService
        里当门卫，每轮问答前都会调；用户被删、令牌未同步等边界情况下，
        宁可安静地不写记忆，也不能让整个问答接口 500。
        """
        from sqlalchemy import select

        from app.db.sql_models import User

        if not user_id:
            return False
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.user_key == user_id))
            if user is None:
                return False
            return bool(user.long_term_memory_enabled)

    def set_enabled(self, user_id: str, enabled: bool) -> bool:
        """更新开关；用户不存在返回 False。

        参数：user_id —— 用户标识；enabled —— 目标开关值（会强制转 bool，
              避免前端传 "false" 之类的字符串被当成真值）。
        返回：True = 已提交；False = 用户不存在，未做任何写入。

        返回值与 is_enabled 的"查不到就 False"保持一致，让上层只用
        "真值 = 成功"一种判断方式，不必区分异常与不存在。
        """
        from sqlalchemy import select

        from app.db.sql_models import User

        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.user_key == user_id))
            if user is None:
                return False
            user.long_term_memory_enabled = bool(enabled)
            session.commit()
            return True
