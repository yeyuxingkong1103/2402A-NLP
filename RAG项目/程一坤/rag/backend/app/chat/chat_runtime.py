"""聊天存储的生产装配：MySQL 会话工厂与默认存储实例（任务 5.4）。

职责边界：
- 本模块只负责"生产路径装配"：连接串构造、会话工厂、写入/读取实例的懒建入口
- 存储逻辑本身在 chat_store.py（写 + 归属校验）与 chat_history.py（读）
- 独立成文件：为 chat_store 的删除接口腾出 300 行空间，同时把三个
  build_default_* 装配函数收拢到一处，连接参数只维护一份
"""

# 导入 ORM 会话工厂类型注解
from sqlalchemy.orm import sessionmaker

# 导入写入侧存储类（含归属校验实现）
from app.chat.chat_store import ChatSessionStore
# 导入读取侧存储类
from app.chat.chat_history import ChatHistoryReader


def build_default_session_factory() -> sessionmaker:
    """构造连接真实 MySQL 的默认会话工厂（写入与读取两个存储共用）。

    参数：
    - 返回 sessionmaker: 绑定 MySQL 引擎的会话工厂

    说明：
    - 连接串只在这一处构造，写入与读取实例复用本函数，
      避免同一份连接参数在多个模块里各自维护
    """
    # 导入 URL 转义工具（数据库密码可能含特殊字符）
    from urllib.parse import quote_plus

    # 导入应用配置（MySQL 连接参数）
    from app.core.config import settings
    # 导入引擎与会话工厂构造器
    from app.db.engine import create_database_engine, create_session_factory

    # 构造 MySQL 连接串（与 retrieval/assembly.py 的构造逻辑保持一致，含密码转义）
    database_url = (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:{quote_plus(settings.mysql_password)}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )

    # 返回绑定了真实引擎的会话工厂
    return create_session_factory(create_database_engine(database_url))


def build_default_chat_store() -> ChatSessionStore:
    """构建连接真实 MySQL 的默认写入存储实例（生产路径；测试用 app.state 注入替身）。"""
    # 连接串与会话工厂统一由共享函数构造
    return ChatSessionStore(session_factory=build_default_session_factory())


def build_default_chat_history() -> ChatHistoryReader:
    """构建连接真实 MySQL 的默认读取存储实例（生产路径；测试用 app.state 注入替身）。"""
    # 连接串与会话工厂统一由共享函数构造
    return ChatHistoryReader(session_factory=build_default_session_factory())
