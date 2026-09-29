# -*- coding: utf-8 -*-
"""
MySQL 数据层：用户表、角色表、对话记录表。

在系统中的位置：
    上游是 main.py（注册/登录、拉角色列表、每轮问答后写审计日志）以及初始化脚本；
    下游只有 MySQL 一个外部依赖。本模块负责「结构化且需要长期保存」的数据，
    不参与向量检索（那是 db_milvus 的职责）。

职责：
    用 SQLAlchemy ORM 定义三张表，并封装建表与增删改查；业务层只调用这里的函数，
    不直接拼 SQL，改表结构或换库时改动范围可控。

关键设计取舍：
    1. 密码只存哈希（password_hash），绝不存明文；哈希由上层算好后传进来；
    2. 连接池 pool_size=5 且 pool_pre_ping=True：MySQL 会主动断开长时间空闲的连接，
       pre_ping 会在取用连接时先探活，避免拿到死连接报 "MySQL server has gone away"；
    3. 每个 CRUD 函数内部自建 session 并在 finally 里 close（归还连接池），不跨函数
       共享 session：实现简单，也避免长事务和连接泄漏；
    4. 表结构用 create_all 在启动时创建，不做版本迁移（教学项目够用，生产应引入 Alembic）。
"""

from datetime import datetime  # 时间戳
from typing import Optional  # 可选类型标注（查不到时返回 None）
import sqlalchemy  # ORM 框架（仅用到 sqlalchemy.Float，避免把每种列类型都 import 一遍）
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean  # 列类型
from sqlalchemy.orm import declarative_base, sessionmaker  # 基类 + 会话工厂

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB  # MySQL 连接配置
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# SQLAlchemy 连接串：mysql+pymysql://用户:密码@地址:端口/库名
# charset 用 utf8mb4 而不是 utf8：MySQL 的 utf8 只支持 3 字节，存不了 emoji 等 4 字节字符
CONN_STR = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"  # 用户密码
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"  # 地址端口库名（库须已存在，本层不负责 CREATE DATABASE）
    f"?charset=utf8mb4"  # UTF-8 编码（支持 emoji）
)

Base = declarative_base()  # ORM 基类：所有表模型继承它（建表信息统一挂在 Base.metadata 上）
engine = create_engine(CONN_STR, echo=False, pool_pre_ping=True, pool_size=5)  # 引擎：连接池 5 个，自动 ping 检活（echo=True 可打印 SQL 便于调试）
SessionLocal = sessionmaker(bind=engine, autoflush=False)  # 会话工厂：每次 CRUD 创建一个 session（autoflush=False 避免查询前被隐式刷盘）


def init_db():
    """
    启动时调用：如果表不存在则自动创建（CREATE TABLE IF NOT EXISTS）

    返回：
        None。
    异常与降级：
        数据库连不上或库不存在时直接抛异常，让服务启动即失败，而不是带病运行。
    """
    Base.metadata.create_all(engine)  # 建表（已存在则跳过，不会覆盖或清空数据）
    logger.info("MySQL 表结构初始化完成")  # 日志


# ==================== ORM 表模型 ====================

class User(Base):  # 用户表：多用户支持
    __tablename__ = "users"  # 表名
    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键，也是 Redis 会话 Key 里的 user_id
    username = Column(String(64), unique=True, nullable=False, index=True)  # 用户名：登录用，唯一、建索引（登录查询走索引）
    password_hash = Column(String(256), nullable=False)  # 密码哈希：只存哈希不存明文，长度 256 兼容 bcrypt/sha256 等
    nickname = Column(String(64))  # 昵称：聊天中显示的名字
    created_at = Column(DateTime, default=datetime.utcnow)  # 注册时间（存 UTC，避免时区混乱）
    is_active = Column(Boolean, default=True)  # 是否启用（软删除用：置 False 而非物理删除，保留历史对话的关联）


class Role(Base):  # 角色表：多角色支持
    __tablename__ = "roles"  # 表名
    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键，也是 Milvus 长期记忆里的 role_id
    name = Column(String(64), unique=True, nullable=False, index=True)  # 角色名：律师/医生/心理医生...
    description = Column(String(256))  # 角色描述：一句话简介，前端角色卡片展示用
    system_prompt = Column(Text, nullable=False)  # 系统提示词：定义角色人格、工作原则、输出格式（用 Text 不设长度上限）
    temperature = Column(sqlalchemy.Float, default=0.5)  # 角色专属温度：律师 0.3（严谨）/ 朋友 0.7（活泼）
    knowledge_collection = Column(String(64))  # 该角色专属的 Milvus 集合名（空则用 config 里的默认知识库）
    created_at = Column(DateTime, default=datetime.utcnow)  # 创建时间
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 修改时间（onupdate 由 ORM 在 UPDATE 时自动刷新）


class ConversationLog(Base):  # 对话日志表：审计 + 调试
    __tablename__ = "conversation_logs"  # 表名
    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键
    user_id = Column(Integer, index=True)  # 用户 ID（建索引方便按用户查历史）
    role_id = Column(Integer, index=True)  # 角色 ID（建索引方便按角色统计）
    role_name = Column(String(64))  # 角色名（冗余字段，查询列表时少一次 join）
    query = Column(Text)  # 用户提问（Text 类型，长问题不截断）
    answer = Column(Text)  # 角色回答
    sources = Column(Text)  # 引用来源（JSON 字符串，存命中的文档来源/章节，便于回溯）
    created_at = Column(DateTime, default=datetime.utcnow)  # 对话时间


# ==================== CRUD 操作函数 ====================

def create_user(username: str, password_hash: str, nickname: str = "") -> User:
    """
    注册新用户，返回 User 对象

    参数：
        username：登录名，须全局唯一（重名会因唯一索引抛 IntegrityError）。
        password_hash：已经算好的密码哈希，由调用方生成；本函数不接触明文密码。
        nickname：昵称，留空时自动用 username 兜底（聊天界面显示更友好）。
    返回：
        User 对象；已 commit 并 refresh，因此 user.id 等数据库生成的值可直接使用。
    异常：
        重名或超长会抛 SQLAlchemy 异常并向上冒泡；但 close() 在 finally 里执行，
        连接一定归还连接池，不会因为异常导致连接泄漏。
    """
    session = SessionLocal()  # 创建会话
    try:  # 异常处理
        user = User(username=username, password_hash=password_hash, nickname=nickname or username)  # 构建对象
        session.add(user)  # 加入会话（此时只在内存，尚未发 SQL）
        session.commit()  # 提交（这一步才真正 INSERT 到 MySQL）
        session.refresh(user)  # 刷新获取自增 ID（保证拿到库里最终的值，而非 Python 侧猜测）
        logger.info(f"创建用户：{username}（id={user.id}）")  # 日志
        return user  # 返回
    finally:  # 无论成功失败
        session.close()  # 关闭会话（归还连接池）


def get_user_by_name(username: str) -> Optional[User]:
    """
    按用户名查用户

    参数：
        username：登录名（命中唯一索引，查询很快）。
    返回：
        User 对象；用户不存在时返回 None（登录流程据此判定「用户名或密码错误」）。
    """
    session = SessionLocal()  # 创建会话
    try:
        return session.query(User).filter(User.username == username).first()  # 查询（first() 只取第一条，避免多结果报错）
    finally:
        session.close()


def get_role(role_id: int) -> Optional[Role]:
    """
    按 ID 查角色

    参数：
        role_id：角色主键（来自请求参数或前端选择）。
    返回：
        Role 对象（含 system_prompt、temperature 等生成参数）；不存在时返回 None，
        调用方需自行判断并返回 404 之类的提示。
    """
    session = SessionLocal()
    try:
        return session.query(Role).filter(Role.id == role_id).first()
    finally:
        session.close()


def list_roles() -> list:
    """
    列出所有可用角色

    返回：
        List[Role]，全部角色对象（按主键顺序）。
    说明：
        过滤条件写成 Role.id > 0 而非直接 all()，是为了保留「手动下线不想要的
        角色」时改条件的余地，同时语义上排除异常主键。
    """
    session = SessionLocal()
    try:
        return session.query(Role).filter(Role.id > 0).all()  # 返回所有角色
    finally:
        session.close()


def save_conversation(user_id: int, role_id: int, role_name: str, query: str, answer: str, sources: str):
    """
    保存一条对话记录到 MySQL（审计 + 调试用）

    参数：
        user_id / role_id：发起对话的用户与角色。
        role_name：角色名，冗余存一份便于直接看日志。
        query：用户原始提问。
        answer：模型最终回答（已剥离思考过程与引用标记的纯文本）。
        sources：引用来源，调用方需先 json.dumps 成字符串再传入。
    返回：
        None。
    说明：
        写日志失败不应影响已被用户看到的回答，因此若需要强容错，由调用方决定是否
        try/except；本函数内部只保证 finally 里关闭 session。
    """
    session = SessionLocal()
    try:
        log = ConversationLog(  # 构建日志对象
            user_id=user_id, role_id=role_id, role_name=role_name,
            query=query, answer=answer, sources=sources,
        )
        session.add(log)
        session.commit()
        logger.info(f"对话已记录：user={user_id} role={role_name} query={query[:30]}...")  # 只截前 30 字打日志，避免日志膨胀
    finally:
        session.close()