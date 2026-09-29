# -*- coding: utf-8 -*-
"""
MySQL 数据层：用户表、角色表、对话记录表
用 SQLAlchemy ORM，建表/增删改查全在这里
"""

from datetime import datetime  # 时间戳
from typing import Optional  # 可选类型标注
import sqlalchemy  # ORM 框架
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean  # 列类型
from sqlalchemy.orm import declarative_base, sessionmaker  # 基类 + 会话工厂

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB  # MySQL 连接配置
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# SQLAlchemy 连接串：mysql+pymysql://用户:密码@地址:端口/库名
CONN_STR = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"  # 用户密码
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"  # 地址端口库名
    f"?charset=utf8mb4"  # UTF-8 编码（支持 emoji）
)

Base = declarative_base()  # ORM 基类：所有表模型继承它
engine = create_engine(CONN_STR, echo=False, pool_pre_ping=True, pool_size=5)  # 引擎：连接池 5 个，自动 ping 检活
SessionLocal = sessionmaker(bind=engine, autoflush=False)  # 会话工厂：每次 CRUD 创建一个 session


def init_db():
    """启动时调用：如果表不存在则自动创建（CREATE TABLE IF NOT EXISTS）"""
    Base.metadata.create_all(engine)  # 建表（已存在则跳过）
    logger.info("MySQL 表结构初始化完成")  # 日志


# ==================== ORM 表模型 ====================

class User(Base):  # 用户表：多用户支持
    __tablename__ = "users"  # 表名
    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键
    username = Column(String(64), unique=True, nullable=False, index=True)  # 用户名：唯一、建索引
    password_hash = Column(String(256), nullable=False)  # 密码哈希（不存明文）
    nickname = Column(String(64))  # 昵称：聊天中显示的名字
    created_at = Column(DateTime, default=datetime.utcnow)  # 注册时间
    is_active = Column(Boolean, default=True)  # 是否启用（软删除用）


class Role(Base):  # 角色表：多角色支持
    __tablename__ = "roles"  # 表名
    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键
    name = Column(String(64), unique=True, nullable=False, index=True)  # 角色名：律师/医生/心理医生...
    description = Column(String(256))  # 角色描述：一句话简介
    system_prompt = Column(Text, nullable=False)  # 系统提示词：定义角色人格、工作原则、输出格式
    temperature = Column(sqlalchemy.Float, default=0.5)  # 角色专属温度：律师 0.3 / 朋友 0.7
    knowledge_collection = Column(String(64))  # 该角色专属的 Milvus 集合名（空则用默认知识库）
    created_at = Column(DateTime, default=datetime.utcnow)  # 创建时间
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 修改时间


class ConversationLog(Base):  # 对话日志表：审计 + 调试
    __tablename__ = "conversation_logs"  # 表名
    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键
    user_id = Column(Integer, index=True)  # 用户 ID（建索引方便按用户查）
    role_id = Column(Integer, index=True)  # 角色 ID
    role_name = Column(String(64))  # 角色名（冗余字段，查询时少一次 join）
    query = Column(Text)  # 用户提问
    answer = Column(Text)  # 角色回答
    sources = Column(Text)  # 引用来源（JSON 字符串）
    created_at = Column(DateTime, default=datetime.utcnow)  # 对话时间


# ==================== CRUD 操作函数 ====================

def create_user(username: str, password_hash: str, nickname: str = "") -> User:
    """注册新用户，返回 User 对象"""
    session = SessionLocal()  # 创建会话
    try:  # 异常处理
        user = User(username=username, password_hash=password_hash, nickname=nickname or username)  # 构建对象
        session.add(user)  # 加入会话
        session.commit()  # 提交
        session.refresh(user)  # 刷新获取自增 ID
        logger.info(f"创建用户：{username}（id={user.id}）")  # 日志
        return user  # 返回
    finally:  # 无论成功失败
        session.close()  # 关闭会话（归还连接池）


def get_user_by_name(username: str) -> Optional[User]:
    """按用户名查用户"""
    session = SessionLocal()  # 创建会话
    try:
        return session.query(User).filter(User.username == username).first()  # 查询
    finally:
        session.close()


def get_role(role_id: int) -> Optional[Role]:
    """按 ID 查角色"""
    session = SessionLocal()
    try:
        return session.query(Role).filter(Role.id == role_id).first()
    finally:
        session.close()


def list_roles() -> list:
    """列出所有可用角色"""
    session = SessionLocal()
    try:
        return session.query(Role).filter(Role.id > 0).all()  # 返回所有角色
    finally:
        session.close()


def save_conversation(user_id: int, role_id: int, role_name: str, query: str, answer: str, sources: str):
    """保存一条对话记录到 MySQL（审计 + 调试用）"""
    session = SessionLocal()
    try:
        log = ConversationLog(  # 构建日志对象
            user_id=user_id, role_id=role_id, role_name=role_name,
            query=query, answer=answer, sources=sources,
        )
        session.add(log)
        session.commit()
        logger.info(f"对话已记录：user={user_id} role={role_name} query={query[:30]}...")
    finally:
        session.close()
