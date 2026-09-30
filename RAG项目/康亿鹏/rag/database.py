"""数据库连接、角色与用户模型（SQLAlchemy + MySQL）。"""  # 模块说明
from datetime import datetime  # 时间戳

from sqlalchemy import (  # SQLAlchemy ORM 组件
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker  # ORM 基类与会话工厂

import config  # 全局配置

# MySQL：通过 SQLAlchemy 连接，参数从 config.py 读取
engine = create_engine(
    f"mysql+pymysql://{config.mysql_user}:{config.mysql_password}"
    f"@{config.mysql_host}:{config.mysql_port}/{config.mysql_db}?charset=utf8mb4",
    echo=False,
)
SessionLocal = sessionmaker(bind=engine)  # 会话工厂：生成数据库会话


class Base(DeclarativeBase):  # 声明式基类
    pass  # SQLAlchemy 2.0 风格


class Role(Base):  # 角色表模型：角色名 + LLM 人设
    __tablename__ = "roles"  # 表名

    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键
    name = Column(String(32), unique=True, nullable=False)  # 角色名（doctor/teacher/admin，唯一）
    persona = Column(Text, nullable=False)  # 该角色的 LLM 人设描述（system prompt 前缀）


class User(Base):  # 用户表模型
    __tablename__ = "users"  # 表名

    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增主键
    username = Column(String(64), unique=True, nullable=False)  # 用户名（唯一）
    password_hash = Column(String(256), nullable=False)  # 密码哈希（bcrypt）
    created_at = Column(DateTime, default=datetime.now)  # 创建时间

    def to_dict(self):  # 转为字典（不含密码）
        return {  # 返回用户信息（不含密码哈希）
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


_seed_roles = {  # 预置角色 -> 人设（首次建表时写入 roles 表）
    "doctor": "你是一位资深临床医生，用专业但通俗的方式回答",  # 医生
    "teacher": "你是一位经验丰富的教师，用启发式的方式回答",  # 教师
    "admin": "你是一位知识渊博的综合专家，用严谨全面的方式回答",  # 管理员
}


def init_db():  # 初始化数据库：建表并预置角色（已存在则跳过）
    """首次运行时调用，创建 roles/users 表并写入预置角色。"""
    Base.metadata.create_all(engine)  # CREATE TABLE IF NOT EXISTS
    db = SessionLocal()  # 开会话写预置角色
    try:
        for name, persona in _seed_roles.items():  # 逐个角色检查
            exists = db.query(Role).filter(Role.name == name).first()  # 是否已存在
            if not exists:  # 不存在才插入，保证幂等
                db.add(Role(name=name, persona=persona))  # 新增角色记录
        db.commit()  # 提交
    finally:
        db.close()  # 关闭会话


def get_role_persona(role_name: str):  # 按角色名查人设（供 rag_chain 使用）
    """返回角色对应的人设文本；角色不存在时返回 None。"""
    db = SessionLocal()  # 独立会话（调用方不持有 db）
    try:
        role = db.query(Role).filter(Role.name == role_name).first()  # 查角色
        return role.persona if role else None  # 返回人设或 None
    finally:
        db.close()  # 关闭会话


def get_db():  # FastAPI 依赖：获取数据库会话（用完自动关闭）
    db = SessionLocal()  # 创建会话
    try:
        yield db  # 交给路由使用
    finally:
        db.close()  # 确保关闭
