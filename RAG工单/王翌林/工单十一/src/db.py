# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/db.py — MySQL 连接管理
"""
import os
from typing import Optional
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

class Base(DeclarativeBase): pass

_engine = None
_SessionLocal = None

def _build_url(database=None) -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3307"))
    user = os.getenv("MYSQL_USER", "root")
    pwd = os.getenv("MYSQL_PASSWORD", "355359")
    db = database or os.getenv("MYSQL_DATABASE", "rag_pdf_qa")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4&connect_timeout=10"

def get_engine(database=None) -> Engine:
    global _engine
    if _engine is None or database is not None:
        _engine = create_engine(_build_url(database), pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)
    return _engine

def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)
    return _SessionLocal()

def create_database_if_not_exists(database=None) -> bool:
    db_name = database or os.getenv("MYSQL_DATABASE", "rag_pdf_qa")
    url = _build_url(database="mysql")
    tmp_engine = create_engine(url, pool_pre_ping=True)
    try:
        with tmp_engine.connect() as conn:
            exists = conn.execute(text(
                "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = :name"), {"name": db_name}).scalar()
            if not exists:
                conn.execute(text(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit(); logger.info(f"已创建数据库: {db_name}")
            else: logger.info(f"数据库已存在: {db_name}")
        return True
    except Exception as e:
        logger.error(f"创建数据库失败: {e}"); return False
    finally: tmp_engine.dispose()

def create_tables():
    from src import models
    Base.metadata.create_all(get_engine())
    logger.info("✅ 建表完成")

def drop_tables():
    from src import models
    Base.metadata.drop_all(get_engine())
    logger.info("🗑️  删表完成")

def test_connection() -> Optional[str]:
    try:
        with get_engine().connect() as conn:
            return conn.execute(text("SELECT VERSION()")).scalar()
    except Exception as e:
        logger.error(f"MySQL 连接失败: {e}"); return None
