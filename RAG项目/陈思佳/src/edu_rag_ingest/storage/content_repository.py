from __future__ import annotations

"""教案和试题的 SQLAlchemy 数据模型及持久化操作。"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .database import default_database_url, ensure_sqlite_parent


class Base(DeclarativeBase):
    pass


class LessonPlan(Base):
    __tablename__ = "lesson_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(64), default="语文")
    grade: Mapped[str] = mapped_column(String(64), default="九年级")
    chapter: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GeneratedQuestionSet(Base):
    __tablename__ = "question_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(64), default="语文")
    grade: Mapped[str] = mapped_column(String(64), default="九年级")
    chapter: Mapped[str] = mapped_column(String(128))
    knowledge_point: Mapped[str] = mapped_column(String(128))
    question_type: Mapped[str] = mapped_column(String(64))
    difficulty: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Database:
    def __init__(self, url: str | None = None, echo: bool = False) -> None:
        self.url = url or default_database_url()
        ensure_sqlite_parent(self.url)
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        self.engine = create_engine(self.url, echo=echo, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self):
        return self.session_factory()


def serialize_lesson_plan(item: LessonPlan, citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": item.id,
        "subject": item.subject,
        "grade": item.grade,
        "chapter": item.chapter,
        "title": item.title,
        "content": item.content,
        "citations": citations,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def serialize_question_set(item: GeneratedQuestionSet, citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": item.id,
        "subject": item.subject,
        "grade": item.grade,
        "chapter": item.chapter,
        "knowledge_point": item.knowledge_point,
        "question_type": item.question_type,
        "difficulty": item.difficulty,
        "content": item.content,
        "citations": citations,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }
