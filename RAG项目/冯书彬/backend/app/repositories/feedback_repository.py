from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, JSON, MetaData, String, Table, delete, insert, select, update
from sqlalchemy.engine import Engine

from backend.app.models.feedback import Feedback, FeedbackAlert


feedback_metadata = MetaData()

feedback_table = Table(
    "feedback",
    feedback_metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("message_id", String(36), nullable=False, index=True),
    Column("rating", String(16), nullable=False),
    Column("risk_level", String(16), nullable=False, index=True),
    Column("category", String(64), nullable=True),
    Column("reason", String(500), nullable=True),
    Column("alert_id", String(36), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
)

feedback_alerts_table = Table(
    "feedback_alerts",
    feedback_metadata,
    Column("id", String(36), primary_key=True),
    Column("feedback_id", String(36), ForeignKey("feedback.id"), nullable=False, index=True),
    Column("risk_level", String(16), nullable=False),
    Column("category", String(64), nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("notification_sent", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
)


def create_feedback_tables(engine: Engine) -> None:
    # 测试和本地验证可显式建表；生产环境使用 Alembic 迁移。
    feedback_metadata.create_all(engine, checkfirst=True)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _feedback_values(feedback: Feedback) -> dict[str, Any]:
    return {
        "id": feedback.id,
        "user_id": feedback.user_id,
        "message_id": feedback.message_id,
        "rating": feedback.rating,
        "risk_level": feedback.risk_level,
        "category": feedback.category,
        "reason": feedback.reason,
        "alert_id": feedback.alert_id,
        "created_at": feedback.created_at,
    }


def _row_to_feedback(row) -> Feedback | None:
    if row is None:
        return None
    data = row._mapping
    return Feedback(
        id=data["id"],
        user_id=data["user_id"],
        message_id=data["message_id"],
        rating=data["rating"],
        risk_level=data["risk_level"],
        category=data["category"],
        reason=data["reason"],
        alert_id=data["alert_id"],
        created_at=_as_utc(data["created_at"]),
    )


def _alert_values(alert: FeedbackAlert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "feedback_id": alert.feedback_id,
        "risk_level": alert.risk_level,
        "category": alert.category,
        "metadata": alert.metadata,
        "notification_sent": alert.notification_sent,
        "created_at": alert.created_at,
    }


def _row_to_alert(row) -> FeedbackAlert | None:
    if row is None:
        return None
    data = row._mapping
    return FeedbackAlert(
        id=data["id"],
        feedback_id=data["feedback_id"],
        risk_level=data["risk_level"],
        category=data["category"],
        metadata=dict(data["metadata"]),
        notification_sent=bool(data["notification_sent"]),
        created_at=_as_utc(data["created_at"]),
    )


class SQLAlchemyFeedbackRepository:
    # 反馈和高风险告警共享事务引擎；不保存额外聊天原文。
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save_feedback(self, feedback: Feedback) -> None:
        values = _feedback_values(feedback)
        with self.engine.begin() as connection:
            exists = connection.execute(select(feedback_table.c.id).where(feedback_table.c.id == feedback.id)).first()
            if exists:
                connection.execute(update(feedback_table).where(feedback_table.c.id == feedback.id).values(**values))
                return
            connection.execute(insert(feedback_table).values(**values))

    def save_alert(self, alert: FeedbackAlert) -> None:
        values = _alert_values(alert)
        with self.engine.begin() as connection:
            exists = connection.execute(select(feedback_alerts_table.c.id).where(feedback_alerts_table.c.id == alert.id)).first()
            if exists:
                connection.execute(update(feedback_alerts_table).where(feedback_alerts_table.c.id == alert.id).values(**values))
                return
            connection.execute(insert(feedback_alerts_table).values(**values))

    def get_feedback(self, feedback_id: str) -> Feedback | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(feedback_table).where(feedback_table.c.id == feedback_id)).first()
        return _row_to_feedback(row)

    def get_alert(self, alert_id: str) -> FeedbackAlert | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(feedback_alerts_table).where(feedback_alerts_table.c.id == alert_id)).first()
        return _row_to_alert(row)

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        with self.engine.begin() as connection:
            feedback_ids = select(feedback_table.c.id).where(feedback_table.c.user_id == user_id)
            alerts = connection.execute(delete(feedback_alerts_table).where(feedback_alerts_table.c.feedback_id.in_(feedback_ids))).rowcount
            feedback = connection.execute(delete(feedback_table).where(feedback_table.c.user_id == user_id)).rowcount
        return {"feedback": feedback, "alerts": alerts}
