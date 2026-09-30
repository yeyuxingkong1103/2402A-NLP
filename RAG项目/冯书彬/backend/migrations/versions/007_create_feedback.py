"""create feedback

Revision ID: 007_create_feedback
Revises: 006_create_documents
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "007_create_feedback"
down_revision = "006_create_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 反馈不复制回答原文，只通过消息 ID 关联既有数据。
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("message_id", sa.String(length=36), sa.ForeignKey("messages.id"), nullable=False, index=True),
        sa.Column("rating", sa.String(length=16), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False, index=True),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("alert_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )
    op.create_table(
        "feedback_alerts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("feedback_id", sa.String(length=36), sa.ForeignKey("feedback.id"), nullable=False, index=True),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("notification_sent", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table("feedback_alerts")
    op.drop_table("feedback")
