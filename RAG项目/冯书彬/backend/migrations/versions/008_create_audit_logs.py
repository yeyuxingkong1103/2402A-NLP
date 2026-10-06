"""create audit logs

Revision ID: 008_create_audit_logs
Revises: 007_create_feedback
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "008_create_audit_logs"
down_revision = "007_create_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 审计表只允许追加写入，业务服务不应提供更新或删除接口。
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("action", sa.String(length=64), nullable=False, index=True),
        sa.Column("actor_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
