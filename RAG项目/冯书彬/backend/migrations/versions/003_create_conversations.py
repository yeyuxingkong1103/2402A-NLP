"""create conversations

Revision ID: 003_create_conversations
Revises: 002_create_roles
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "003_create_conversations"
down_revision = "002_create_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 会话只保存用户关联和编排状态，聊天正文单独存储。
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("follow_up_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("conversations")
