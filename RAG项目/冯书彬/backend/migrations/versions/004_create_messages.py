"""create messages

Revision ID: 004_create_messages
Revises: 003_create_conversations
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "004_create_messages"
down_revision = "003_create_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 消息正文和回答均加密存储，状态字段供编排和删除流程查询。
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id"), nullable=False, index=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("encrypted_user_text", sa.Text(), nullable=False),
        sa.Column("encrypted_answer", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, index=True),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("regeneration_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("corrected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source_message_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("messages")
