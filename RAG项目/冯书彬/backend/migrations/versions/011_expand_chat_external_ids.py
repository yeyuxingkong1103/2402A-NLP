"""expand chat external ids

Revision ID: 011_expand_chat_external_ids
Revises: 010_expand_knowledge_text
Create Date: 2026-09-21
"""

from alembic import op
import sqlalchemy as sa

revision = "011_expand_chat_external_ids"
down_revision = "010_expand_knowledge_text"
branch_labels = None
depends_on = None


def _is_mysql() -> bool:
    # MySQL 需要显式处理已有外键，SQLite batch 模式会自动重建约束。
    return op.get_bind().dialect.name == "mysql"


def upgrade() -> None:
    # 前端真实浏览器会生成带 web- 前缀的 UUID，会话 ID 需要超过 36 字符。
    if _is_mysql():
        with op.batch_alter_table("messages") as batch_op:
            batch_op.drop_constraint("messages_ibfk_1", type_="foreignkey")
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.alter_column("id", existing_type=sa.String(length=36), type_=sa.String(length=64), existing_nullable=False)
    with op.batch_alter_table("messages") as batch_op:
        batch_op.alter_column("conversation_id", existing_type=sa.String(length=36), type_=sa.String(length=64), existing_nullable=False)
        batch_op.alter_column("source_message_id", existing_type=sa.String(length=36), type_=sa.String(length=64), existing_nullable=True)
        if _is_mysql():
            batch_op.create_foreign_key("messages_ibfk_1", "conversations", ["conversation_id"], ["id"])


def downgrade() -> None:
    # 回滚前需确认不存在超过 36 字符的前端会话 ID。
    if _is_mysql():
        with op.batch_alter_table("messages") as batch_op:
            batch_op.drop_constraint("messages_ibfk_1", type_="foreignkey")
    with op.batch_alter_table("messages") as batch_op:
        batch_op.alter_column("source_message_id", existing_type=sa.String(length=64), type_=sa.String(length=36), existing_nullable=True)
        batch_op.alter_column("conversation_id", existing_type=sa.String(length=64), type_=sa.String(length=36), existing_nullable=False)
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.alter_column("id", existing_type=sa.String(length=64), type_=sa.String(length=36), existing_nullable=False)
    if _is_mysql():
        with op.batch_alter_table("messages") as batch_op:
            batch_op.create_foreign_key("messages_ibfk_1", "conversations", ["conversation_id"], ["id"])
