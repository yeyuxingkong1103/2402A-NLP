"""expand document chunk ids for hierarchy

Revision ID: 013_expand_document_chunk_ids
Revises: 012_add_document_chunk_hierarchy
Create Date: 2026-09-23
"""

from alembic import op
import sqlalchemy as sa

revision = "013_expand_document_chunk_ids"
down_revision = "012_add_document_chunk_hierarchy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.alter_column("id", existing_type=sa.String(length=36), type_=sa.String(length=128), existing_nullable=False)


def downgrade() -> None:
    # 回滚前需确认所有 chunk id 长度不超过 36 字符。
    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.alter_column("id", existing_type=sa.String(length=128), type_=sa.String(length=36), existing_nullable=False)
