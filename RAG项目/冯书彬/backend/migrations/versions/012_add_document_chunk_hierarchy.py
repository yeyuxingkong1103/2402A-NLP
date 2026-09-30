"""add document chunk hierarchy

Revision ID: 012_add_document_chunk_hierarchy
Revises: 011_expand_chat_external_ids
Create Date: 2026-09-23
"""

from alembic import op
import sqlalchemy as sa

revision = "012_add_document_chunk_hierarchy"
down_revision = "011_expand_chat_external_ids"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.add_column(sa.Column("parent_chunk_id", sa.String(length=128), nullable=True))
        batch_op.create_index("ix_document_chunks_parent_chunk_id", ["parent_chunk_id"])
        batch_op.add_column(sa.Column("chunk_level", sa.String(length=16), nullable=False, server_default="child"))
        batch_op.add_column(sa.Column("structure_type", sa.String(length=32), nullable=False, server_default="document"))


def downgrade() -> None:
    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.drop_index("ix_document_chunks_parent_chunk_id")
        batch_op.drop_column("structure_type")
        batch_op.drop_column("chunk_level")
        batch_op.drop_column("parent_chunk_id")
