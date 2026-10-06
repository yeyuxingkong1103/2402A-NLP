"""create documents

Revision ID: 006_create_documents
Revises: 005_create_knowledge_bases
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "006_create_documents"
down_revision = "005_create_knowledge_bases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 文档索引元数据与知识库治理解耦，向量内容由外部索引服务保存。
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("material_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False, index=True),
        sa.Column("status", sa.String(length=32), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("documents.id"), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citation", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
