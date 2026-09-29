"""create knowledge base governance tables

Revision ID: 005_create_knowledge_bases
Revises: 004_create_messages
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "005_create_knowledge_bases"
down_revision = "004_create_messages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 来源白名单只保存来源治理元数据，不保存正文。
    op.create_table(
        "source_whitelist_entries",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("url", sa.String(length=512), nullable=False, unique=True, index=True),
        sa.Column("publisher", sa.String(length=255), nullable=False, index=True),
        sa.Column("material_type", sa.String(length=64), nullable=False, index=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true(), index=True),
        sa.Column("created_by", sa.String(length=36), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    # 抓取快照保存原文摘要、来源和附件元数据；生产实现应加密或隔离 raw_text。
    op.create_table(
        "crawl_snapshots",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_url", sa.String(length=512), nullable=False, index=True),
        sa.Column("publisher", sa.String(length=255), nullable=False, index=True),
        sa.Column("material_type", sa.String(length=64), nullable=False, index=True),
        sa.Column("raw_content_hash", sa.String(length=64), nullable=False, index=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("attachments", sa.JSON(), nullable=False),
        sa.Column("backup_source_urls", sa.JSON(), nullable=False),
        sa.Column("crawled_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("status", sa.String(length=32), nullable=False, index=True),
        sa.Column("searchable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("failure_reason", sa.String(length=1024), nullable=True),
    )
    # 知识材料从快照提交审核后生成，发布前不可检索。
    op.create_table(
        "knowledge_materials",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("snapshot_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("source_url", sa.String(length=512), nullable=False, index=True),
        sa.Column("publisher", sa.String(length=255), nullable=False, index=True),
        sa.Column("material_type", sa.String(length=64), nullable=False, index=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, index=True),
        sa.Column("searchable", sa.Boolean(), nullable=False, server_default=sa.false(), index=True),
        sa.Column("reviewed_by", sa.String(length=36), nullable=True, index=True),
        sa.Column("published_by", sa.String(length=36), nullable=True, index=True),
        sa.Column("effective_from", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "material_attachments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("material_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("url", sa.String(length=1024), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.create_table(
        "review_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("material_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("decision", sa.String(length=32), nullable=False, index=True),
        sa.Column("reason", sa.String(length=1024), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "publish_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("material_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("publisher_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "material_status_history",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("material_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("from_status", sa.String(length=32), nullable=True),
        sa.Column("to_status", sa.String(length=32), nullable=False, index=True),
        sa.Column("actor_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("reason", sa.String(length=1024), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    # 回滚按依赖反序删除治理表。
    op.drop_table("material_status_history")
    op.drop_table("publish_records")
    op.drop_table("review_records")
    op.drop_table("material_attachments")
    op.drop_table("knowledge_materials")
    op.drop_table("crawl_snapshots")
    op.drop_table("source_whitelist_entries")
