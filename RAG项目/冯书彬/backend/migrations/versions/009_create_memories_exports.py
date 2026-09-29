"""create memory and export tables

Revision ID: 009_create_memories_exports
Revises: 008_create_audit_logs
Create Date: 2026-09-17
"""

from alembic import op
import sqlalchemy as sa

revision = "009_create_memories_exports"
down_revision = "008_create_audit_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 长期记忆只保存加密事实和摘要，不落聊天原文。
    op.create_table(
        "memories",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("encrypted_facts", sa.Text(), nullable=False),
        sa.Column("encrypted_summary", sa.Text(), nullable=False),
        sa.Column("source_conversation_ids", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("prompt_required", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "memory_candidates",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("memory_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("user_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("new_value", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "memory_preferences",
        sa.Column("user_id", sa.String(length=36), primary_key=True),
        sa.Column("auto_extract_enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "export_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("encrypted_zip", sa.Text(), nullable=False),
        sa.Column("password_hmac", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("max_downloads", sa.Integer(), nullable=False),
        sa.Column("download_count", sa.Integer(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        sa.Column("excluded_from_backup", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("export_jobs")
    op.drop_table("memory_preferences")
    op.drop_table("memory_candidates")
    op.drop_table("memories")
