"""expand knowledge text columns

Revision ID: 010_expand_knowledge_text
Revises: 009_create_memories_exports
Create Date: 2026-09-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "010_expand_knowledge_text"
down_revision = "009_create_memories_exports"
branch_labels = None
depends_on = None


def _large_text_type():
    # MySQL 使用 LONGTEXT，SQLite 演练使用普通 TEXT 以保持迁移链可回归。
    if op.get_bind().dialect.name == "mysql":
        return mysql.LONGTEXT()
    return sa.Text()


def upgrade() -> None:
    # MinerU 解析后的法律材料可能达到数 MB，普通 TEXT 无法保存完整正文。
    large_text_type = _large_text_type()
    for table_name, column_name in (
        ("crawl_snapshots", "raw_text"),
        ("knowledge_materials", "raw_text"),
        ("document_chunks", "content"),
    ):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(column_name, type_=large_text_type, existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    # 回滚前需确认数据长度未超过 MySQL TEXT 上限，否则数据库会拒绝降级。
    large_text_type = _large_text_type()
    for table_name, column_name in (
        ("document_chunks", "content"),
        ("knowledge_materials", "raw_text"),
        ("crawl_snapshots", "raw_text"),
    ):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(column_name, type_=sa.Text(), existing_type=large_text_type, nullable=False)
