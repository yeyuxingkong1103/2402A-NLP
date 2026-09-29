"""初始迁移：创建全部业务表。

Revision ID: 0001_init
Revises:
Create Date: 2026-09-15
"""
from alembic import op

from src.models import Base

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)