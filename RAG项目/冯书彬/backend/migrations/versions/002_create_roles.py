"""create roles

Revision ID: 002_create_roles
Revises: 001_create_users
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "002_create_roles"
down_revision = "001_create_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 角色名称作为唯一权限边界，避免业务层重复维护字符串。
    op.create_table(
        "roles",
        sa.Column("name", sa.String(length=64), primary_key=True),
        sa.Column("description", sa.String(length=255), nullable=False),
    )
    op.bulk_insert(
        sa.table("roles", sa.column("name", sa.String), sa.column("description", sa.String)),
        [
            {"name": "super_admin", "description": "超级管理员"},
            {"name": "content_reviewer", "description": "内容审核员"},
            {"name": "support_operator", "description": "客服运营员"},
        ],
    )


def downgrade() -> None:
    # 角色表无业务外键依赖时可直接回滚。
    op.drop_table("roles")
