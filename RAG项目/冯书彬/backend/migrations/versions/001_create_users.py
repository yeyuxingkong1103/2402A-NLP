"""create users and device sessions

Revision ID: 001_create_users
Revises:
Create Date: 2026-09-15
"""

from alembic import op
import sqlalchemy as sa

revision = "001_create_users"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users 表不保存明文手机号，只保存密文和 HMAC。
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("encrypted_phone", sa.Text(), nullable=False),
        sa.Column("phone_nonce", sa.String(length=128), nullable=False),
        sa.Column("phone_encrypted_data_key", sa.String(length=128), nullable=False),
        sa.Column("phone_key_version", sa.String(length=32), nullable=False),
        sa.Column("phone_hmac", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("agreement_version", sa.String(length=32), nullable=False),
        sa.Column("privacy_policy_version", sa.String(length=32), nullable=False),
        sa.Column("adult_confirmed", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    # device_sessions 表只保存 refresh token 摘要和设备元数据。
    op.create_table(
        "device_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("refresh_token_hash", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("device_identifier", sa.String(length=128), nullable=False),
        sa.Column("user_agent", sa.String(length=512), nullable=False),
        sa.Column("login_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("province_location", sa.String(length=64), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    # 回滚时先删子表，再删用户表。
    op.drop_table("device_sessions")
    op.drop_table("users")
