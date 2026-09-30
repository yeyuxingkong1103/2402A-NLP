"""批次7-1：create_admin CLI 单元测试。

覆盖用户要求的三类分支：
- 已注册邮箱 → 升级为管理员成功
- 未注册邮箱 → 明确报错，不静默创建账号
- 重复执行 → 幂等提示，不报错

核心逻辑抽为 promote_to_admin（可注入会话工厂，测试用 SQLite），
CLI 入口只做参数解析与输出，保证可测性。
"""

import pytest

from conftest import create_test_session
from app.cli.create_admin import promote_to_admin
from app.db.sql_models import User


def _seed_user(session, email: str, is_admin: bool = False) -> None:
    session.add(
        User(
            email=email,
            user_key="a" * 32,
            password_hash="x" * 64,
            is_admin=is_admin,
            is_active=True,
        )
    )
    session.commit()


def test_promote_existing_user_succeeds() -> None:
    """已注册邮箱 → is_admin 置 1，返回升级成功标记。"""
    session = create_test_session()
    _seed_user(session, "someone@example.com")

    status = promote_to_admin(session, "someone@example.com")

    assert status == "promoted"
    record = session.query(User).filter_by(email="someone@example.com").one()
    assert record.is_admin is True


def test_promote_unregistered_email_raises() -> None:
    """未注册邮箱 → 报错且不创建任何新用户行。"""
    session = create_test_session()
    before = session.query(User).count()

    with pytest.raises(ValueError, match="尚未注册"):
        promote_to_admin(session, "ghost@example.com")

    assert session.query(User).count() == before


def test_promote_repeat_is_idempotent() -> None:
    """重复执行 → 幂等提示，不报错，状态不变。"""
    session = create_test_session()
    _seed_user(session, "someone@example.com", is_admin=True)

    status = promote_to_admin(session, "someone@example.com")

    assert status == "already_admin"
    record = session.query(User).filter_by(email="someone@example.com").one()
    assert record.is_admin is True


def test_promote_email_is_case_insensitive() -> None:
    """邮箱大小写归一：注册为小写，CLI 传大写也要能命中。"""
    session = create_test_session()
    _seed_user(session, "someone@example.com")

    status = promote_to_admin(session, "SOMEONE@EXAMPLE.COM")

    assert status == "promoted"
