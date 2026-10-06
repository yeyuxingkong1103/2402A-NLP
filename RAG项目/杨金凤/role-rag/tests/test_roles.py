"""roles.py 单元测试：加载、按名查询、列表顺序、默认角色（mock db，不连 MySQL）。"""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import roles


def _make_role(id_, name, persona, collection):
    return SimpleNamespace(
        id=id_, name=name, persona=persona, collection_name=collection
    )


@pytest.fixture
def _load(monkeypatch):
    """把 db.get_db 替换为返回固定行集的假 contextmanager，并加载角色。"""
    rows = [
        _make_role(1, "高血压医生", "医生人设", "hypertension_guide"),
        _make_role(2, "心理医生", "心理人设", "psychology_guide"),
    ]
    session = MagicMock()
    session.query.return_value.order_by.return_value.all.return_value = rows

    @contextmanager
    def fake_get_db():
        yield session

    monkeypatch.setattr(roles.db, "get_db", fake_get_db)
    roles.load_roles()
    return rows


def test_load_roles_populates_cache(_load):
    assert len(roles.list_roles()) == 2


def test_get_role_hit(_load):
    assert roles.get_role("心理医生") == {
        "id": 2,
        "name": "心理医生",
        "persona": "心理人设",
        "collection_name": "psychology_guide",
    }


def test_get_role_miss_returns_none(_load):
    assert roles.get_role("不存在的角色") is None


def test_list_roles_id_order(_load):
    assert [r["name"] for r in roles.list_roles()] == ["高血压医生", "心理医生"]


def test_first_role(_load):
    assert roles.first_role()["name"] == "高血压医生"


def test_load_roles_failure_keeps_empty(monkeypatch):
    """DB 异常时降级为空缓存，get_role 返回 None。"""
    @contextmanager
    def boom():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(roles.db, "get_db", boom)
    roles._roles = {}  # 先清空，验证加载失败后仍为空
    roles.load_roles()
    assert roles.list_roles() == []
    assert roles.first_role() is None
