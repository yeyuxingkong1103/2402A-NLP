# -*- coding: utf-8 -*-
"""Redis 五种数据类型的业务用法测试（memory.py 的 LIST 之外新增四类）。

本模块在守护什么
================

`app/services/redis_extra.py` 把四种 Redis 类型接到真实业务上，每类守护一条不变量：

  1. String（限流计数器）：限额内放行、超限拒绝且给重试秒数、
     **fail-open**——Redis 挂了必须放行，限流不能反过来变成全站故障点。
  2. Hash（角色缓存）：round-trip 类型还原（bool/int/float/dict）、
     read-through 只回源 MySQL 一次、失效后确实读不到。
  3. Set（分类索引）：加入/移除/查询正确；全量重建后与 MySQL 一致。
  4. zSet（热度榜）：同角色累加、跨角色按分数降序。

隔离策略：测试用 99000x 高位 id 与专用分类名，避免碰真实用户/角色数据；
zSet 榜单键 monkeypatch 成测试键，不污染真实热度。所有测试键 teardown 清理
（char_cat / rank 无 TTL，残留会永久留存，必须显式删）。

Redis 未运行时整模块 skip——这些测试验证的是「与 Redis 的正确协作」，
没有 Redis 时无事可测；服务本身的降级行为由 fail-open 用例单独覆盖。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import redis_extra  # noqa: E402

# 测试专用的隔离 id / 键。teardown 会全部删除。
_U = 990_001          # 限流测试用户
_CH = 990_002         # Hash 缓存测试角色
_CAT = "测试分类勿用"   # Set 测试分类
_RANK_KEY = "rank:test_hot"


@pytest.fixture(autouse=True)
def _require_redis_and_cleanup():
    from app.services import memory as memory_svc
    if not memory_svc.is_available():
        pytest.skip("Redis 未运行，跳过 Redis 数据类型测试")

    from app.services.memory import get_client
    keys = [f"rl:chat:{_U}", f"char:{_CH}", f"char_cat:{_CAT}", _RANK_KEY]
    get_client().delete(*keys)   # 先清：上次异常退出可能留下无 TTL 键
    yield
    get_client().delete(*keys)


def _boom():
    raise RuntimeError("redis down (模拟)")


# ================================================================ String 限流
def test_rate_limit_allows_within_quota():
    assert redis_extra.check_rate_limit(_U, limit=2) == (True, 0)
    assert redis_extra.check_rate_limit(_U, limit=2) == (True, 0)


def test_rate_limit_blocks_over_quota():
    redis_extra.check_rate_limit(_U, limit=2)
    redis_extra.check_rate_limit(_U, limit=2)
    allowed, retry_after = redis_extra.check_rate_limit(_U, limit=2)
    assert not allowed
    assert retry_after >= 1          # 给出可等待的秒数（TTL > 0）


def test_rate_limit_fails_open_when_redis_down(monkeypatch):
    """限流是保护措施：Redis 挂了必须放行，不能变成「全员禁止聊天」。"""
    monkeypatch.setattr(redis_extra, "get_client", _boom)
    assert redis_extra.check_rate_limit(_U, limit=2) == (True, 0)


# ================================================================ Hash 角色缓存
def _char_dict() -> dict:
    return {
        "id": _CH, "slug": "test-char", "name": "测试角色",
        "category": "测试", "avatar": "🧪", "description": None,
        "identity_block": "你是测试角色。", "style_json": {"tone": "简洁"},
        "domain_constraints": "不编造。", "prompt_template": "TPL",
        "kb_collection": "kb_medical", "is_builtin": False,
        "recall_top_k": 20, "rerank_top_k": 5, "temperature": 0.3,
    }


def test_character_cache_roundtrip_restores_types():
    """HSET 扁平化后读回必须还原 Python 类型，否则 CharacterOut 校验会炸。"""
    assert redis_extra.write_character_cache(_char_dict())
    out = redis_extra.read_character_cache(_CH)
    assert out["id"] == _CH and isinstance(out["id"], int)
    assert out["is_builtin"] is False
    assert out["style_json"] == {"tone": "简洁"}
    assert out["recall_top_k"] == 20 and out["rerank_top_k"] == 5
    assert out["temperature"] == 0.3
    assert out["description"] is None     # DB 里为 None 的字段读回也是 None


def test_character_cache_invalidate():
    redis_extra.write_character_cache(_char_dict())
    redis_extra.invalidate_character_cache(_CH)
    assert redis_extra.read_character_cache(_CH) is None


def test_character_cache_read_through_hits_mysql_once():
    """read-through：首次回源 MySQL 并回填，第二次直接走缓存。"""
    obj = SimpleNamespace(**_char_dict())

    class _FakeDB:
        def __init__(self):
            self.get_calls = 0

        def get(self, model, oid):
            self.get_calls += 1
            return obj if oid == obj.id else None

    db = _FakeDB()
    first = redis_extra.get_character_cached(_CH, db)
    second = redis_extra.get_character_cached(_CH, db)

    assert first["name"] == "测试角色"
    assert second["name"] == "测试角色"
    assert db.get_calls == 1          # 第二次命中缓存，MySQL 只碰一次


def test_character_cache_fails_open_to_none(monkeypatch):
    monkeypatch.setattr(redis_extra, "get_client", _boom)
    assert redis_extra.read_character_cache(_CH) is None


# ================================================================ Set 分类索引
def test_category_index_add_remove():
    assert redis_extra.index_character(_CH, _CAT) is True
    assert redis_extra.category_ids(_CAT) == [_CH]
    redis_extra.unindex_character(_CH, _CAT)
    assert redis_extra.category_ids(_CAT) == []


def test_category_index_none_category_ignored():
    assert redis_extra.index_character(_CH, None) is False


def test_rebuild_index_matches_mysql():
    """全量重建后，内置角色的分类索引必须与 MySQL 一致。"""
    from app.core.db import SessionLocal
    from app.models import Character
    try:
        db = SessionLocal()
        rows = db.query(Character).filter(Character.category.isnot(None)).all()
    except Exception:
        pytest.skip("MySQL 不可用，跳过重建一致性测试")
    finally:
        try:
            db.close()
        except Exception:
            pass
    if not rows:
        pytest.skip("角色表为空（服务从未启动种角色），跳过")

    n = redis_extra.rebuild_category_index(db)
    assert n >= len(rows)
    for ch in rows:
        assert ch.id in (redis_extra.category_ids(ch.category) or [])


# ================================================================ zSet 热度榜
def test_heat_rank_aggregates_and_sorts(monkeypatch):
    monkeypatch.setattr(redis_extra, "RANK_KEY", _RANK_KEY)
    redis_extra.bump_character_heat(990_003, 1.0)     # 1 次
    redis_extra.bump_character_heat(990_004, 3.0)     # 3 次
    redis_extra.bump_character_heat(990_004, 3.0)
    pairs = redis_extra.hot_characters(top_n=10)
    assert pairs == [(990_004, 6.0), (990_003, 1.0)]  # 聚合正确且降序


def test_heat_rank_empty_when_redis_down(monkeypatch):
    monkeypatch.setattr(redis_extra, "get_client", _boom)
    assert redis_extra.hot_characters() == []
