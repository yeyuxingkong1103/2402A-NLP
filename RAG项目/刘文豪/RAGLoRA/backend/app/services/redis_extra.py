# -*- coding: utf-8 -*-
"""Redis 其余四种数据类型的业务用法（与 memory.py 的 LIST 合计覆盖 5/5）。

    String  rl:chat:{user_id}      每用户问答限流计数器（INCR + EXPIRE 60s）
    Hash    char:{id}              角色详情 read-through 缓存（改角色时失效）
    Set     char_cat:{category}    角色分类二级索引（启动时幂等重建）
    zSet    rank:char_hot          角色热度榜（每轮问答 ZINCRBY 一次）

设计原则与 memory.py 一致：Redis 不可用时**全部降级**，绝不让缓存/榜单/
限流这类增强能力变成主链路的故障点——限流 fail-open，其余退回 MySQL
或返回空。客户端复用 memory.get_client()（继承 GBK 解码容错与连接参数）。
"""
import json

from ..core import config
from ..core.logging import get_logger
from .memory import get_client

log = get_logger("redis_extra")

_unavailable_logged = False

# zSet 榜单键名抽成常量：测试需要换成隔离键，避免污染真实热度数据。
RANK_KEY = "rank:char_hot"


def _warn_once(e: Exception) -> None:
    global _unavailable_logged
    if not _unavailable_logged:
        log.warning("Redis 不可用，扩展功能降级（限流放行/缓存回源/索引回源）: %s",
                    str(e)[:120])
        _unavailable_logged = True


# ================================================================ String：限流
def check_rate_limit(user_id: int, limit: int | None = None) -> tuple[bool, int]:
    """每用户每分钟问答次数限制。返回 (是否放行, 建议重试等待秒数)。

    - 计数键 ``rl:chat:{user_id}``，INCR 后首次设 60s 过期，窗口自然滚动。
    - **fail-open**：Redis 不可用时放行。限流是保护措施，不能反过来
      变成「Redis 一挂所有用户都不能聊天」的故障点。
    - TTL 防御：若进程在 INCR 与 EXPIRE 之间被杀，键会变成无过期，
      该用户将被永久限流——每次调用检查 ttl<0 并补设，杜绝这种情况。
    """
    limit = config.RATE_LIMIT_PER_MIN if limit is None else limit
    if limit <= 0:
        return True, 0
    key = f"rl:chat:{user_id}"
    try:
        client = get_client()
        n = client.incr(key)
        if n == 1 or client.ttl(key) < 0:
            client.expire(key, 60)
        if n <= limit:
            return True, 0
        return False, max(client.ttl(key), 1)
    except Exception as e:
        _warn_once(e)
        return True, 0


# ================================================================ Hash：角色缓存
# 需要进缓存的字段与 CharacterOut 对齐；style_json 是 JSON 列，HSET 前需序列化。
_CACHE_FIELDS = (
    "id", "slug", "name", "category", "avatar", "description",
    "identity_block", "style_json", "domain_constraints", "prompt_template",
    "kb_collection", "is_builtin", "recall_top_k", "rerank_top_k", "temperature",
)


def write_character_cache(ch: dict) -> bool:
    """把角色字段写入 Hash 缓存。MySQL 为权威源，此处只做扁平化。"""
    mapping = {}
    for f in _CACHE_FIELDS:
        v = ch.get(f)
        if f == "style_json" and v is not None:
            v = json.dumps(v, ensure_ascii=False)
        if v is None:
            continue  # HSET 不接受 None，缺失字段读取时自然回 None
        if f == "is_builtin":
            v = "1" if v else "0"
        mapping[f] = v
    if not mapping:
        return False
    try:
        key = f"char:{ch['id']}"
        client = get_client()
        client.hset(key, mapping=mapping)
        client.expire(key, config.CHAR_CACHE_TTL)
        return True
    except Exception as e:
        _warn_once(e)
        return False


def read_character_cache(character_id: int) -> dict | None:
    """读角色缓存，未命中返回 None。类型还原到 CharacterOut 的口径。"""
    try:
        raw = get_client().hgetall(f"char:{character_id}")
    except Exception as e:
        _warn_once(e)
        return None
    if not raw:
        return None
    out = {f: None for f in _CACHE_FIELDS}
    out.update(raw)
    out["id"] = int(out["id"])
    out["is_builtin"] = out["is_builtin"] == "1"
    for f in ("recall_top_k", "rerank_top_k"):
        if out[f] is not None:
            out[f] = int(out[f])
    if out["temperature"] is not None:
        out["temperature"] = float(out["temperature"])
    if out["style_json"]:
        try:
            out["style_json"] = json.loads(out["style_json"])
        except json.JSONDecodeError:
            out["style_json"] = None
    return out


def invalidate_character_cache(character_id: int) -> None:
    """角色被修改/删除后清缓存。失败仅告警——过期 TTL 是最终兜底。"""
    try:
        get_client().delete(f"char:{character_id}")
    except Exception as e:
        _warn_once(e)


def get_character_cached(character_id: int, db):
    """read-through：先 Hash，未命中回源 MySQL 并回填。返回 dict 或 None。"""
    cached = read_character_cache(character_id)
    if cached is not None:
        return cached
    from ..models import Character
    ch = db.get(Character, character_id)
    if not ch:
        return None
    data = {f: getattr(ch, f) for f in _CACHE_FIELDS}
    write_character_cache(data)
    return data


# ================================================================ Set：分类索引
def index_character(character_id: int, category: str | None) -> bool:
    """把角色 id 加入分类集合（二级索引）。category 为空则忽略。"""
    if not category:
        return False
    try:
        get_client().sadd(f"char_cat:{category}", character_id)
        return True
    except Exception as e:
        _warn_once(e)
        return False


def unindex_character(character_id: int, category: str | None) -> None:
    try:
        get_client().srem(f"char_cat:{category}", character_id)
    except Exception as e:
        _warn_once(e)


def category_ids(category: str) -> list[int] | None:
    """按分类取角色 id 列表。返回 None 表示 Redis 不可用（调用方回源 MySQL）。"""
    try:
        return [int(x) for x in get_client().smembers(f"char_cat:{category}")]
    except Exception as e:
        _warn_once(e)
        return None


def rebuild_category_index(db=None) -> int:
    """全量重建分类索引（幂等）。启动时调用，自愈「加了角色但索引漏更」。

    先删所有 char_cat:* 再重建——比 diff 增量简单且不可能出错；
    角色表就几十行，全量重建成本可忽略。Redis 不可用时返回 0 并告警，
    路由层查询会自动回源 MySQL，不影响功能。
    """
    own_session = db is None
    if own_session:
        from ..core.db import SessionLocal
        db = SessionLocal()
    try:
        try:
            client = get_client()
            keys = list(client.scan_iter(match="char_cat:*"))
            if keys:
                client.delete(*keys)
        except Exception as e:
            _warn_once(e)
            return 0

        from ..models import Character
        n = 0
        for ch in db.query(Character).filter(Character.category.isnot(None)).all():
            if index_character(ch.id, ch.category):
                n += 1
        return n
    finally:
        if own_session:
            db.close()


# ================================================================ zSet：热度榜
def bump_character_heat(character_id: int, delta: float = 1.0) -> None:
    """每完成一轮问答给角色 +1 热度。失败仅告警，不影响问答主流程。"""
    try:
        get_client().zincrby(RANK_KEY, delta, character_id)
    except Exception as e:
        _warn_once(e)


def hot_characters(top_n: int = 10) -> list[tuple[int, float]]:
    """热度榜前 N：[(character_id, score)]，按热度降序。Redis 不可用时为空。"""
    try:
        raw = get_client().zrevrange(RANK_KEY, 0, top_n - 1, withscores=True)
        return [(int(mid), float(score)) for mid, score in raw]
    except Exception as e:
        _warn_once(e)
        return []
