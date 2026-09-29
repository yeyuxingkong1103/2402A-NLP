"""角色加载：从 MySQL roles 表读入内存 dict，供 rag.py / api.py 查询。

启动时 load_roles() 一次加载；不做热加载，重启服务刷新。
"""
import logging

import db

logger = logging.getLogger(__name__)

# name -> {id, name, persona, collection_name}，按 id 升序插入
_roles: dict[str, dict] = {}


def load_roles() -> None:
    """从 roles 表全量加载到内存 dict（按 id 升序）。失败降级为空缓存，不阻断启动。"""
    global _roles
    try:
        with db.get_db() as session:
            rows = session.query(db.Role).order_by(db.Role.id).all()
        _roles = {
            r.name: {
                "id": r.id,
                "name": r.name,
                "persona": r.persona,
                "collection_name": r.collection_name,
            }
            for r in rows
        }
        logger.info("已加载 %d 个角色：%s", len(_roles), list(_roles))
    except Exception as e:  # noqa: BLE001
        logger.warning("角色加载失败，降级为空缓存（默认角色回退）: %s", e)


def get_role(name: str) -> dict | None:
    """按名字查角色，不存在返回 None。"""
    return _roles.get(name)


def list_roles() -> list[dict]:
    """返回全部角色（按 id 升序）。"""
    return list(_roles.values())


def first_role() -> dict | None:
    """返回第一个角色（默认角色）；无角色时返回 None。"""
    return next(iter(_roles.values()), None)
