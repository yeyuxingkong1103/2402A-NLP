# -*- coding: utf-8 -*-
"""账号偏好的响应序列化 —— 从 ``api/app.py`` 拆出。

为什么单独一个小模块：这个函数的**字段白名单**是契约（`AC-PR-1` 规定响应字段集合
⊆ {theme, role_id} + 必要元数据），与路由放在一起容易被顺手加字段。
集中在这里、并保留原注释，是让"不许加字段"这件事一眼可见。
"""
from __future__ import annotations

from ...roles import DEFAULT_ROLE_ID

__all__ = ["prefs_payload"]


def prefs_payload(user_id: str, row: dict | None) -> dict:
    """把库里的偏好转成响应（**字段集合 ⊆ {theme, role_id} + 必要元数据**，`AC-PR-1`）。

    * ``theme``：``null`` = **未显式选择 = 跟随系统**（不折叠成 dark/light，`AC-PR-2`）；
    * ``role_id``：**当前生效值**（未设置时 = 默认角色 `lawyer`，`AC-L-22` 的默认约定）；
    * ``role_id_is_default``：区分"没设过"与"显式选了 lawyer"；
    * ``updated_at``：元数据（最近写入时间；未设置过 = 0）。
    """
    theme = None if row is None else row.get("theme")
    stored_role = "" if row is None else str(row.get("role_id") or "")
    return {
        "user_id": user_id,
        "theme": theme if theme in ("dark", "light") else None,
        "role_id": stored_role or DEFAULT_ROLE_ID,
        "role_id_is_default": not stored_role,
        "updated_at": 0.0 if row is None else float(row.get("updated_at") or 0.0),
    }