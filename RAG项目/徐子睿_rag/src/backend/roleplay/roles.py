# -*- coding: utf-8 -*-
"""roleplay/roles.py —— 角色卡的增删改查。

在链路中的位置：
    backend/server.py 的 /api/roleplay/roles 接口 → 【本文件】 → SQLite 的 roles 表
    chat.py 在开始对话前也会用 get_role() 取角色卡。

角色的存储方式：
    config_json 一列存整份角色配置的 JSON —— 以后给角色卡加字段不必改表结构；
    id/name/description 另存独立列，便于直接查询与排序。
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from .db import _db, _now

def _row_to_role(row: sqlite3.Row) -> dict[str, Any]:
    """把 roles 表的一行还原成角色字典。

    参数：
        row: roles 表的一行
    返回：
        角色字典。

    注意 config_json 里已经存了完整的角色配置（含 id/name/description），
    这里再用列值覆盖一遍：列是权威值（可能通过其他途径更新过），
    以列为准可以避免 JSON 与列不一致时读到过期数据。
    """
    role = json.loads(row["config_json"])
    role.update({"id": row["id"], "name": row["name"], "description": row["description"], "updated_at": row["updated_at"]})
    return role

def list_roles() -> list[dict[str, Any]]:
    """列出全部角色（内置 + 自定义）。

    返回：
        角色字典列表，按 id 排序保证顺序稳定。
    """
    rows = _db().execute("SELECT * FROM roles ORDER BY id").fetchall()
    return [_row_to_role(row) for row in rows]

def get_role(role_id: str) -> dict[str, Any] | None:
    """按 id 取角色。

    参数：
        role_id: 角色 id
    返回：
        角色字典；不存在时返回 None。
    """
    row = _db().execute("SELECT * FROM roles WHERE id=?", (role_id,)).fetchone()
    return _row_to_role(row) if row else None

def save_role(data: dict[str, Any]) -> dict[str, Any]:
    """保存（新建或覆盖）一个角色。

    参数：
        data: 角色配置，字段同角色卡四要素
    返回：
        保存后从库里读回的角色字典（保证返回的是真正落库的内容，而不是入参）。

    异常：
        id 清洗后为空 -> ValueError。

    安全处理：
        1. id 只保留字母数字下划线连字符，其余字符替换成下划线 ——
           角色 id 参与拼接和检索过滤，放任任意字符会带来注入和匹配问题
        2. 每个字段都做了长度截断（name 80 / description 300 / persona 3000 / style 1000 /
           safety_notice 1500）—— 这些字段最终会进提示词，不限制长度就等于
           让用户能随意塞满模型的上下文窗口，把系统指令挤掉
        3. 每个字段都给中文默认值，保证角色卡不会因为缺字段而让提示词结构塌掉

    UPSERT 写法（ON CONFLICT ... DO UPDATE）：
        用一条 SQL 完成"有则更新、无则插入"，避免先查后写的竞态。
        注意 created_at 不在更新列表里 —— 修改角色不应改变它的创建时间。
    """
    role_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(data.get("id") or "").strip())[:64]
    if not role_id:
        raise ValueError("角色 id 不能为空")
    role = {
        "id": role_id,
        "name": str(data.get("name") or role_id)[:80],
        "description": str(data.get("description") or "自定义角色")[:300],
        "persona": str(data.get("persona") or "你是一个可靠、诚实的角色扮演助手。")[:3000],
        "style": str(data.get("style") or "清晰、自然地回答。")[:1000],
        "safety_notice": str(data.get("safety_notice") or "不确定时明确说明，不编造事实。")[:1500],
        "knowledge_sources": list(data.get("knowledge_sources") or []),
    }
    now = _now()
    db = _db()
    db.execute(
        "INSERT INTO roles(id,name,description,config_json,created_at,updated_at) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET name=excluded.name,description=excluded.description,config_json=excluded.config_json,updated_at=excluded.updated_at",
        (role_id, role["name"], role["description"], json.dumps(role, ensure_ascii=False), now, now),
    )
    db.commit()
    return get_role(role_id) or role
