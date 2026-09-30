# -*- coding: utf-8 -*-
"""vector_store/filters.py —— 过滤表达式构造与 Milvus 返回记录的整形。

在链路中的位置：
    为 vector_store/ops.py 与 backend/pipeline.py、backend/retrieval.py 提供两类纯函数：
        过滤表达式构造  equal_filter / and_filter / _escape
        记录结构整形   _payload_from_row
    本文件不访问网络、不持有连接，因此可以在没有 Milvus 的环境里被单独测试 ——
    tests/test_vector_store.py 正是靠这些函数验证转义与 payload 摊平的正确性。
"""
from __future__ import annotations

from typing import Any

def _escape(value: str) -> str:
    """转义过滤表达式里的字符串字面量。

    参数：
        value: 待转义的值
    返回：
        反斜杠和双引号已转义的值。

    为什么必须转义：
        过滤表达式是一段被 Milvus 解析的字符串。文件名里若含双引号，
        不转义就会提前闭合字面量、让后面的内容被当成表达式解析 ——
        轻则报错，重则过滤条件被改写（删错数据）。
        先转义反斜杠再转义引号，顺序不能反，否则会把刚加的反斜杠又转一遍。
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def equal_filter(field: str, value: Any) -> str:
    """构造 "字段 == 值" 的过滤表达式。

    参数：
        field: 字段名（如 "source"）
        value: 比较值
    返回：
        Milvus 过滤表达式字符串。

    按类型分别处理字面量：
        布尔 -> true/false（不能写成 Python 的 True/False，Milvus 不认）
        数字 -> 不加引号
        其他 -> 加双引号并转义
        类型判断必须在前，否则数字会被当成字符串加上引号、比较结果恒为假。
    """
    if isinstance(value, bool):
        literal = "true" if value else "false"
    elif isinstance(value, (int, float)):
        literal = str(value)
    else:
        literal = f'"{_escape(str(value))}"'
    return f"{field} == {literal}"


def and_filter(*expressions: str) -> str:
    """用 and 组合多个过滤条件。

    参数：
        *expressions: 若干过滤表达式，允许混入空串
    返回：
        形如 "(a) and (b)" 的表达式；全部为空时返回空串（表示不过滤）。

    每个条件都加括号：
        过滤表达式里 and/or 有优先级，不加括号时组合条件容易被解析成非预期的结构。
        空串过滤掉是为了让调用方能放心传"可选条件"。
    """
    values = [expression for expression in expressions if expression]
    return " and ".join(f"({expression})" for expression in values)


def _payload_from_row(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """从 Milvus 返回的一行里拆出 (主键, 业务字段)。

    参数：
        row: query/search 返回的一行原始数据
    返回：
        (point_id 字符串, payload 字典)

    为什么写这么绕：
        不同版本、不同调用方式下 Milvus 的返回结构不一样 ——
        有的把业务字段包在 "entity" 里，有的是平铺的。
        这里统一成两种分支处理，并把 vector/distance/score 这些非业务字段剔掉，
        让上层拿到的 payload 永远只有"来源、页码、章节、正文"这些干净字段。
    """
    entity = row.get("entity")
    if isinstance(entity, dict):
        # 嵌套结构：主键在外层，业务字段在 entity 里
        payload = dict(entity)
        point_id = row.get("id", row.get("pk", payload.pop("id", "")))
    else:
        # 平铺结构：主键和业务字段混在一起，要手工剔除向量和分数
        payload = dict(row)
        point_id = payload.pop("id", payload.pop("pk", ""))
        payload.pop("vector", None)
        payload.pop("distance", None)
        payload.pop("score", None)
    return str(point_id), payload
