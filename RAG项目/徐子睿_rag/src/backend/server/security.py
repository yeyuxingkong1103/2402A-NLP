# -*- coding: utf-8 -*-
"""server/security.py —— 上传与删除接口的文件名校验。

在链路中的位置：
    routes_build.py 的上传、routes_kb.py 的删除都先调它。

为什么必须校验：
    文件名会直接参与拼接落盘路径。不校验的话，
    传 "../../etc/passwd" 这类名字就等于把"文件写到哪"的决定权交给客户端 ——
    这是路径穿越漏洞，属于必须拦住的输入。
"""
from __future__ import annotations

from pathlib import Path

def pdf_name(value: str) -> str | None:
    """校验并规范化上传的文件名。

    参数：
        value: 客户端传来的原始文件名
    返回：
        合法的纯文件名；不合法时返回 None。

    安全校验（不可省）：
        拒绝含 "/" 或 "\\" 的名字，防止 "../../etc/passwd" 这类路径穿越 ——
        上传接口的文件名直接参与拼路径，不校验就等于把文件写入位置交给客户端决定。
        另外只接受 .pdf 扩展名，不符合的直接拒绝。
    """
    value = (value or "").strip()
    if not value or "/" in value or "\\" in value:
        return None
    name = Path(value).name
    return name if name.lower().endswith(".pdf") else None
