"""Canned reply used when nothing in the knowledge base is relevant.

Shared by the non-streaming and streaming chat paths so the two cannot drift.
"""

from __future__ import annotations

import os


def build_refusal_message() -> str:
    """Build the "not found in knowledge base" reply.

    联系方式从环境变量读。以前这段是写死在 chat_routes 里的占位符
    （400-888-9999 / service@example.com），而且因为拒答分支永远触发不了，
    一直没人发现它有机会被真的发给用户。
    """
    lines = [
        "非常抱歉，我在知识库中没有找到相关信息。您可以：",
        "1. 换个方式描述您的问题",
    ]

    hotline = os.getenv("SUPPORT_HOTLINE", "").strip()
    email = os.getenv("SUPPORT_EMAIL", "").strip()

    if hotline:
        lines.append(f"{len(lines)}. 联系人工客服：{hotline}")
    if email:
        lines.append(f"{len(lines)}. 发送邮件：{email}")

    return "\n".join(lines)
