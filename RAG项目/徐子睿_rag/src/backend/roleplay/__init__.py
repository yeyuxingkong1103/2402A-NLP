# -*- coding: utf-8 -*-
"""roleplay —— 角色扮演（同名包）。

在链路中的位置：
    backend/server.py 的 /api/roleplay/* → 【本包】
        → SQLite（角色/用户/会话/消息）
        → Milvus（长期记忆向量）
        → backend/retrieval（知识检索）
        → Ollama（生成回答）

一句话理解这个包：它不是另一套系统，而是在 RAG 主链路上加了两样东西 ——
    「角色」决定用什么身份、什么风格说话；「记忆」让对话跨轮次连贯、跨会话可回忆。
    检索知识的部分完全复用 retrieval 包，没有重写一遍。

为什么改成了包：
    原 roleplay.py 有 720 行（其中注释 320 行），超出"单文件 300 行"的上限，
    也是 backend 里最大的一个文件。按职责拆成 config / db / roles / sessions /
    memory / prompt / llm / chat 八个模块后，每个文件都在 300 行以内。

关键设计：本文件把各子模块的公开名全部再导出，
    所以 `from roleplay import chat, list_roles, save_role, history` 之类用法
    与拆分前完全一致 —— server.py 一行都不用改。

需要同步调整的调用方（唯一一处）：
    tests/test_roleplay.py 直接改写模块全局（roleplay._DB / roleplay.ROLEPLAY_DB）
    并用 @patch("roleplay.call_llm") 之类打在模块属性上。
    拆包后这些名字分别落在 roleplay.db 与 roleplay.chat 中，
    测试的 patch 目标必须跟着指到"实际使用它们的模块"，详见该测试文件里的注释。

包内分工：
    config.py    配置常量与五个内置角色卡
    db.py        SQLite 连接与建表
    roles.py     角色 CRUD
    sessions.py  用户、会话、消息
    memory.py    长期记忆（Milvus）
    prompt.py    五层提示词与答案后处理
    llm.py       模型调用与降级兜底
    chat.py      一次角色对话的总流程
    本文件        再导出
"""
from __future__ import annotations

from .chat import chat
from .config import (
    BASE_DIR,
    DEFAULT_ROLES,
    MAX_MESSAGE_CHARS,
    MEMORY_COLLECTION,
    OLLAMA_CHAT_URL,
    ROLEPLAY_DB,
    ROLEPLAY_MODEL,
    SHORT_MEMORY_LIMIT,
)
from .db import _db, _now
from .llm import call_llm, fallback_answer
from .memory import remember_long_term, retrieve_long_term
from .prompt import build_messages, postprocess_answer
from .roles import get_role, list_roles, save_role
from .sessions import (
    append_message,
    create_session,
    ensure_user,
    get_session,
    history,
    list_sessions,
    recent_messages,
)



# 显式声明对外接口：这就是"拆包不改调用方"的契约清单。
# 注意 _db / _now 是下划线开头的内部名，但 tests/test_roleplay.py 会直接引用它们
# （改写 _DB、以及 _db() 的行为），所以一并保留在导出清单里。
__all__ = [
    "BASE_DIR",
    "DEFAULT_ROLES",
    "MAX_MESSAGE_CHARS",
    "MEMORY_COLLECTION",
    "OLLAMA_CHAT_URL",
    "ROLEPLAY_DB",
    "ROLEPLAY_MODEL",
    "SHORT_MEMORY_LIMIT",
    "_db",
    "_now",
    "append_message",
    "build_messages",
    "call_llm",
    "chat",
    "create_session",
    "ensure_user",
    "fallback_answer",
    "get_role",
    "get_session",
    "history",
    "list_roles",
    "list_sessions",
    "postprocess_answer",
    "recent_messages",
    "remember_long_term",
    "retrieve_long_term",
    "save_role",
]
