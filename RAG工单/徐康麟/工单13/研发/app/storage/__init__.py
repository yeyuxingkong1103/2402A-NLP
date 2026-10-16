# -*- coding: utf-8 -*-
"""工单4 存储包：问答运行时持久化（``sqlite3`` 标准库）。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

* ``sqlite_manager.SqliteManager`` —— 问答历史 / 会话 / 答案级缓存；
* 与 T4 的知识库 ``研发/data/index/rag.sqlite3`` **分库**（问答侧写 ``qa_runtime.sqlite3``），
  避免两条链路互相覆盖。
"""

from .sqlite_manager import HistoryRow, SqliteManager, cache_key

__all__ = ["SqliteManager", "HistoryRow", "cache_key"]
