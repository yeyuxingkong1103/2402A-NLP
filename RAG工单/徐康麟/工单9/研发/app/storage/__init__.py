# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单4 存储包：问答运行时持久化（``sqlite3`` 标准库）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

* ``sqlite_manager.SqliteManager`` —— 问答历史 / 会话 / 答案级缓存；
* 与 T4 的知识库 ``研发/data/index/rag.sqlite3`` **分库**（问答侧写 ``qa_runtime.sqlite3``），
  避免两条链路互相覆盖。
"""

from .sqlite_manager import HistoryRow, SqliteManager, cache_key

__all__ = ["SqliteManager", "HistoryRow", "cache_key"]
