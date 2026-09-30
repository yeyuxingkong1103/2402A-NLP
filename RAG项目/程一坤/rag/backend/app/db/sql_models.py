"""数据库实体的统一导入入口（本文件不定义任何表、不写任何逻辑）。

实体定义按域分置：
- 文档域（Document / DocumentVersion / DocumentChunk / CrawlRecord / ImportRecord）
  → app/db/document_models.py
- 法规域（Law / LawVersion / Article）→ app/db/law_models.py
- 用户域（User）→ app/db/user_models.py
- 聊天域（ChatSession / ChatMessage）→ app/db/chat_models.py（历史独立文件）
- 共用件（Base / utc_now）→ app/db/base.py

改动表结构时请到对应域文件修改；本文件只负责 re-export，
让既有调用方继续 `from app.db.sql_models import ...` 而不必改动。
"""

from app.db.base import Base, utc_now
from app.db.document_models import (
    ChunkSummary,
    CrawlRecord,
    Document,
    DocumentChunk,
    DocumentVersion,
    ImportRecord,
)
from app.db.law_models import Article, Law, LawVersion
from app.db.user_models import User

__all__ = [
    "Article",
    "Base",
    "ChunkSummary",
    "CrawlRecord",
    "Document",
    "DocumentChunk",
    "DocumentVersion",
    "ImportRecord",
    "Law",
    "LawVersion",
    "User",
    "utc_now",
]
