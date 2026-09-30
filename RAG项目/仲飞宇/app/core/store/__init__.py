"""存储层门面：向量库 / 关系库 / 短期记忆三件套的统一出口。

只做再导出、不放实现。上层一律写成 `from ..store import MilvusStore`，于是"换后端"
这件事被收敛成 Settings 里的几个值（MILVUS_DB_URI / SQL_URL / MEMORY_BACKEND），
调用方一行不用改——这也正是三个模块各自留一个同签名类的原因。

三者的分工（很容易混，读代码前先分清）：
    MilvusStore  向量 + chunk 原文，检索的唯一数据来源
    SQLStore     用户 / 角色 / **文档登记**，只管"哪些文件入过库"，不存正文
    MemoryStore  多轮对话上下文，唯一有 TTL 语义的一层
文档正文只在 Milvus、登记只在 SQL，所以重灌时必须两边一起收拾（见 dedup_document）。
"""
from .memory import InMemoryMemoryStore, MemoryStore, RedisMemoryStore, create_memory_store
from .milvus_store import MilvusStore
from .sql_store import SQLStore

__all__ = [
    "MilvusStore",
    "SQLStore",
    "MemoryStore",
    "InMemoryMemoryStore",
    "RedisMemoryStore",
    "create_memory_store",
]
