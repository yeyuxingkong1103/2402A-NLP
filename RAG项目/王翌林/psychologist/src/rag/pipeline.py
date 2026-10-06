"""RAG 流水线组合根（需求文档 §12 的 pipeline.py）。

组合根（Composition Root）模式：它像一个总装工位——只负责把各处实现"接线"，
不负责"造零件"。所谓"组合根"，就是整个应用里唯一一个允许同时认识各方实现、
并把它们拼装到一起的地方。

离线链路：解析 → 分块 → 向量化 → 入库 → 元数据（委托 knowledge_service）
在线链路：Query 改写 → 检索 → 提示词 → 大模型 → 后处理 → 记忆（委托 rag_service / retrieval_service）

为什么 rag 层唯一允许"反向依赖" services？正常依赖方向是 api → services → rag → db/core，
即上层依赖下层。但本文件是编排者：它需要同时认识 rag 的底层能力与 services 的业务
流程，只能"向上"引用 services。若让 retriever/parser 等底层模块也去 import services，
就会形成 services → rag → services 的循环依赖。于是约定：只开这一个口子，
底层模块一律不许反向依赖，从而在不破坏分层的前提下完成组装。

分层约定（CLAUDE.md §3）：rag 层只依赖 db/core；唯一例外是本组合根
（composition root）——它只做编排转发，不含业务逻辑，且仅允许被
scripts/ 与 api 层调用。若本文件开始出现条件分支/计算，请下沉到对应服务模块。
"""
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.services import knowledge_service, rag_service, retrieval_service


# ==================== 离线链路 ====================
# 自检规则：本文件的函数都应长成"一行 return xxx_service.func(...)"的转发样式。
# 一旦这里出现 if / for / 计算 / 拼接，就说明业务逻辑漏到了编排层，应下沉到对应 service。
def offline_ingest_file(db: Session, file_path: str, persona_id: int,
                        strategy: str = "paragraph", title: Optional[str] = None) -> Dict[str, Any]:
    """单文件：解析 → 分块 → 向量化 → Milvus 入库 → MySQL 元数据。"""
    # 真实实现落在 knowledge_service.ingest_file：
    # rag.parser 解析 → rag.chunker 分块 → rag.embedder 向量化 → db.milvus 写向量 → db 写元数据
    return knowledge_service.ingest_file(db, file_path, persona_id, strategy=strategy, title=title)


def offline_ingest_directory(db: Session, dir_path: str, persona_id: int,
                             strategy: str = "paragraph") -> List[Dict[str, Any]]:
    """目录批量入库。"""
    # 真实实现落在 knowledge_service.ingest_directory：遍历目录后逐文件复用 ingest_file
    return knowledge_service.ingest_directory(db, dir_path, persona_id, strategy=strategy)


def offline_rebuild_persona(db: Session, persona_id: int, dirs: List[str],
                            drop_existing: bool = False,
                            strategy: str = "paragraph") -> Dict[str, Any]:
    """按角色知识库目录重建索引（可选先清空旧向量与元数据）。"""
    # 真实实现落在 knowledge_service.rebuild_persona_index：drop_existing=True 时先清空旧数据再重建
    return knowledge_service.rebuild_persona_index(
        db, persona_id, dirs, drop_existing=drop_existing, strategy=strategy
    )


# ==================== 在线链路 ====================
def online_search(persona_id: int, query: str, top_k: int = 5):
    """检索调试入口：Query 改写 → 混合检索 → 重排 → 阈值。"""
    # 真实实现落在 retrieval_service.search_with_query_rewrite：
    # 先做 Query 改写，再调用 rag.retriever.retrieve 完成四阶段检索漏斗
    return retrieval_service.search_with_query_rewrite(persona_id, query, top_k=top_k)


def online_answer(db: Session, user_id: int, persona_id: int, message: str,
                  conversation_id: Optional[int] = None) -> Dict[str, Any]:
    """非流式问答全链路。"""
    # 真实实现落在 rag_service.answer：检索 → 注入三层记忆与提示词 → 调大模型 → 后处理 → 写记忆
    return rag_service.answer(db, user_id, persona_id, message, conversation_id)


def online_answer_stream(db: Session, user_id: int, persona_id: int, message: str,
                         conversation_id: Optional[int] = None):
    """SSE 流式问答全链路。"""
    # 真实实现落在 rag_service.answer_stream：与非流式同链路，只是以 SSE 增量吐出 token
    return rag_service.answer_stream(db, user_id, persona_id, message, conversation_id)
