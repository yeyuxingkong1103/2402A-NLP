"""长期记忆模块（Milvus）
长期记忆（文档.txt 要求）存用户画像 / 对话摘要到 Milvus，可向量化检索。
按用户隔离：把 user_id 写入 `source` 字段，检索/清除时按 `source` 过滤。
"""
from typing import List, Dict, Any, Optional
from src.config import config
from src.utils.logger import logger
from src.rag.vector_store import VectorStore
from src.rag.embedding import EmbeddingModel
class LongTermMemory:
    """长期记忆 - 使用 Milvus 存储用户/角色特征"""
    def __init__(self):
        self.vector_store = VectorStore("user_memories")
        self.embedding_model = EmbeddingModel()
    def initialize(self):
        """初始化"""
        self.vector_store.create_collection(drop_existing=False)
        self.vector_store.load_collection()
        self.embedding_model.load()
        logger.info("长期记忆初始化完成")
    def save_user_profile(
        self,
        user_id: str,
        profile_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """保存用户画像"""
        vector = self.embedding_model.encode(profile_text)
        metadata = metadata or {}
        metadata.update({"user_id": user_id, "type": "user_pr"
                                                     "ofile"})
        self.vector_store.insert(
            texts=[profile_text],
            vectors=vector,
            chunk_ids=[f"profile_{user_id}"],
            sources=[user_id],
            metadata=[metadata],
        )
        logger.info(f"保存用户 {user_id} 画像")
    def save_conversation_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """保存对话摘要"""
        vector = self.embedding_model.encode(summary)

        metadata = metadata or {}
        metadata.update({"user_id": user_id, "session_id": session_id, "type": "conversation_summary"})

        self.vector_store.insert(
            texts=[summary],
            vectors=vector,
            chunk_ids=[f"summary_{session_id}"],
            sources=[user_id],
            metadata=[metadata],
        )

        logger.info(f"保存会话 {session_id} 摘要")

    def retrieve_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """检索记忆（按用户隔离）"""
        query_vector = self.embedding_model.encode(query)

        expr = f'source == "{user_id}"' if user_id else None
        return self.vector_store.search(query_vector, top_k, expr)

    def clear_user_memories(self, user_id: str):
        """清除用户记忆"""
        self.vector_store.delete_by_source(user_id)
        logger.info(f"清除用户 {user_id} 的记忆")


# 全局实例
long_term_memory = LongTermMemory()
