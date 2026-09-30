"""Role RAG_try：多用户 / 多角色的本地化检索增强问答系统。

模型全部来自本地 D:/modelscope；向量库 Milvus；Redis 保存聊天记录与短期记忆；
检索为「稠密 + 稀疏 + BM25」三路召回后加权 RRF 融合。
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
