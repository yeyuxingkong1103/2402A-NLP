# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：检索配置管理
功能：定义检索模式、嵌入模型、重排算法、融合策略
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RetrievalConfig:
    """检索配置"""

    mode: str = "hybrid"  # vector / fulltext / hybrid

    # 向量检索
    embedding_model: str = "BAAI/bge-m3"
    vector_top_k: int = 20
    vector_score_threshold: float = 0.0

    # 全文检索
    fulltext_top_k: int = 20
    fulltext_method: str = "bm25"

    # 重排
    use_rerank: bool = True
    rerank_method: str = "cross_encoder"  # cross_encoder / tfidf / adaptive
    rerank_top_k: int = 3

    # 混合融合
    fusion_method: str = "rrf"  # rrf / weighted / voting
    weight_vector: float = 0.6
    weight_fulltext: float = 0.4
    rrf_k: int = 60

    EMBEDDING_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "bge-m3": "BAAI/bge-m3",
        "bge-base": "BAAI/bge-base-zh-v1.5",
        "m3e": "moka-ai/m3e-base",
    })

    RERANK_METHODS: List[str] = field(default_factory=lambda: [
        "cross_encoder",
        "tfidf",
        "adaptive",
    ])

    def switch_mode(self, mode: str):
        assert mode in ["vector", "fulltext", "hybrid"]
        self.mode = mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "embedding_model": self.embedding_model,
            "vector_top_k": self.vector_top_k,
            "fulltext_top_k": self.fulltext_top_k,
            "use_rerank": self.use_rerank,
            "rerank_method": self.rerank_method,
            "rerank_top_k": self.rerank_top_k,
            "fusion_method": self.fusion_method,
            "weight_vector": self.weight_vector,
            "weight_fulltext": self.weight_fulltext,
        }


if __name__ == "__main__":
    cfg = RetrievalConfig()
    print("默认配置：")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print()
    print("嵌入模型库：", list(cfg.EMBEDDING_MODELS.keys()))
    print("重排算法库：", cfg.RERANK_METHODS)
