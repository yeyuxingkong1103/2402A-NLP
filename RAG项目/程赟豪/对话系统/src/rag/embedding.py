"""向量化模块"""
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config
from src.utils.logger import logger


class EmbeddingModel:
    """向量化模型"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.get('embedding.model', 'BAAI/bge-m3')
        self.device = device or config.get('embedding.device', 'cuda')
        self.model = None
        self.batch_size = config.get('embedding.batch_size', 32)
        self.normalize = config.get('embedding.normalize', True)
    
    def load(self):
        """加载模型"""
        if self.model is None:
            logger.info(f"加载向量化模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"模型加载完成，维度: {self.get_dimension()}")
        return self
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.model is None:
            self.load()
        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        向量化文本
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批次大小
            normalize: 是否归一化
        
        Returns:
            numpy数组
        """
        if self.model is None:
            self.load()
        
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.batch_size
        normalize = normalize if normalize is not None else self.normalize
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 10
        )
        
        return embeddings
    
    def encode_multi(
        self, 
        texts: List[str], 
        query: str,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        多文本向量化（用于混合检索）
        
        Returns:
            堆叠的向量
        """
        all_texts = [query] + texts
        embeddings = self.encode(all_texts, batch_size=batch_size)
        return embeddings


class BGEEmbedder(EmbeddingModel):
    """BGE向量化器（别名）"""
    pass
