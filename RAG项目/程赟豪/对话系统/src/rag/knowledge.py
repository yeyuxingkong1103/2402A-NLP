"""知识库管理服务：文档加载 → 分块 → 向量化 → 入库"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import config
from src.utils.logger import logger
from src.rag.document_loader import get_loader
from src.rag.text_chunking import get_chunker
from src.rag.embedding import EmbeddingModel
from src.rag.vector_store import VectorStore


class KnowledgeService:
    """知识库服务：把文档（PDF/txt/md/json）解析、分块、向量化并写入向量库"""

    SUPPORTED_EXTS = (".pdf", ".txt", ".md", ".json")

    def __init__(self):
        self._embedder = None
        self._store = None

    @property
    def embedder(self) -> EmbeddingModel:
        """懒加载向量化模型"""
        if self._embedder is None:
            self._embedder = EmbeddingModel().load()
        return self._embedder

    @property
    def store(self) -> VectorStore:
        """懒加载向量库（不存在则创建集合）"""
        if self._store is None:
            self._store = VectorStore()
            self._store.create_collection()
        return self._store

    def _get_chunker(self, method: Optional[str] = None):
        method = method or config.get("document.chunking_method", "semantic")
        return get_chunker(
            method=method,
            chunk_size=config.get("document.chunk_size", 512),
            chunk_overlap=config.get("document.chunk_overlap", 50),
        )

    def ingest_file(
        self,
        file_path: str,
        source_name: Optional[str] = None,
        chunking_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """加载单个文件，分块、向量化并入库

        Returns:
            {"source": 来源名, "chunks": 入库分块数}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTS:
            raise ValueError(f"不支持的文件类型: {path.suffix}（支持 {self.SUPPORTED_EXTS}）")

        loader = get_loader(
            str(path),
            remove_watermark=config.get("document.remove_watermark", True),
        )
        pages = loader.load(str(path))
        source = source_name or path.name
        chunker = self._get_chunker(chunking_method)

        # 分块
        all_chunks = []
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue
            chunks = chunker.chunk(
                text,
                metadata={
                    "source": source,
                    "page_number": page.get("page_number", 1),
                },
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning(f"{path.name} 未提取到任何文本，跳过")
            return {"source": source, "chunks": 0}

        # 向量化 + 入库
        texts = [c.text for c in all_chunks]
        vectors = self.embedder.encode(texts)
        summaries = [self._make_summary(t) for t in texts]

        # 先删除同来源旧分块，保证重复索引幂等（不会产生重复数据）
        self.store.delete_by_source(source)

        self.store.insert(
            texts=texts,
            vectors=vectors,
            chunk_ids=[c.chunk_id for c in all_chunks],
            sources=[c.metadata["source"] for c in all_chunks],
            page_numbers=[c.metadata["page_number"] for c in all_chunks],
            metadata=[c.metadata for c in all_chunks],
            summaries=summaries,
        )

        logger.info(f"入库完成: {source} → {len(all_chunks)} 个分块")
        return {"source": source, "chunks": len(all_chunks)}

    @staticmethod
    def _make_summary(text: str, max_len: int = 120) -> str:
        """生成分块摘要（文档.txt 要求 Milvus collection 含「摘要」字段）。

        当前用轻量规则：取文本开头 max_len 个字符作为摘要，避免引入额外的摘要模型。
        """
        text = (text or "").strip().replace("\n", " ")
        return text[:max_len]

    def ingest_directory(self, dir_path: str, drop_existing: bool = False) -> Dict[str, Any]:
        """批量处理目录下所有受支持的文档

        Returns:
            {"files": 处理文件数, "chunks": 总入库分块数, "errors": 失败列表}
        """
        raw_path = Path(dir_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        files = [
            f for f in sorted(raw_path.rglob("*"))
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTS
        ]
        if not files:
            logger.warning(f"{dir_path} 下没有可处理的文档")
            return {"files": 0, "chunks": 0, "errors": []}

        # 重建集合（可选）
        if drop_existing:
            self._store = VectorStore()
            self._store.create_collection(drop_existing=True)

        total_chunks = 0
        errors = []
        for f in files:
            try:
                result = self.ingest_file(str(f))
                total_chunks += result["chunks"]
                logger.info(f"  ✓ {f.name}: {result['chunks']} 个分块")
            except Exception as e:
                logger.error(f"处理 {f.name} 失败: {e}")
                errors.append({"file": f.name, "error": str(e)})

        return {"files": len(files) - len(errors), "chunks": total_chunks, "errors": errors}

    def get_stats(self) -> Dict[str, Any]:
        """返回知识库统计"""
        try:
            count = self.store.get_count()
            sources = self.store.get_distinct_sources()
        except Exception as e:
            logger.warning(f"获取知识库统计失败: {e}")
            count = 0
            sources = []
        return {"total_docs": len(sources), "total_chunks": count, "sources": sources}

    def list_sources(self) -> List[str]:
        """列出知识库中所有文档来源"""
        try:
            return self.store.get_distinct_sources()
        except Exception as e:
            logger.warning(f"列出知识库来源失败: {e}")
            return []

    def delete_source(self, source: str) -> Dict[str, Any]:
        """按来源删除某文档的所有分块（知识库动态更新）。"""
        try:
            self.store.delete_by_source(source)
        except Exception as e:
            raise RuntimeError(f"删除来源失败: {e}")
        logger.info(f"已从知识库删除来源: {source}")
        return {"source": source, "deleted": True}


# 全局实例（惰性加载，避免启动时加载模型）
knowledge_service = KnowledgeService()
