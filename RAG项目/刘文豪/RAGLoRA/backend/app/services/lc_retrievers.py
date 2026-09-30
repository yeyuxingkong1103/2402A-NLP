# -*- coding: utf-8 -*-
"""LangChain 检索器适配层：把现有检索能力接入 LangChain 接口。

两处关键取舍（设计文档 §3.2）：

1. **不重复加载 bge-m3**
   langchain_huggingface.HuggingFaceEmbeddings 会新建一份模型实例
   （bge-m3 fp32 约 2.3GB）。本机 15.7GB 内存下，两份即 4.6GB，不可接受。
   因此 BgeM3DenseEmbeddings 委托给 app.services.embed 的全局单例。

2. **混合检索仍走手写实现**
   bge-m3 的 learned sparse 分支 LangChain 未覆盖。强行只做 dense 会丢失
   混合检索能力，故 HybridRetriever 包装现有 retrieval.hybrid_search，
   保留 dense ∥ sparse → RRF 的完整链路。
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ..core import config
from ..core.logging import get_logger
from . import retrieval
from .embed import encode, encode_one

log = get_logger("lc_retrievers")

_META_KEYS = ("id", "score", "source", "page", "law_name", "article_no", "collection")


def hit_to_doc(hit: dict) -> Document:
    """检索命中项 -> LangChain Document。"""
    return Document(
        page_content=hit.get("text") or "",
        metadata={k: hit.get(k) for k in _META_KEYS},
    )


def doc_to_hit(doc: Document) -> dict:
    """LangChain Document -> 检索命中项（供精排与 persona 复用）。"""
    hit = {k: doc.metadata.get(k) for k in _META_KEYS}
    hit["text"] = doc.page_content
    return hit


class BgeM3DenseEmbeddings(Embeddings):
    """bge-m3 的 dense 分支适配为 LangChain Embeddings。

    只暴露 dense —— sparse 由 HybridRetriever 内部处理。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        dense, _ = encode(list(texts), device="cpu")
        return dense

    def embed_query(self, text: str) -> list[float]:
        dense, _ = encode_one(text, device="cpu")
        return dense


class HybridRetriever(BaseRetriever):
    """包装现有混合检索（dense ∥ sparse → RRF）。"""

    collection: str = Field(description="Qdrant collection 名")
    k: int = Field(default=config.RECALL_TOP_K, description="召回条数")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        hits, _ = retrieval.hybrid_search(query, self.collection, self.k)
        log.debug("LangChain 检索 %s -> %d 条", self.collection, len(hits))
        return [hit_to_doc(h) for h in hits]
