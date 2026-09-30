"""
knowledge_base.py — 知识库管理

把一个领域的文档变成可检索的向量：
    PDF → 解析 → 分块 → 质量过滤 → 去重 → 向量化 → 写入集合

每个领域一个集合（kb_legal / kb_medical / kb_english）。父子块策略下父块与
子块都入库，子块的 parent_id 指向父块主键，命中子块时由 retrieval 层把
parent_text 一并交给大模型。
"""

from __future__ import annotations

import time
from pathlib import Path

import config
import embeddings
from chunking import DEFAULT_STRATEGY, Chunker
from pdf_parser import PDFParser
from quality import filter_chunks
from rerank_filter import deduplicate as deduplicate_by_vector
from retrieval import hybrid_retrieve, invalidate_index
from vector_store import get_store


class KnowledgeBase:
    """知识库的导入与检索入口。"""

    def __init__(self, size: int | None = None, overlap: int | None = None):
        self.chunker = Chunker(size or config.CHUNK_SIZE, overlap or config.CHUNK_OVERLAP)
        self.parser = PDFParser()

    # ------------------------------------------------------------ 清洗

    def deduplicate(self, chunks: list[dict]) -> list[dict]:
        """按向量相似度去重，超过 DUPLICATE_THRESHOLD 的只保留一条。"""
        # 先补向量：既用于相似度比对，也保证入库时每条都带 embedding
        missing = [c for c in chunks if not c.get("embedding")]
        if missing:
            vectors = embeddings.encode_texts([c.get("text", "") for c in missing])
            for chunk, vector in zip(missing, vectors):
                chunk["embedding"] = vector

        if len(chunks) <= 1:
            return chunks
        return deduplicate_by_vector(chunks, threshold=config.DUPLICATE_THRESHOLD)

    # ------------------------------------------------------------ 导入

    def _prepare(self, chunks: list[dict], source: str) -> list[dict]:
        """质量过滤 + 补向量 + 去重，返回可直接入库的分块。"""
        for chunk in chunks:
            chunk.setdefault("source", source)
            chunk.setdefault("summary", (chunk.get("text") or "")[:120])

        return self.deduplicate(filter_chunks(chunks))

    @staticmethod
    def _record(
        text: str,
        source: str,
        page: int,
        chunk_type: str,
        parent_id: int,
        embedding: list[float],
        now: int,
    ) -> dict:
        """构造一条入库记录。

        父块、子块、普通块三处的字段完全一致，集中在这里构造，
        避免以后加字段时只改了其中一两处。
        """
        return {
            "text": text,
            "summary": text[:120],
            "source": source,
            "page": page,
            "chunk_type": chunk_type,
            "parent_id": parent_id,
            "embedding": embedding,
            "created_at": now,
            "updated_at": now,
        }

    def _store(self, chunks: list[dict], domain: str) -> int:
        """写入向量库。父子块策略下先写父块，再回填子块的 parent_id。"""
        collection = config.KB_COLLECTIONS.get(domain)
        if collection is None:
            raise ValueError(f"未知领域：{domain}，可选：{', '.join(config.KB_COLLECTIONS)}")
        if not chunks:
            return 0

        store = get_store()
        now = int(time.time())

        # 同一个来源重新导入时先删掉旧记录。Milvus 的 insert 不去重，
        # 少了这一步，重导一次条数就翻一倍。
        for src in {c.get("source", "") for c in chunks if c.get("source")}:
            store.delete(collection, {"source": src})

        groups: dict[int, list[dict]] = {}
        plain: list[dict] = []
        for chunk in chunks:
            if chunk.get("parent_text"):
                groups.setdefault(int(chunk.get("parent_index", 0)), []).append(chunk)
            else:
                plain.append(chunk)

        written = 0

        # 父块先入库，拿到主键后才能给子块填 parent_id
        if groups:
            ordered = sorted(groups)
            parent_records = [
                self._record(
                    groups[idx][0].get("parent_text", ""),
                    groups[idx][0].get("source", ""),
                    int(groups[idx][0].get("page") or 0),
                    "parent",
                    0,
                    groups[idx][0]["embedding"],
                    now,
                )
                for idx in ordered
            ]
            parent_ids = store.upsert(collection, parent_records)
            written += len(parent_ids)
            id_map = dict(zip(ordered, parent_ids))

            child_records = [
                self._record(
                    chunk["text"],
                    chunk.get("source", ""),
                    int(chunk.get("page") or 0),
                    chunk.get("chunk_type", "parent_child"),
                    id_map.get(int(chunk.get("parent_index", 0)), 0),
                    chunk["embedding"],
                    now,
                )
                for chunk in chunks
                if chunk.get("parent_text")
            ]
            written += len(store.upsert(collection, child_records))

        if plain:
            records = [
                self._record(
                    chunk["text"],
                    chunk.get("source", ""),
                    int(chunk.get("page") or 0),
                    chunk.get("chunk_type", "paragraph"),
                    0,
                    chunk["embedding"],
                    now,
                )
                for chunk in plain
            ]
            written += len(store.upsert(collection, records))

        return written

    def import_pdf(self, pdf_path: str | Path, domain: str, strategy: str = DEFAULT_STRATEGY) -> dict:
        """导入 PDF：解析 → 分块 → 质量过滤 → 去重 → 入库。"""
        path = Path(pdf_path)
        blocks = self.parser.parse(path)

        raw: list[dict] = []
        for block in blocks:
            # source 只放文件名，页码单独用 page 字段承载，
            # 引用时才拼得成"文件名 第N页"，也便于按文件清理旧记录
            for chunk in self.chunker.chunk(block["text"], strategy):
                chunk["source"] = path.name
                chunk["page"] = int(block.get("page") or 0)
                raw.append(chunk)

        prepared = self._prepare(raw, path.name)
        stored = self._store(prepared, domain)
        invalidate_index(config.KB_COLLECTIONS.get(domain, ""))

        return {
            "file": path.name,
            "domain": domain,
            "strategy": strategy,
            "pages": len({b["page"] for b in blocks}),
            "raw_chunks": len(raw),
            "filtered_chunks": len(prepared),
            "stored": stored,
        }

    def import_texts(
        self, texts: list[str], domain: str, source: str = "manual", strategy: str = DEFAULT_STRATEGY
    ) -> int:
        """直接导入纯文本列表（测试与快速建库用）。"""
        raw: list[dict] = []
        for text in texts:
            for chunk in self.chunker.chunk(text, strategy):
                chunk["source"] = source
                chunk["page"] = 0  # 纯文本没有页码
                raw.append(chunk)

        prepared = self._prepare(raw, source)
        stored = self._store(prepared, domain)
        invalidate_index(config.KB_COLLECTIONS.get(domain, ""))
        return stored

    def import_directory(self, directory: str | Path, domain: str, strategy: str = DEFAULT_STRATEGY) -> list[dict]:
        """批量导入目录下所有 PDF。"""
        reports: list[dict] = []
        for pdf_path in sorted(Path(directory).glob("*.pdf")):
            try:
                reports.append(self.import_pdf(pdf_path, domain, strategy))
            except Exception as exc:
                reports.append({"file": pdf_path.name, "domain": domain, "error": str(exc)})
        return reports

    # ------------------------------------------------------------ 检索

    def search(self, query: str, domain: str, top_k: int = 5) -> list[dict]:
        """混合检索 + 重排序 + 余弦过滤。"""
        collection = config.KB_COLLECTIONS.get(domain)
        if collection is None:
            return []
        return hybrid_retrieve(query, collection, top_k=top_k)

    # ------------------------------------------------------------ 运维

    def stats(self) -> dict:
        """各领域集合的条数与来源文件数。"""
        store = get_store()
        result: dict[str, dict] = {}
        for domain, collection in config.KB_COLLECTIONS.items():
            # 不吞异常：读取失败时要让 /kb/stats 明确报错，
            # 静默返回 0 会让人误以为知识库是空的
            rows = store.query(collection, limit=100000)
            sources = {r.get("source") for r in rows if r.get("source")}
            result[domain] = {
                "collection": collection,
                "chunks": len(rows),
                "documents": len(sources),
            }
        return result

    def list_sources(self, domain: str) -> list[str]:
        """列出某个领域已入库的来源文件。"""
        collection = config.KB_COLLECTIONS.get(domain)
        if collection is None:
            return []
        rows = get_store().query(collection, limit=100000)
        return sorted({r.get("source", "") for r in rows if r.get("source")})

    def clear(self, domain: str) -> None:
        """清空某个领域的知识库。"""
        collection = config.KB_COLLECTIONS.get(domain)
        if collection is None:
            raise ValueError(f"未知领域：{domain}")
        get_store().drop(collection)
        invalidate_index(collection)


if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.clear("legal")
    count = kb.import_texts(
        [
            "劳动合同解除时，用人单位应当向劳动者支付经济补偿。经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资。",
            "劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。",
        ],
        domain="legal", source="劳动法节选.txt",
    )
    print(f"导入 {count} 条")
    hits = kb.search("试用期怎么辞职", "legal", top_k=2)
    for hit in hits:
        print(f"  {hit.get('rerank_score', 0):.3f} | {hit['text'][:30]}")
    assert hits and kb.stats()["legal"]["chunks"] > 0, "知识库写入失败"
    print("knowledge_base 自检通过。")
