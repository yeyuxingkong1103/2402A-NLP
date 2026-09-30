"""PDF 解析 + 分块 + 向量化 + 入库（Milvus，USE_MILVUS=false 时降级 Chroma）。

所有执行逻辑都放在 if __name__ == "__main__": 里，模块顶层只保留常量与函数，
保证 rag.py 可以安全 import 而不触发入库或加载模型。
纯分块逻辑已拆到 chunker.py，此处经 re-export 保住 ingest.split_chunks 等旧引用。

用法：
    python ingest.py
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import chromadb
from dotenv import load_dotenv

import milvus_store
import parser
from chunker import (  # noqa: F401  供 tests / eval / scripts 经 ingest.* 访问
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PARENT_HARD_CUT,
    PARENT_SIZE,
    clean_chunks,
    make_summary,
    normalize_whitespace,
    split_chunks,
    split_parents,
)
from milvus_store import USE_MILVUS

logger = logging.getLogger(__name__)

PDF_PATH = "data/raw/guide.pdf"
CHROMA_DIR = "data/chroma"
COLLECTION = milvus_store.MILVUS_COLLECTION
EMBED_MODEL = os.getenv("EMBED_MODEL", "/mnt/d/models/models/BAAI--bge-m3/snapshots/master")

BATCH_SIZE = 32        # 向量化批大小
SOURCE_NAME = "guide.pdf"  # 入库 metadata 的来源文件名


@lru_cache(maxsize=1)
def load_embedder() -> "SentenceTransformer":
    """加载向量模型（进程内缓存，供 rag.py 复用）。

    延迟 import sentence_transformers，确保 load_dotenv() 设置的 HF_ENDPOINT
    在 huggingface_hub 首次导入前生效。
    """
    from sentence_transformers import SentenceTransformer

    logger.info("加载向量模型：%s ...", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=8)
def get_collection(collection_name: str = COLLECTION):
    """按 USE_MILVUS 返回 Milvus 客户端或 Chroma collection（按 collection 进程内缓存）。"""
    if USE_MILVUS:
        return milvus_store.get_milvus_client()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )


@lru_cache(maxsize=8)
def load_keyword_index(collection_name: str = COLLECTION):
    """拉全量 chunk，jieba 分词后构建 BM25Okapi 索引（按 collection 进程内缓存）。

    Milvus 模式从 Milvus query_all 拉全量；Chroma 模式从 .get() 拉全量。
    返回 (bm25, chunks)，chunks 与语料顺序对齐：[{"id","content","page","parent_content"}, ...]。
    构建失败（索引缺失/分词失败等）返回 (None, [])，由调用方降级为纯向量检索。
    """
    try:
        import jieba
        from rank_bm25 import BM25Okapi

        jieba.setLogLevel(logging.WARNING)  # 压掉 jieba 首次建词典的 DEBUG 噪音
        if USE_MILVUS:
            rows = milvus_store.query_all(collection_name)
            chunks = [
                {
                    "id": r["id"], "content": r["content"], "page": int(r.get("page", 0)),
                    "parent_content": r.get("parent_content", ""),
                }
                for r in rows
            ]
        else:
            res = get_collection(collection_name).get(include=["documents", "metadatas"])
            chunks = [
                {"id": cid, "content": doc, "page": int(meta.get("page", 0))}
                for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
            ]
        corpus = [jieba.lcut(c["content"]) for c in chunks]
        logger.info("BM25 索引构建完成，共 %d 个 chunk", len(chunks))
        return BM25Okapi(corpus), chunks
    except Exception as e:  # noqa: BLE001
        logger.warning("BM25 索引不可用，回退纯向量检索: %s", e)
        return None, []


def ingest_pdf(pdf_path: str, source: str, domain: str, collection: str) -> int:
    """解析 PDF -> 父子分块 -> 清洗 -> 向量化 -> 覆盖式入库，返回入库 chunk 数。

    父块 ~1500 字（split_parents），父块内再切 ~400 字子块（split_chunks），
    每个子块冗余记录 parent_content（父块全文）。
    """
    pages = parser.extract_text(pdf_path)
    tables = parser.extract_tables(pdf_path)
    logger.info("抽取到 %d 页正文、%d 个表格", len(pages), len(tables))

    all_chunks: list[dict] = []
    for p in pages:
        idx = 0
        # 先切父块（在 normalize 之前，保留空行段落边界），再逐父块 normalize + 切子块
        for parent in split_parents(p["text"]):
            parent = normalize_whitespace(parent)
            subs = split_chunks(parent, p["page"], start=idx)
            for s in subs:
                s["parent_content"] = parent
            all_chunks.extend(subs)
            idx += len(subs)
    for i, t in enumerate(tables):
        all_chunks.append({
            "id": f"p{t['page']}_t{i}",
            "content": normalize_whitespace(t["text"]),
            "page": t["page"],
        })
    logger.info("分块后共 %d 个 chunk", len(all_chunks))
    return _embed_and_store(all_chunks, source, domain, collection)


def ingest_image(image_path: str, source: str, domain: str, collection: str) -> int:
    """OCR 单张图片 -> 作为一条 chunk 清洗 -> 向量化 -> 覆盖式入库，返回入库 chunk 数。"""
    pages = parser.extract_from_image(image_path)
    if not pages:
        logger.warning("OCR 无结果，跳过: %s", image_path)
        return 0

    stem = os.path.basename(image_path).rsplit(".", 1)[0]
    all_chunks: list[dict] = []
    for p in pages:
        all_chunks.append({
            "id": f"{stem}_p{p['page']}_c0",
            "content": p["text"],
            "page": p["page"],
        })
    logger.info("OCR 共 %d 个 chunk", len(all_chunks))
    return _embed_and_store(all_chunks, source, domain, collection)


def _embed_and_store(
    all_chunks: list[dict], source: str, domain: str, collection: str
) -> int:
    """清洗 -> 向量化 -> 覆盖式入库（Milvus/Chroma），返回入库 chunk 数。"""
    if not all_chunks:
        return 0
    all_chunks, _ = clean_chunks(all_chunks)

    embedder = load_embedder()
    ids = [c["id"] for c in all_chunks]
    documents = [c["content"] for c in all_chunks]

    logger.info("向量化中 ...")
    embeddings = embedder.encode(
        documents, batch_size=BATCH_SIZE, normalize_embeddings=True
    ).tolist()

    now = int(time.time())
    if USE_MILVUS:
        milvus_store.create_collection(collection)
        milvus_store.delete_by_source(collection, source)
        records = [
            {
                "id": c["id"],
                "embedding": emb,
                "content": c["content"],
                "page": c["page"],
                "source": source,
                "domain": domain,
                "created_at": now,
                "updated_at": now,
                "summary": make_summary(c["content"]),
                "parent_content": c.get("parent_content", ""),
            }
            for c, emb in zip(all_chunks, embeddings)
        ]
        start = time.perf_counter()
        milvus_store.insert(collection, records)
        elapsed = time.perf_counter() - start
        logger.info(
            "已写入 %d 条到 Milvus（collection: %s），耗时 %.2fs",
            len(records), collection, elapsed,
        )
    else:
        metadatas = [{"page": c["page"], "source": source} for c in all_chunks]
        chroma_coll = get_collection(collection)
        existing = chroma_coll.get()
        if existing["ids"]:
            chroma_coll.delete(ids=existing["ids"])
        chroma_coll.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("已入库 %d 个 chunks（collection: %s）", len(ids), collection)
    return len(ids)


def ingest() -> int:
    """CLI 默认入库：data/raw/guide.pdf -> 默认 collection（domain 空串）。"""
    return ingest_pdf(PDF_PATH, SOURCE_NAME, "", COLLECTION)


if __name__ == "__main__":
    load_dotenv()  # 加载 .env（如 HF_ENDPOINT 镜像）；import 本模块时不会执行
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    ingest()
"""
离线入库模块`ingest.py`执行流程分为三步。第一，入口函数`ingest()`作为 CLI 调用入口，调用`ingest_pdf()`处理 PDF，或调用`ingest_image()`处理图片；`ingest_pdf()`通过`parser.extract_text()`提取正文、`parser.extract_tables()`提取表格，正文先调用`split_parents()`切父块，再调用`split_chunks()`拆分子块，子块保存`parent_content`父文本，表格单独生成 chunk；`ingest_image()`通过`parser.extract_from_image()`执行 OCR，从图片提取文本并构造 chunk。第二，生成的全部 chunk 送入`_embed_and_store()`，先调用`clean_chunks()`清洗文本，再调用`load_embedder()`加载向量模型，按批次完成文本向量化；入库前删除同一数据源的旧数据，`get_collection()`根据`USE_MILVUS`配置选择后端，开启则写入 Milvus，否则降级使用 Chroma。第三，模块还提供`load_keyword_index()`，读取向量库内所有 chunk，用 jieba 分词构建 BM25 索引；索引构建失败会返回空，检索链路自动降级为纯向量检索。模块顶层只定义常量与函数，只有脚本直接运行，也就是`if __name__ == "__main__"`分支时，才加载环境变量并触发入库；单纯 import 该模块不会执行入库逻辑，最终返回入库成功的 chunk 数量。
"""
