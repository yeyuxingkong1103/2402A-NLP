# -*- coding: utf-8 -*-
"""
向量化与入库脚本（离线数据管线 · 第三步）

职责：
    读取 chunk_split.py 的产物，用 BGE-M3 生成向量写入 Milvus，
    同时把文档元信息与 chunk 元数据（**含页码**）写入 MySQL。

双写的一致性设计：
    - Milvus 负责「哪些块相关」
    - MySQL 负责「这块来自哪一页」
    两者以 chunk_id 关联。MySQL 是页码溯源的唯一真相源，
    即使 Milvus 集合被重建，页码信息也不会丢失。

幂等性：
    重复执行同一份文档时，会先清理该文档在 Milvus 与 MySQL 中的旧数据，
    再写入新数据，不会产生重复 chunk。

用法：
    python -m scripts.embedding_store                  # 处理全部 chunk 产物
    python -m scripts.embedding_store --file xxx       # 只处理指定文档
    python -m scripts.embedding_store --rebuild        # 重建 Milvus 集合（清空重来）
    python -m scripts.embedding_store --skip-embedding # 跳过向量化（仅重建 MySQL 元数据）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 允许以脚本方式直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.db import milvus_client, mysql, redis_client
from backend.embedder import get_embedder
from backend.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ===========================================================================
# 单份文档入库
# ===========================================================================

def store_one(chunks_file: Path, *, skip_embedding: bool = False) -> Dict[str, Any]:
    """把一份文档的 chunk 列表写入 Milvus 与 MySQL"""
    # 读入 chunk_split.py 产出的中间文件：分好的块 + 每个块的页码元数据。
    data = json.loads(chunks_file.read_text(encoding="utf-8"))

    # 文档级元信息，这几项要写进 MySQL 的 documents 表。
    doc_id: str = data["doc_id"]
    file_name: str = data["file_name"]
    file_path: str = data.get("file_path", "")
    file_hash: str = data.get("file_hash", "")
    page_count: int = int(data.get("page_count", 0))
    parse_engine: str = data.get("parse_engine", "")
    # 这份文档切出来的全部块，每个块都带着 chunk_id、页码和正文。
    chunks: List[Dict[str, Any]] = data.get("chunks", [])

    # 一个块都没有就没必要往下走（例如解析出来是空文档）。
    if not chunks:
        logger.warning("该文档没有 chunk，跳过：%s", file_name)
        return {"doc_id": doc_id, "file_name": file_name, "chunk_count": 0}

    logger.info("开始入库：%s | %d 个 chunk", file_name, len(chunks))

    # ---- 1. 先登记文档元信息（chunks 表有外键依赖，必须先有主记录）----
    # file_hash 是文件内容指纹，用来判断同一份 PDF 是否被重新解析过。
    mysql.upsert_document(
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        file_hash=file_hash,
        page_count=page_count,
        chunk_count=len(chunks),
        parse_engine=parse_engine,
        status="indexed",
    )

    # ---- 2. 清理旧的 chunk 数据（保证幂等）----
    # 删掉文档主记录，chunks 表靠外键级联把它的旧块一并删掉 ——
    # 这样同一份文档重复入库不会累积出重复 chunk。
    mysql.delete_document(doc_id)          # 外键级联删除旧 chunk
    # 因为上一步把主记录也删了，这里必须重新登记一次，
    # 否则下面写 chunk 时会因缺少主记录而违反外键约束。
    mysql.upsert_document(
        doc_id=doc_id, file_name=file_name, file_path=file_path,
        file_hash=file_hash, page_count=page_count, chunk_count=len(chunks),
        parse_engine=parse_engine, status="indexed",
    )

    # ---- 3. 写入 chunk 元数据到 MySQL（页码溯源落库）----
    # ★ 这一步是「页码溯源」的数据基础 ★
    # Milvus 里只有向量，它只能回答"哪些块相关"；而"这块来自第几页、
    # 出自哪份文件"全都在 MySQL 的这张表里，在线问答时按 chunk_id 回来查。
    written_mysql = mysql.insert_chunks(chunks)
    logger.info("MySQL chunk 元数据写入完成：%d 条", written_mysql)

    # ---- 4. 向量化并写入 Milvus ----
    written_milvus = 0
    if skip_embedding:
        # 只重建 MySQL 元数据的排查模式：向量库那边保持原样不动。
        logger.warning("已跳过向量化，Milvus 未写入：%s", file_name)
    else:
        # 取出每个块的正文，准备送去编码。
        texts = [c["content"] for c in chunks]
        logger.info("正在向量化 %d 个 chunk（CPU 推理，请稍候）...", len(texts))
        started = time.time()
        # V2：一次 forward 同时产出稠密向量与稀疏权重（BGE-M3 共用同一编码器）
        # 稠密向量负责"意思相近"的语义检索，稀疏权重负责关键词字面匹配，
        # 两者互为补充，正是 V2/V3 混合检索的两路召回来源。
        vectors, sparse_vectors = get_embedder().encode_dense_and_sparse(
            texts, show_progress=True
        )
        logger.info("向量化完成，耗时 %.1f 秒", time.time() - started)

        # 数量对不上说明编码环节出了问题。如果不拦住，下面按位置配对就会
        # 悄悄错位 —— 把 A 块的向量挂到 B 块头上，检索结果就全乱了。
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"向量数量与 chunk 数量不一致：{len(vectors)} != {len(chunks)}"
            )
        # 稀疏向量同样必须与块一一对应。
        if len(sparse_vectors) != len(chunks):
            raise RuntimeError(
                f"稀疏向量数量与 chunk 数量不一致：{len(sparse_vectors)} != {len(chunks)}"
            )

        # 与 MySQL 那步同理：先按文档 ID 清掉旧向量，保证重复入库幂等。
        milvus_client.delete_by_doc_id(doc_id)   # 先清理旧向量
        # 按位置一一对应地写入：第 i 条的 ID、页码、正文、稠密/稀疏向量
        # 必须来自同一个 chunk 记录。
        written_milvus = milvus_client.insert_vectors(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["doc_id"] for c in chunks],
            page_nos=[c["page_no"] for c in chunks],
            contents=texts,
            embeddings=vectors,
            sparse_embeddings=sparse_vectors,
        )
        logger.info("Milvus 向量写入完成：%d 条（含稀疏向量）", written_milvus)

    # 返回本次入库的统计数据，供 main() 最后汇总展示。
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "mysql_chunks": written_mysql,
        "milvus_vectors": written_milvus,
    }


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> int:
    # 命令行参数：--file 用于只重建某一份文档（调试时最常用），
    # --rebuild 用于给 Milvus 集合补上稀疏向量字段。
    parser = argparse.ArgumentParser(description="BGE-M3 向量化并写入 Milvus + MySQL")
    parser.add_argument("--file", type=str, default=None,
                        help="只处理文件名包含该关键字的文档")
    parser.add_argument("--rebuild", action="store_true",
                        help="重建 Milvus 集合（删除后重新创建，清空全部向量）")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="跳过向量化，只重建 MySQL 元数据（用于快速排查）")
    args = parser.parse_args()

    setup_logging()
    # 先确保各个数据目录都存在，不然后面读写产物会直接失败。
    settings.ensure_directories()

    # ---- 初始化表结构与集合 ----
    # enable_sparse=True：V2 混合检索需要稀疏向量字段。
    # 注意集合 schema 一旦建立就不会再变（create_collection 对已存在集合直接跳过），
    # 因此从 V1 升级到 V2 时**必须加 --rebuild** 才能加上该字段。
    # 建 MySQL 表结构（已存在就跳过，是幂等的）。
    mysql.init_schema()
    # 建 Milvus 集合。drop_existing=True 对应 --rebuild：
    # 把旧集合连同里面的向量一起删掉重建。
    milvus_client.create_collection(
        drop_existing=args.rebuild, enable_sparse=True
    )
    # 老集合（V1 时期建的）没有稀疏向量字段。检测到就提醒用 --rebuild，
    # 否则 V2/V3 的混合检索会悄悄退化成只剩稠密检索，而且没有任何报错。
    if not args.rebuild and not milvus_client.has_sparse_field():
        logger.warning(
            "当前集合没有稀疏字段，V2 混合检索将不可用。"
            "请用 --rebuild 重建集合以启用稀疏向量。"
        )

    # ---- 收集待处理文件 ----
    # 扫出分块阶段产出的全部中间文件（*.chunks.json）。
    chunk_files = sorted(settings.parsed_path.glob("*.chunks.json"))
    # 带 --file 时按文件名关键字过滤，方便只重建指定的那一份文档。
    if args.file:
        chunk_files = [p for p in chunk_files if args.file in p.name]

    if not chunk_files:
        logger.error("未找到分块产物（*.chunks.json），请先运行 scripts.chunk_split")
        return 1

    logger.info("开始入库，共 %d 份文档", len(chunk_files))

    results: List[Dict[str, Any]] = []
    failed = 0
    # 逐份文档入库。单份失败只记录并继续，不让一份坏文件把整库重建打断。
    for chunk_file in chunk_files:
        try:
            results.append(store_one(chunk_file, skip_embedding=args.skip_embedding))
        except Exception as exc:
            logger.exception("入库失败：%s | %s", chunk_file.name, exc)
            failed += 1

    # ---- 知识库已变更：版本号自增，历史查询缓存自然失效 ----
    # 缓存 Key 里带了知识库版本号，版本一涨旧缓存就再也命中不了，
    # 用户下次提问必然走一遍新库检索 —— 不用去逐个删缓存。
    if results and not args.skip_embedding:
        redis_client.bump_kb_version()

    # ---- 汇总 ----
    # 汇总本次一共入库了多少个块、写出了多少条向量。
    total_chunks = sum(r.get("chunk_count", 0) for r in results)
    total_vectors = sum(r.get("milvus_vectors", 0) for r in results)
    logger.info(
        "入库结束 | 文档 %d 份 | chunk %d 个 | 向量 %d 条 | 失败 %d 份",
        len(results), total_chunks, total_vectors, failed,
    )

    # 顺带报一下整个知识库的当前规模，方便确认这次入库确实生效了。
    stats = mysql.get_kb_stats()
    logger.info("知识库当前规模：%s", stats)
    # 退出码：全部成功返回 0，有失败返回 1，便于脚本/流水线判断成败。
    return 0 if failed == 0 else 1


# 支持 python -m scripts.embedding_store 直接运行
if __name__ == "__main__":
    sys.exit(main())
