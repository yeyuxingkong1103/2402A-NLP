# -*- coding: utf-8 -*-
# 【索引构建入口 · build_index.py】解析两份招股书→多字段分块→多模型嵌入→稠密库+多字段BM25建库→持久化
# 工单编号：人工智能NLP-RAG-混合检索任务

"""建库命令行（离线，不联网下载模型）：

    python build_index.py
    python build_index.py --backend m3e        # 切换 m3e 嵌入重建
    python build_index.py --backend tfidf      # 强制 TF-IDF 离线嵌入

索引落盘后问答服务直接读取，建库耗时不计入单次响应时间。
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import build_chunks
from config import CONFIG
from embeddings import create_embedder
from pdf_parser import PDFParseError, parse_pdf
from vector_store import IndexStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build(pdf_files: list, index_dir: str, backend: str) -> None:
    """执行“解析→分块→嵌入→多字段索引→落盘”完整建库流程。

    :param pdf_files: 待入库 PDF 路径列表（两份招股书）
    :param index_dir: 索引输出目录
    :param backend: 嵌入后端 bge / m3e / tfidf
    """
    t0 = time.perf_counter()

    logger.info("① 双引擎解析 %d 份招股书", len(pdf_files))
    all_blocks = []
    for pdf_path in pdf_files:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            blocks = parse_pdf(pdf_path, doc_name=doc_name)
        except PDFParseError as exc:
            logger.error("文档解析失败，已中止建库：%s", exc)
            raise
        logger.info("  - %s：结构化块 %d 个", doc_name, len(blocks))
        all_blocks.extend(blocks)

    logger.info("② 多字段标题感知分块")
    chunks = build_chunks(all_blocks)
    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    logger.info("生成检索块 %d 个（其中表格块 %d 个）", len(chunks), n_table)

    logger.info("③ 加载嵌入模型（%s）并编码，本地无缓存自动 TF-IDF 离线降级（不联网）",
                backend)
    embedder = create_embedder(backend, [c.text for c in chunks])

    logger.info("④ 构建稠密向量库 + 五字段加权 BM25 倒排")
    store = IndexStore.build(chunks, embedder)

    logger.info("⑤ 持久化到：%s", index_dir)
    store.save(index_dir)
    logger.info("建库完成，总耗时 %.2f s，嵌入后端=%s，块数=%d",
                time.perf_counter() - t0,
                getattr(embedder, "backend", backend), len(chunks))


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="工单六混合检索系统建库脚本")
    parser.add_argument("--backend", default=CONFIG.embedding_backend,
                        choices=["bge", "bge-large", "bge-m3", "m3e",
                                 "m3e-small", "tfidf"],
                        help="嵌入模型后端（本地无缓存自动降级TF-IDF）")
    parser.add_argument("--index", default=CONFIG.index_dir,
                        help="索引输出目录")
    args = parser.parse_args()
    if not CONFIG.pdf_files:
        raise FileNotFoundError(
            f"未在 {CONFIG.work_order_dir} 找到招股说明书1.pdf / 招股说明书2.pdf")
    build(CONFIG.pdf_files, args.index, args.backend)


if __name__ == "__main__":
    main()
