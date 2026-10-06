# -*- coding: utf-8 -*-
# 【索引构建入口 · build_index.py】解析PDF→分块→建双路索引→持久化
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""建库命令行：

    python build_index.py --pdf ../../招股说明书1.pdf

启动时单例加载，索引落盘后问答服务直接读取，不计入单次响应耗时。
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build(pdf_path: str, index_dir: str) -> None:
    """执行“解析→分块→嵌入→索引→落盘”完整建库流程。

    :param pdf_path: 待入库 PDF 路径
    :param index_dir: 索引输出目录
    """
    t0 = time.perf_counter()
    logger.info("① 解析PDF：%s", pdf_path)
    try:
        blocks = parse_pdf(pdf_path)
    except PDFParseError as exc:
        logger.error("文档解析失败，已中止建库：%s", exc)
        raise
    logger.info("解析得到结构化块 %d 个", len(blocks))

    logger.info("② 标题感知分块")
    chunks = build_chunks(blocks)
    logger.info("生成检索块 %d 个", len(chunks))

    logger.info("③ 加载嵌入模型并编码")
    embedder = create_embedder([c.text for c in chunks])

    logger.info("④ 构建稠密向量 + BM25 双路索引")
    store = IndexStore.build(chunks, embedder)

    logger.info("⑤ 持久化到：%s", index_dir)
    store.save(index_dir)
    logger.info("建库完成，总耗时 %.2f s", time.perf_counter() - t0)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="工单二问答系统建库脚本")
    parser.add_argument("--pdf", default=CONFIG.pdf_path, help="待解析PDF路径")
    parser.add_argument("--index", default=CONFIG.index_dir, help="索引输出目录")
    args = parser.parse_args()
    build(args.pdf, args.index)


if __name__ == "__main__":
    main()
