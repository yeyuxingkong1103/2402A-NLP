# -*- coding: utf-8 -*-
# 【索引构建入口 · build_index.py】解析多份PDF→分块→建TF-IDF+BM25索引→持久化
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""建库命令行：

    python build_index.py

默认读取工单五目录下的 招股说明书1.pdf 与 招股说明书2.pdf，合并建库。
索引落盘后问答服务直接读取，不计入单次响应耗时。
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import build_chunks
from config import CONFIG
from pdf_parser import PDFParseError, parse_pdf
from vector_store import IndexStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build(pdf_paths, index_dir: str) -> None:
    """执行“解析→分块→嵌入→索引→落盘”完整建库流程。

    每份 PDF 单独解析并分块（确保各文档主体公司正确），再合并建索引。

    :param pdf_paths: 待入库 PDF 路径列表
    :param index_dir: 索引输出目录
    """
    t0 = time.perf_counter()
    all_chunks = []
    for path in pdf_paths:
        logger.info("① 解析PDF：%s", path)
        try:
            blocks = parse_pdf(path)
        except PDFParseError as exc:
            logger.error("文档解析失败，已中止建库：%s", exc)
            raise
        logger.info("解析得到结构化块 %d 个", len(blocks))
        # 每份文档单独分块，确保主体公司标注正确
        chunks = build_chunks(blocks)
        logger.info("生成检索块 %d 个", len(chunks))
        all_chunks.extend(chunks)

    logger.info("② 合并 %d 个检索块，构建 TF-IDF + BM25 双路索引", len(all_chunks))
    store = IndexStore.build(all_chunks)

    logger.info("③ 持久化到：%s", index_dir)
    store.save(index_dir)
    logger.info("建库完成，总耗时 %.2f s", time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="工单五多轮问答系统建库脚本")
    parser.add_argument("--pdf", action="append", default=CONFIG.pdf_paths,
                        help="待解析PDF路径（可多次指定，默认招股说明书1、2）")
    parser.add_argument("--index", default=CONFIG.index_dir, help="索引输出目录")
    args = parser.parse_args()
    build(args.pdf, args.index)


if __name__ == "__main__":
    main()
