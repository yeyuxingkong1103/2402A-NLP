# -*- coding: utf-8 -*-
"""pipeline/build.py —— 离线构建总编排 build_pipeline。

在链路中的位置：
    backend/server.py 的后台构建线程 → 【本文件】 → 依次调用解析/清洗/分块/向量化/入库

固定顺序（不可调换）：
    解析 → 清洗 → 分块 → 向量化 → 删除旧向量 → 入库

每一步前后都通过 progress 回调发出状态，前端的进度条才能显示"当前卡在哪一步" ——
这是 V2 迭代"链路可观测"在离线侧的体现。
"""
from __future__ import annotations

import time
from typing import Callable

from .chunking import chunk_pages
from .config import EMBED_MODEL, MAX_CHUNK, PDF_DIR
from .pdf_parse import parse_pdf
from .text_clean import clean_text, keep_body_pages
from .vectors import delete_document_vectors, embed, store_chunks

def build_pipeline(filename: str, progress: Callable[[str, str, str], None]) -> dict:
    """执行一次完整构建，并通过 progress 报告每一步。

    参数：
        filename: data/pdfs/ 下的 PDF 文件名（不是完整路径）
        progress: 回调 (步骤名, 状态, 说明) -> None。
                  状态取 "running" / "done"，server.py 拿它更新构建状态供前端轮询。
    返回：
        {"pages": 页数, "chars": 正文字符数, "chunks": 块数,
         "vectors": 入库向量数, "seconds": 总耗时秒}

    固定顺序：解析 → 清洗 → 分块 → 向量化 → 删除旧向量 → 入库

    设计要点：
        每一步前后都发进度，前端的进度条才能显示"当前卡在哪一步"，
        而不是一个长时间不动的转圈 —— 这是 V2 迭代"链路可观测"在离线侧的体现。
    """
    pdf_path = PDF_DIR / filename
    started = time.time()

    progress("解析PDF", "running", "MinerU 优先，pypdf 兜底")
    pages = parse_pdf(pdf_path)
    progress("解析PDF", "done", f"{len(pages)} 页")

    progress("清洗文本", "running", "去噪、统一标准编号和 SF6 写法")
    # 逐页清洗后再统一裁掉封面目录：裁切需要跨页判断"1 范围"在哪一页出现
    cleaned_pages = keep_body_pages([{"page": page["page"], "text": clean_text(page["text"])} for page in pages])
    characters = sum(len(page["text"]) for page in cleaned_pages)
    progress("清洗文本", "done", f"{characters} 字符，保留 {len(cleaned_pages)} 页")

    progress("分块", "running", f"按章节切分，单块约 {MAX_CHUNK} 字符")
    chunks = chunk_pages(cleaned_pages)
    progress("分块", "done", f"{len(chunks)} 块")

    progress("向量化", "running", f"{EMBED_MODEL} 编码 {len(chunks)} 块")
    vectors = embed([chunk["text"] for chunk in chunks])
    progress("向量化", "done", f"{len(vectors)} 个向量")

    progress("入库", "running", "删除同名旧向量并写入 Milvus")
    delete_document_vectors(filename)  # 先删后写，实现同名文档"替换"语义
    count = store_chunks(filename, chunks, vectors)
    progress("入库", "done", f"{count} 个向量")

    return {
        "pages": len(pages),
        "chars": characters,
        "chunks": len(chunks),
        "vectors": count,
        "seconds": round(time.time() - started, 1),
    }
