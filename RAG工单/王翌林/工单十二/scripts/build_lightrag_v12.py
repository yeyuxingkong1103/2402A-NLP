#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-LightRAG优化
scripts/build_lightrag_v12.py —— 从招股说明书 PDF 构建 LightRAG 知识图谱

流程：
1. 用 PyMuPDF 提取《招股说明书1.pdf》《招股说明书2.pdf》文本
2. 调用 LightRAG.insert() 构建知识图谱（实体/关系抽取由 DeepSeek 完成）
3. 图谱持久化到 data/lightrag_v12/（NetworkX JSON + NanoVectorDB）
"""
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORK_ORDER = "人工智能NLP-RAG-LightRAG优化"
PDF_DIR = Path(__file__).resolve().parents[1] / "附件"
PDFS = [
    ("招股说明书1", PDF_DIR / "招股说明书1-无水印.pdf"),
    ("招股说明书2", PDF_DIR / "招股说明书2.pdf"),
]


def extract_pdf_text(pdf_path: Path) -> str:
    """工单十二：用 PyMuPDF 提取 PDF 全文"""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(f"[第{i+1}页]\n{text}")
    doc.close()
    return "\n".join(pages)


def main():
    from src.lightrag_v12 import get_lightrag, lightrag_insert_text

    print(f"[v12] {WORK_ORDER}")
    print(f"[v12] 开始构建 LightRAG 知识图谱...")

    rag = get_lightrag()
    print(f"[v12] LightRAG 初始化完成，working_dir={rag.working_dir}")

    for doc_id, pdf_path in PDFS:
        if not pdf_path.exists():
            print(f"[v12] 跳过不存在的文件: {pdf_path}")
            continue

        print(f"\n[v12] 处理 {doc_id}: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
        t0 = time.perf_counter()
        text = extract_pdf_text(pdf_path)
        print(f"[v12]   提取文本 {len(text)} 字符，耗时 {time.perf_counter()-t0:.1f}s")

        t1 = time.perf_counter()
        try:
            lightrag_insert_text(text, doc_id=doc_id)
            elapsed = time.perf_counter() - t1
            print(f"[v12]   知识图谱构建完成，耗时 {elapsed:.1f}s")
        except Exception as e:
            print(f"[v12]   构建失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[v12] 全部文档处理完成")
    print(f"[v12] 图谱存储目录: {rag.working_dir}")
    print(f"[v12] 文件列表:")
    for f in sorted(Path(rag.working_dir).glob("*")):
        print(f"       {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
