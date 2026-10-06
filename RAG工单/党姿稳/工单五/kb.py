# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
知识库装配：合并 文本块 + 表格块 + 图像块，形成统一检索库。
"""
import os
from pdf_parser import load_chunks, build_chunks
from image_parser import load_image_chunks, build_image_chunks
from config import CHUNKS_FILE, IMAGE_CHUNKS_FILE


def load_all_chunks():
    """返回 (全部块, 纯文本/表格块, 图像块)"""
    text_chunks = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    img_chunks = load_image_chunks() if os.path.exists(IMAGE_CHUNKS_FILE) else build_image_chunks()
    return text_chunks + img_chunks, text_chunks, img_chunks


if __name__ == "__main__":
    allc, tc, ic = load_all_chunks()
    print(f"文本/表格块 {len(tc)}，图像块 {len(ic)}，合计 {len(allc)}")
