# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：PDF解析模块（优化版）
功能：解析PDF、提取表格、智能分块（含章节识别）
"""

import fitz
import pdfplumber
from typing import List, Dict, Any
import re


class PDFParser:
    """PDF文档解析器（优化版）"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_text(self) -> List[Dict[str, Any]]:
        """提取PDF文字（按页）"""
        pages_content = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            text = self._clean_text(text)
            pages_content.append({
                "page": page_num + 1,
                "content": text,
                "source": f"{self.pdf_path}#page={page_num + 1}"
            })
        return pages_content

    def extract_tables(self) -> List[Dict[str, Any]]:
        """提取PDF表格"""
        tables_content = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        tables_content.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "data": table,
                            "source": f"{self.pdf_path}#page={page_num + 1}&table={table_idx}"
                        })
        return tables_content

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        text = re.sub(r"\s+", " ", text)
        text = text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
        return text.strip()


    def _detect_section(self, text: str) -> str:
        """识别章节标题（如：第五节 业务与技术）"""
        patterns = [
            r"第[一二三四五六七八九十]+节\s*[^\s]{2,20}",
            r"第[一二三四五六七八九十]+章\s*[^\s]{2,20}",
            r"[一二三四五六七八九十]+、[^\s]{2,20}",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(0)[:30]
        return ""

    def chunk_text(self, pages_content: List[Dict],
                   chunk_size: int = 300,
                   overlap: int = 80) -> List[Dict[str, Any]]:
        """优化分块：加章节前缀 + 表格不切"""
        chunks = []
        current_section = ""

        for page_data in pages_content:
            text = page_data["content"]
            page_num = page_data["page"]
            source = page_data["source"]

            # 尝试更新章节
            sec = self._detect_section(text)
            if sec:
                current_section = sec

            sentences = re.split(r"(?<=[。！？；])", text)
            current_chunk = ""
            chunk_idx = 0

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        prefix = f"[第{page_num}页" + (f" §{current_section}" if current_section else "") + "] "
                        chunks.append({
                            "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                            "content": prefix + current_chunk,
                            "page": page_num,
                            "source": source,
                            "section": current_section,
                        })
                        chunk_idx += 1
                    current_chunk = current_chunk[-overlap:] + sentence if overlap > 0 else sentence

            if current_chunk:
                prefix = f"[第{page_num}页" + (f" §{current_section}" if current_section else "") + "] "
                chunks.append({
                    "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                    "content": prefix + current_chunk,
                    "page": page_num,
                    "source": source,
                    "section": current_section,
                })

        return chunks


if __name__ == "__main__":
    parser = PDFParser("./data/招股说明书1.pdf")
    pages = parser.extract_text()
    print("PDF pages:", len(pages))
    chunks = parser.chunk_text(pages, chunk_size=300, overlap=80)
    print("Chunks:", len(chunks))
    for i, c in enumerate(chunks[:3]):
        print("[", i+1, "]", c["chunk_id"])
        print("   ", c["content"][:80])
