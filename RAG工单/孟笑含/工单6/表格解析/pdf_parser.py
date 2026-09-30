# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：PDF解析模块（表格优化版）
功能：多PDF解析 + 表格按行拆块 + 章节识别
"""

import fitz
import pdfplumber
import os
import re
from typing import List, Dict, Any


class PDFParser:
    """PDF文档解析器（多文档 + 表格优化版）"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc_name = os.path.basename(pdf_path)
        self.doc = fitz.open(pdf_path)

    def extract_text(self) -> List[Dict[str, Any]]:
        """提取PDF文字（按页，含文档名）"""
        pages_content = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            text = self._clean_text(text)
            pages_content.append({
                "page": page_num + 1,
                "content": text,
                "source": f"{self.doc_name}#page={page_num + 1}",
                "doc": self.doc_name,
            })
        return pages_content

    def extract_tables(self) -> List[Dict[str, Any]]:
        """提取PDF表格（原始）"""
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
                            "source": f"{self.doc_name}#page={page_num + 1}&table={table_idx}",
                            "doc": self.doc_name,
                        })
        return tables_content

    def extract_table_chunks(self) -> List[Dict[str, Any]]:
        """表格按行拆块（关键优化）"""
        chunks = []
        tables = self.extract_tables()
        for t in tables:
            rows = t.get("data", [])
            if not rows or len(rows) < 2:
                continue
            headers = [str(c).strip() if c else "" for c in rows[0]]
            for row_idx, row in enumerate(rows[1:], 1):
                cells = [str(c).strip() if c else "" for c in row]
                if not any(cells):
                    continue
                # 每行拼成"表头: 值"格式，便于检索
                pairs = []
                for h, v in zip(headers, cells):
                    if h and v:
                        pairs.append(f"{h}={v}")
                    elif v:
                        pairs.append(v)
                text = "[表格] " + " | ".join(pairs)
                if len(text) < 10:
                    continue
                chunks.append({
                    "chunk_id": f"table_p{t['page']}_r{row_idx}",
                    "content": text,
                    "page": t["page"],
                    "source": t["source"],
                    "doc": t["doc"],
                    "type": "table_row",
                })
        return chunks

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        text = re.sub(r"\s+", " ", text)
        return "".join(c for c in text if ord(c) >= 32 or c in "\n\t").strip()

    def _detect_section(self, text: str) -> str:
        """识别章节标题"""
        patterns = [
            r"第[一二三四五六七八九十]+节\s*[^\s]{2,20}",
            r"第[一二三四五六七八九十]+章\s*[^\s]{2,20}",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(0)[:30]
        return ""

    def chunk_text(self, pages_content: List[Dict],
                   chunk_size: int = 300,
                   overlap: int = 80) -> List[Dict[str, Any]]:
        """文本分块（含文档名 + 章节前缀）"""
        chunks = []
        current_section = ""

        for page_data in pages_content:
            text = page_data["content"]
            page_num = page_data["page"]
            source = page_data["source"]
            doc = page_data.get("doc", self.doc_name)

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
                        prefix = f"[{doc} 第{page_num}页" + (f" §{current_section}" if current_section else "") + "] "
                        chunks.append({
                            "chunk_id": f"{doc}_page_{page_num}_c{chunk_idx}",
                            "content": prefix + current_chunk,
                            "page": page_num,
                            "source": source,
                            "doc": doc,
                            "section": current_section,
                            "type": "text",
                        })
                        chunk_idx += 1
                    current_chunk = current_chunk[-overlap:] + sentence if overlap > 0 else sentence

            if current_chunk:
                prefix = f"[{doc} 第{page_num}页" + (f" §{current_section}" if current_section else "") + "] "
                chunks.append({
                    "chunk_id": f"{doc}_page_{page_num}_c{chunk_idx}",
                    "content": prefix + current_chunk,
                    "page": page_num,
                    "source": source,
                    "doc": doc,
                    "section": current_section,
                    "type": "text",
                })

        return chunks


if __name__ == "__main__":
    for path in ["./data/招股说明书1.pdf", "./data/招股说明书2.pdf"]:
        parser = PDFParser(path)
        pages = parser.extract_text()
        text_chunks = parser.chunk_text(pages, chunk_size=300, overlap=80)
        table_chunks = parser.extract_table_chunks()
        print(f"\n【{parser.doc_name}】")
        print(f"  页数：{len(pages)}")
        print(f"  文本块：{len(text_chunks)}")
        print(f"  表格行块：{len(table_chunks)}")
        if table_chunks:
            print(f"  表格块示例：{table_chunks[0]['content'][:100]}")
