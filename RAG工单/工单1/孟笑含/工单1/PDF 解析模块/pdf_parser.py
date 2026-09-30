# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：PDF解析模块
功能：解析PDF文档，提取文字和表格内容
"""

import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any
import re


class PDFParser:
    """PDF文档解析器"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def extract_text(self) -> List[Dict[str, Any]]:
        """提取PDF中的文字内容，按页返回"""
        pages_content = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            # 清理文本
            text = self._clean_text(text)
            pages_content.append({
                "page": page_num + 1,
                "content": text,
                "source": f"{self.pdf_path}#page={page_num + 1}"
            })
        return pages_content
    
    def extract_tables(self) -> List[Dict[str, Any]]:
        """提取PDF中的表格数据"""
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
        """清洗文本，去除多余空白和特殊字符"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        return text.strip()
    
    def chunk_text(self, pages_content: List[Dict], 
                   chunk_size: int = 500, 
                   overlap: int = 100) -> List[Dict[str, Any]]:
        """将文本分块，便于向量化"""
        chunks = []
        for page_data in pages_content:
            text = page_data["content"]
            page_num = page_data["page"]
            source = page_data["source"]
            
            # 按句子分割
            sentences = re.split(r'(?<=[。！？；])', text)
            
            current_chunk = ""
            chunk_idx = 0
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                            "content": current_chunk,
                            "page": page_num,
                            "source": source
                        })
                        chunk_idx += 1
                    # 保留重叠部分
                    current_chunk = current_chunk[-overlap:] + sentence if overlap > 0 else sentence
            
            if current_chunk:
                chunks.append({
                    "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                    "content": current_chunk,
                    "page": page_num,
                    "source": source
                })
        
        return chunks