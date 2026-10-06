"""文档加载模块"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io

from src.utils.logger import logger
from src.utils.tools import clean_text


class DocumentLoader:
    """文档加载器基类"""
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PDFLoader(DocumentLoader):
    """PDF文档加载器"""
    
    def __init__(self, remove_watermark: bool = True):
        self.remove_watermark = remove_watermark
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载PDF文档
        
        Returns:
            List[Dict]: 每页内容，包含 page_number, text, images
        """
        documents = []
        
        try:
            # 使用PyMuPDF加载PDF
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本
                text = page.get_text()
                text = clean_text(text)
                
                # 提取图片
                images = self._extract_images(page)
                
                documents.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'images': images,
                    'source': os.path.basename(file_path)
                })
                
            doc.close()
            logger.info(f"成功加载PDF: {file_path}, 共 {len(documents)} 页")
            
        except Exception as e:
            logger.error(f"加载PDF失败: {e}")
            raise
        
        return documents
    
    def _extract_images(self, page) -> List[Dict[str, Any]]:
        """提取图片（返回字节，供后续 OCR/多模态使用；失败不阻断文本提取）"""
        images = []
        doc = page.parent

        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            try:
                info = doc.extract_image(xref)
                images.append({
                    'index': img_index,
                    'bytes': info.get('image'),
                    'width': info.get('width', img[2]),
                    'height': info.get('height', img[3])
                })
            except Exception as e:
                logger.warning(f"提取图片失败: {e}")
                continue

        return images


class PDFTableLoader(PDFLoader):
    """PDF表格加载器"""
    
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """提取表格"""
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            tables.append({
                                'page_number': page_num + 1,
                                'table_index': table_idx,
                                'table': table,
                                'source': os.path.basename(file_path)
                            })
            
            logger.info(f"从 {file_path} 提取了 {len(tables)} 个表格")
            
        except Exception as e:
            logger.error(f"提取表格失败: {e}")
            raise
        
        return tables


class TextLoader(DocumentLoader):
    """文本文件加载器"""
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """加载文本文件"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            documents.append({
                'page_number': 1,
                'text': clean_text(text),
                'images': [],
                'source': os.path.basename(file_path)
            })
            
            logger.info(f"成功加载文本文件: {file_path}")
            
        except Exception as e:
            logger.error(f"加载文本文件失败: {e}")
            raise
        
        return documents


def get_loader(file_path: str, remove_watermark: bool = True) -> DocumentLoader:
    """
    根据文件类型获取加载器
    
    Args:
        file_path: 文件路径
        remove_watermark: 是否去水印
    
    Returns:
        DocumentLoader实例
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return PDFLoader(remove_watermark=remove_watermark)
    elif ext in ['.txt', '.md', '.json']:
        return TextLoader()
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
