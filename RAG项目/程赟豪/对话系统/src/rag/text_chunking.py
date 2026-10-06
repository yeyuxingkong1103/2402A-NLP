"""文本分块模块"""
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class TextChunk:
    """文本块"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点切分句子。

    中文标点（。！？；）后无需空格即可切分；英文标点（.!?）后需跟空白再切，
    避免把小数/缩写（如 3.14、e.g.）误切。
    """
    text = (text or '').replace('\n', ' ')
    # 中文句末标点后直接切
    parts = re.split(r'(?<=[。！？；])', text)
    out: List[str] = []
    for p in parts:
        out.extend(re.split(r'(?<=[.!?])\s+', p))
    return [s.strip() for s in out if s.strip()]


class TextChunker:
    """文本分块器基类"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        raise NotImplementedError


class FixedLengthChunker(TextChunker):
    """固定长度分块器"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """按固定长度分块"""
        chunks = []
        metadata = metadata or {}

        # 按句子分割
        sentences = split_sentences(text)

        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # 保存当前块
                chunk_text = ''.join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    metadata={**metadata, 'chunk_index': chunk_id}
                ))
                chunk_id += 1
                
                # 处理重叠
                if self.chunk_overlap > 0:
                    # 保留最后重叠部分
                    overlap_text = ''.join(current_chunk)[-self.chunk_overlap:]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # 处理最后一个块
        if current_chunk:
            chunks.append(TextChunk(
                chunk_id=f"chunk_{chunk_id}",
                text=''.join(current_chunk),
                metadata={**metadata, 'chunk_index': chunk_id}
            ))
        
        logger.info(f"固定长度分块: 产生 {len(chunks)} 个块")
        return chunks


class SentenceChunker(TextChunker):
    """句子级分块器"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """按句子分块"""
        chunks = []
        metadata = metadata or {}

        # 按句子分割
        sentences = split_sentences(text)

        chunk_id = 0
        for idx, sentence in enumerate(sentences):
            if sentence.strip():
                chunks.append(TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=sentence.strip(),
                    metadata={**metadata, 'sentence_index': idx}
                ))
                chunk_id += 1
        
        logger.info(f"句子分块: 产生 {len(chunks)} 个块")
        return chunks


class ParagraphChunker(TextChunker):
    """段落级分块器"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """按段落分块"""
        chunks = []
        metadata = metadata or {}
        
        # 按段落分割（双换行或多个换行）
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunk_id = 0
        for idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append(TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=paragraph,
                    metadata={**metadata, 'paragraph_index': idx}
                ))
                chunk_id += 1
        
        logger.info(f"段落分块: 产生 {len(chunks)} 个块")
        return chunks


class SemanticChunker(TextChunker):
    """语义分块器 - 基于标题和主题"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """语义分块 - 优先按标题分割"""
        chunks = []
        metadata = metadata or {}
        
        # 按标题分割（# 开头的行，或大写字母开头的行）
        parts = re.split(r'(?=^#{1,6}\s|.^[A-Z][^a-z]+:\s)', text, flags=re.MULTILINE)
        
        chunk_id = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # 如果块太大，进一步分割
            if len(part) > self.chunk_size:
                sub_chunks = self._split_large_chunk(part, metadata, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                chunks.append(TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=part,
                    metadata={**metadata, 'chunk_index': chunk_id}
                ))
                chunk_id += 1
        
        logger.info(f"语义分块: 产生 {len(chunks)} 个块")
        return chunks
    
    def _split_large_chunk(self, text: str, metadata: Dict, start_id: int) -> List[TextChunk]:
        """分割大块"""
        chunks = []

        # 按句子分割
        sentences = split_sentences(text)

        current = []
        current_len = 0
        chunk_id = start_id
        
        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                chunks.append(TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=''.join(current),
                    metadata={**metadata, 'chunk_index': chunk_id}
                ))
                chunk_id += 1
                current = []
                current_len = 0
            
            current.append(sentence)
            current_len += len(sentence)
        
        if current:
            chunks.append(TextChunk(
                chunk_id=f"chunk_{chunk_id}",
                text=''.join(current),
                metadata={**metadata, 'chunk_index': chunk_id}
            ))
        
        return chunks


def get_chunker(method: str = "semantic", **kwargs) -> TextChunker:
    """
    获取分块器
    
    Args:
        method: 分块方式 fixed / sentence / paragraph / semantic
        **kwargs: 分块器参数
    
    Returns:
        TextChunker实例
    """
    chunkers = {
        'fixed': FixedLengthChunker,
        'sentence': SentenceChunker,
        'paragraph': ParagraphChunker,
        'semantic': SemanticChunker
    }
    
    if method not in chunkers:
        logger.warning(f"未知的分块方式: {method}, 使用 semantic")
        method = 'semantic'
    
    return chunkers[method](**kwargs)
