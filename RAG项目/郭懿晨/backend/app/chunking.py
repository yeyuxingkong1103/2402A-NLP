from uuid import uuid4

from backend.app.mineru import MinerUBlock
from backend.app.models import Chunk


def clean_text(text: str) -> str:
    """清洗文本中的多余空行和首尾空白。"""
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def clean_blocks(blocks: list[MinerUBlock]) -> list[MinerUBlock]:
    """对解析出的块做清洗：去掉多余空白并剔除空文本块。"""
    cleaned: list[MinerUBlock] = []
    for block in blocks:
        text = clean_text(block.text)
        if not text:
            continue
        cleaned.append(block.model_copy(update={"text": text}))
    return cleaned


def build_chunks(document_id: str, blocks: list[MinerUBlock], max_chars: int = 800) -> list[Chunk]:
    """从 MinerU 块构建可溯源 chunk。"""
    chunks: list[Chunk] = []
    for block in blocks:
        if block.page < 1:
            raise ValueError("chunk 缺少有效页码，禁止入库")
        cleaned = clean_text(block.text)
        if not cleaned:
            continue
        for start in range(0, len(cleaned), max_chars):
            part = cleaned[start : start + max_chars]
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    page=block.page,
                    category=block.label or "未分类",
                    text=part,
                    source_span=block.source_span,
                )
            )
    return chunks
