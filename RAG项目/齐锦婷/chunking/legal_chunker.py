import re
from typing import Any

from backend.app.core.config import get_settings


settings = get_settings()


class LegalChunker:
    """负责按法律章节和条款进行结构化分块。"""

    def chunk(self, blocks: list[dict[str, Any]], document_id: int, user_id: int, knowledge_base_id: int) -> list[dict[str, Any]]:
        # 准备保存最终分块。
        chunks: list[dict[str, Any]] = []
        # 记录当前章节路径，例如“第一章 总则”。
        current_titles: list[str] = []
        # 记录块序号，用于生成稳定的 chunk_uid。
        chunk_index = 0
        # 遍历清洗后的文本块。
        for block in blocks:
            # 获取当前块文本。
            text = str(block.get("text", "")).strip()
            # 获取当前块页码。
            page = int(block.get("page", 0) or 0)
            # 按行扫描，识别章节标题和条款。
            for piece, current_titles in self._split_by_law_structure(text, current_titles):
                # 对超过最大长度的内容继续按句子切分。
                for small_piece in self._split_long_text(piece):
                    # 小于最小长度的内容直接丢弃。
                    if len(small_piece.strip()) < settings.min_chunk_length:
                        continue
                    # 块序号递增。
                    chunk_index += 1
                    # 构造唯一块编号，方便 MySQL 和 Milvus 对齐。
                    chunk_uid = f"doc{document_id}-chunk{chunk_index}"
                    # 组装一个标准分块对象。
                    chunks.append(
                        {
                            "chunk_uid": chunk_uid,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                            "document_id": document_id,
                            "content_type": block.get("type", "text"),
                            "title_path": " / ".join(current_titles),
                            "content": small_piece.strip(),
                            "page_number": page,
                            "token_count": len(small_piece),
                        }
                    )
        # 返回所有结构化分块。
        return chunks

    def _split_by_law_structure(self, text: str, current_titles: list[str]) -> list[tuple[str, list[str]]]:
        # 准备输出列表，每个元素是“文本片段 + 当时的标题路径”。
        pieces: list[tuple[str, list[str]]] = []
        # 准备当前正在累积的条文内容。
        buffer: list[str] = []
        # 复制标题路径，避免修改外部列表。
        titles = list(current_titles)
        # 逐行处理文本。
        for line in text.splitlines():
            # 去掉行首尾空白。
            stripped = line.strip()
            # 空行跳过。
            if not stripped:
                continue
            # 判断是否是章节标题，例如“第一章 总则”。
            if re.match(r"^第[一二三四五六七八九十百千万\d]+[章节编]\s*", stripped):
                # 如果前面已经累积正文，先输出正文。
                if buffer:
                    pieces.append(("\n".join(buffer), list(titles)))
                    buffer = []
                # 章节标题进入标题路径。
                titles = [stripped]
                # 标题本身也保留为一个可检索片段。
                buffer.append(stripped)
                continue
            # 判断是否是法条开头，例如“第一条”。
            if re.match(r"^第[一二三四五六七八九十百千万\d]+条", stripped):
                # 新法条出现时，先输出上一条。
                if buffer:
                    pieces.append(("\n".join(buffer), list(titles)))
                    buffer = []
                # 开始累积新法条。
                buffer.append(stripped)
                continue
            # 普通正文继续累积到当前条文或段落。
            buffer.append(stripped)
        # 文件末尾如果还有累积内容，也要输出。
        if buffer:
            pieces.append(("\n".join(buffer), list(titles)))
        # 返回切好的结构化片段。
        return pieces

    def _split_long_text(self, text: str) -> list[str]:
        # 如果文本没有超过最大长度，直接返回。
        if len(text) <= settings.max_chunk_length:
            return [text]
        # 使用中文标点切句，尽量不截断句子。
        sentences = re.split(r"(?<=[。；！？])", text)
        # 准备保存切好的小块。
        chunks: list[str] = []
        # 当前正在累积的小块。
        current = ""
        # 逐句累积。
        for sentence in sentences:
            # 如果加上当前句子会超过最大长度，就先保存已有内容。
            if current and len(current) + len(sentence) > settings.max_chunk_length:
                chunks.append(current)
                current = sentence
            else:
                current += sentence
        # 保存最后一个小块。
        if current:
            chunks.append(current)
        # 如果仍然有超长小块，就按硬长度再次切分。
        final_chunks: list[str] = []
        for chunk in chunks:
            for start in range(0, len(chunk), settings.max_chunk_length):
                final_chunks.append(chunk[start : start + settings.max_chunk_length])
        # 返回最终结果。
        return final_chunks
