import re
from collections import Counter
from typing import Any

from backend.app.core.config import get_settings


settings = get_settings()


class DocumentCleaner:
    """负责把解析结果中的页码、页眉、页脚和明显噪声清理掉。"""

    def clean(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 先统计短行出现次数，重复出现的短行很可能是页眉或页脚。
        repeated_lines = self._find_repeated_short_lines(blocks)
        # 准备保存清洗后的块。
        cleaned_blocks: list[dict[str, Any]] = []
        # 逐个处理解析得到的块。
        for block in blocks:
            # 只处理文本内容，其他类型如果没有文本就会被跳过。
            text = str(block.get("text", ""))
            # 对当前块执行逐行清理。
            cleaned_text = self._clean_text(text, repeated_lines)
            # 如果清理后长度过短，就丢弃，避免噪声进入检索。
            if len(cleaned_text.strip()) < settings.min_chunk_length:
                continue
            # 复制原块，避免直接修改输入数据。
            new_block = dict(block)
            # 写入清洗后的文本。
            new_block["text"] = cleaned_text
            # 放入清洗结果。
            cleaned_blocks.append(new_block)
        # 返回所有清洗后的块。
        return cleaned_blocks

    def _find_repeated_short_lines(self, blocks: list[dict[str, Any]]) -> set[str]:
        # 准备一个计数器，用来统计每一行出现了几次。
        counter: Counter[str] = Counter()
        # 遍历所有文本块。
        for block in blocks:
            # 按行拆分文本，逐行判断是否可能是页眉页脚。
            for line in str(block.get("text", "")).splitlines():
                # 去掉首尾空白，便于合并相同内容。
                normalized = line.strip()
                # 只统计较短的行，因为页眉页脚通常不会很长。
                if 0 < len(normalized) <= 30:
                    counter[normalized] += 1
        # 出现三次以上的短行，认为是重复页眉页脚候选。
        return {line for line, count in counter.items() if count >= 3}

    def _clean_text(self, text: str, repeated_lines: set[str]) -> str:
        # 准备保存保留下来的正文行。
        kept_lines: list[str] = []
        # 按行清理，避免误删整段法律条文。
        for raw_line in text.splitlines():
            # 去掉每行首尾空白。
            line = raw_line.strip()
            # 空行不保留。
            if not line:
                continue
            # 删除纯页码，例如 “1” 或 “第 1 页”。
            if re.fullmatch(r"第?\s*\d+\s*页?", line):
                continue
            # 删除重复短行页眉页脚，但保留包含“第X条”的法律条文标题。
            if line in repeated_lines and not re.search(r"第[一二三四五六七八九十百千万\d]+条", line):
                continue
            # 删除明显的目录噪声行。
            if re.fullmatch(r"目\s*录", line):
                continue
            # 保留当前行。
            kept_lines.append(line)
        # 使用换行拼回文本，保留法律条文的层次感。
        return "\n".join(kept_lines)


