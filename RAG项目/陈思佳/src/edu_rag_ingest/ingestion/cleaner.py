from __future__ import annotations

"""原始文档文本清洗工具，负责去除噪声并修复不连续文本。"""

import re

_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_PAGE_PATTERN = re.compile(r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页")
_STANDALONE_PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_EXTRA_BLANK_PATTERN = re.compile(r"\n{3,}")
_SPACE_PATTERN = re.compile(r"[ \t]+")
_TOC_DOTS_PATTERN = re.compile(r"[·•.．…]{3,}\s*\d+\s*$")
_MARKDOWN_PAGE_BREAK_PATTERN = re.compile(r"^-{3,}\s*$", re.MULTILINE)


def clean_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _IMAGE_PATTERN.sub("", cleaned)
    cleaned = _MARKDOWN_PAGE_BREAK_PATTERN.sub("", cleaned)
    cleaned = _PAGE_PATTERN.sub("", cleaned)
    cleaned = _STANDALONE_PAGE_NUMBER_PATTERN.sub("", cleaned)
    cleaned = _remove_repeated_lines(cleaned)
    cleaned = _normalize_lines(cleaned)
    cleaned = _remove_table_of_contents_block(cleaned)
    cleaned = _merge_broken_chinese_lines(cleaned)
    cleaned = _SPACE_PATTERN.sub(" ", cleaned)
    cleaned = _EXTRA_BLANK_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()


def _normalize_lines(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        line = line.replace("（ ", "（").replace(" ）", "）")
        line = line.replace("( ", "(").replace(" )", ")")
        line = re.sub(r"^#+\s*", lambda match: match.group(0).replace("　", " "), line)
        line = re.sub(r"##\s*([一二三四五六七八九十]+)、", r"## \1、", line)
        line = re.sub(r"##\s*([0-9]+)[.、]", r"### \1. ", line)
        if _is_low_value_line(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _is_low_value_line(line: str) -> bool:
    if line in {"目 录", "目录", "前言", "下载", "收藏"}:
        return False
    if _TOC_DOTS_PATTERN.search(line):
        return True
    if re.fullmatch(r"[、，。；：,.\s]+", line):
        return True
    if re.fullmatch(r"[.．…·\-—_\s]{3,}", line):
        return True
    return False


def _remove_table_of_contents_block(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_toc = False
    toc_lines_seen = 0

    for line in lines:
        stripped = line.strip()
        if stripped in {"## 目 录", "## 目录", "# 目 录", "# 目录"}:
            in_toc = True
            toc_lines_seen = 0
            continue

        if in_toc:
            toc_lines_seen += 1
            if stripped.startswith("## 一、") or stripped.startswith("# 一、"):
                in_toc = False
                output.append(stripped)
            elif toc_lines_seen > 40 and stripped.startswith("## "):
                in_toc = False
                output.append(stripped)
            continue

        output.append(line)

    return "\n".join(output)


def _remove_repeated_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    counts: dict[str, int] = {}
    for line in lines:
        if 2 <= len(line) <= 40 and not line.startswith("#"):
            counts[line] = counts.get(line, 0) + 1

    repeated = {line for line, count in counts.items() if count >= 3}
    return "\n".join(line for line in lines if line not in repeated)


def _merge_broken_chinese_lines(text: str) -> str:
    lines = text.splitlines()
    merged: list[str] = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")
            continue

        if stripped.startswith("#") or stripped.startswith("|"):
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(stripped)
            continue

        if buffer and not re.search(r"[。！？；：.!?;:]$", buffer):
            buffer += stripped
        else:
            if buffer:
                merged.append(buffer)
            buffer = stripped

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)
