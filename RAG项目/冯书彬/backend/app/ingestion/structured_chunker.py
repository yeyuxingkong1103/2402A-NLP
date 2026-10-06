"""按法律文档结构生成父块和子块。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


MAX_CHILD_CHARS = 1200
CHILD_OVERLAP_CHARS = 160

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_LEGAL_SECTION_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+[章节篇部编卷].*$")
_ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+条(?:\s|$)")


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    index: int
    content: str
    structure_type: str


@dataclass
class ParentChunk:
    id: str
    index: int
    title: str
    structure_type: str
    content: str
    children: list[ChildChunk] = field(default_factory=list)


def build_parent_child_chunks(text: str) -> list[ParentChunk]:
    lines = [line.strip() for line in text.splitlines()]
    lines = _remove_empty_edges(lines)
    if not lines:
        return []
    units = _split_structural_units(lines)
    parents: list[ParentChunk] = []
    for index, (title, structure_type, unit_lines) in enumerate(units):
        content = "\n".join(unit_lines).strip()
        parent_id = f"parent-{index}"
        parent = ParentChunk(parent_id, index, title, structure_type, content)
        parent.children = _build_children(parent)
        parents.append(parent)
    return parents


def flatten_chunks(parents: list[ParentChunk]) -> list[tuple[str, str, str | None, int, str, str]]:
    """返回兼容 document_chunks 的记录：id、正文、父 ID、序号、层级、结构类型。"""
    rows: list[tuple[str, str, str | None, int, str, str]] = []
    chunk_index = 0
    for parent in parents:
        rows.append((parent.id, parent.content, None, chunk_index, "parent", parent.structure_type))
        chunk_index += 1
        for child in parent.children:
            rows.append((child.id, child.content, parent.id, chunk_index, "child", child.structure_type))
            chunk_index += 1
    return rows


def _split_structural_units(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    units: list[tuple[str, str, list[str]]] = []
    current_lines: list[str] = []
    current_title = "全文"
    current_type = "document"
    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading or _LEGAL_SECTION_RE.match(line) or _ARTICLE_RE.match(line):
            if current_lines:
                units.append((current_title, current_type, current_lines))
            current_title = heading.group(1) if heading else line
            current_type = _structure_type(line, heading is not None)
            current_lines = [line]
            continue
        current_lines.append(line)
    if current_lines:
        units.append((current_title, current_type, current_lines))
    return units


def _build_children(parent: ParentChunk) -> list[ChildChunk]:
    paragraphs = _paragraphs(parent.content)
    children: list[ChildChunk] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= MAX_CHILD_CHARS:
            current = candidate
            continue
        if current:
            children.extend(_make_child_chunks(parent, current))
        current = paragraph
    if current:
        children.extend(_make_child_chunks(parent, current))
    for index, child in enumerate(children):
        child.index = index
        child.id = f"{parent.id}:child-{index}"
    return children


def _make_child_chunks(parent: ParentChunk, content: str) -> list[ChildChunk]:
    if len(content) <= MAX_CHILD_CHARS:
        return [ChildChunk(f"{parent.id}:child-0", parent.id, 0, content, parent.structure_type)]
    result: list[ChildChunk] = []
    start = 0
    index = 0
    while start < len(content):
        end = min(start + MAX_CHILD_CHARS, len(content))
        result.append(ChildChunk(f"{parent.id}:child-{index}", parent.id, index, content[start:end], parent.structure_type))
        if end == len(content):
            break
        start = max(end - CHILD_OVERLAP_CHARS, start + 1)
        index += 1
    return result


def _paragraphs(content: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]


def _structure_type(line: str, is_markdown_heading: bool) -> str:
    if _ARTICLE_RE.match(line):
        return "article"
    if _LEGAL_SECTION_RE.match(line):
        return "section"
    return "heading" if is_markdown_heading else "document"


def _remove_empty_edges(lines: list[str]) -> list[str]:
    start = next((index for index, line in enumerate(lines) if line), len(lines))
    end = next((index for index in range(len(lines) - 1, -1, -1) if lines[index]), -1)
    return lines[start : end + 1]
