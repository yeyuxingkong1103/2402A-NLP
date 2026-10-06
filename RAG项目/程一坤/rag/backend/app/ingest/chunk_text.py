"""切块用的文本工具（自 chunker.py 拆出）。

三个无状态小工具，被 chunker（块构造）、paragraph_items（项切分）共同
使用——独立成模块避免 chunker ⇄ paragraph_items 循环导入：
- _normalize_text：统一清理（压行内空白、压多余换行，保留段落边界）
- _build_retrieval_content：拼接检索增强文本（标题 + 条号 + 正文）
- _document_key：稳定文档标识（块 ID 前缀）
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _normalize_text(content: str) -> str:
    """统一清理文本：只压行内空白与多余换行，保留段落换行。"""
    # 行内连续空格/制表符/全角空格压成单个空格（不碰换行）
    content = re.sub(r"[ \t\u3000]+", " ", content)
    # 3 个以上连续换行压成 1 个（保留 1-2 个换行作为段落边界）
    content = re.sub(r"\n{3,}", "\n", content)
    return content.strip()


def _build_retrieval_content(
    document_title: str,
    article_no: str | None,
    content: str,
) -> str:
    """构造包含法规标题和条文上下文的检索文本。"""
    # 按已有信息组装检索上下文
    context_parts = [
        part for part in (document_title, article_no, content) if part
    ]

    # 使用空格连接上下文和正文
    return " ".join(context_parts)


def _document_key(source_path: Path, document_id: str | None) -> str:
    """生成稳定的文档级 ID 前缀。"""
    value = document_id or str(source_path.resolve())
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    if not document_id:
        return digest

    readable_prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    if not readable_prefix:
        readable_prefix = "document"
    return f"{readable_prefix[:64]}-{digest}"
