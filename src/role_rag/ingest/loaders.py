"""知识文档加载：Markdown / 文本 / PDF。

支持极简 front-matter（文件开头 ``---`` 包裹的 key: value），
可用于覆盖标题与标签，例如：

    ---
    title: 资产配置基础
    tags: 资产配置, 风险, 复利
    ---
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..errors import IngestError
from ..logging_conf import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(slots=True)
class LoadedDoc:
    """一篇已加载的原始文档。"""

    doc_id: str
    title: str
    text: str
    source: str
    role: str
    scope: str
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return len(self.text)


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if key.strip() == "tags":
            meta["tags"] = [item.strip() for item in re.split(r"[,，]", value) if item.strip()]
        else:
            meta[key.strip()] = value
    return meta, text[match.end():]


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _clean_text(text: str) -> str:
    """轻量清洗：统一换行、压缩多余空行、去掉零宽字符。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> str:
    pages: list[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                content = page.extract_text() or ""
                if content.strip():
                    pages.append(f"<!-- 第 {index} 页 -->\n{content.strip()}")
    except Exception as exc:  # pragma: no cover - 回退到 PyMuPDF
        logger.warning("pdfplumber 解析失败（%s），尝试 PyMuPDF：%s", path.name, exc)
        try:
            import fitz

            with fitz.open(str(path)) as doc:
                for index, page in enumerate(doc, start=1):
                    content = page.get_text("text") or ""
                    if content.strip():
                        pages.append(f"<!-- 第 {index} 页 -->\n{content.strip()}")
        except Exception as inner:  # pragma: no cover
            raise IngestError(f"PDF 解析失败：{path}（{inner}）") from inner
    return "\n\n".join(pages)


def load_document(path: Path, role: str, scope: str) -> LoadedDoc:
    """加载单个文件为 LoadedDoc。"""

    path = Path(path)
    if not path.is_file():
        raise IngestError(f"文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"不支持的文件类型：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")

    if suffix == ".pdf":
        raw = _load_pdf(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    meta, body = _parse_front_matter(raw)
    text = _clean_text(body)
    if not text:
        raise IngestError(f"文件内容为空：{path}")

    title = str(meta.get("title") or _first_heading(text) or path.stem)
    tags = list(meta.get("tags") or [])
    source = str(path)
    doc_id = hashlib.sha1(f"{scope}|{source}".encode("utf-8")).hexdigest()[:16]
    return LoadedDoc(
        doc_id=doc_id,
        title=title,
        text=text,
        source=source,
        role=str(meta.get("role") or role),
        scope=str(meta.get("scope") or scope),
        tags=tags,
        extra={"suffix": suffix, "mtime": path.stat().st_mtime},
    )


def discover_kb_files(kb_root: Path, scope: str) -> list[Path]:
    """列出某个知识库目录下的全部支持文件。"""

    directory = Path(kb_root) / scope
    if not directory.is_dir():
        return []
    files = [path for path in sorted(directory.rglob("*")) if path.suffix.lower() in SUPPORTED_SUFFIXES]
    return [path for path in files if path.is_file() and not path.name.startswith(".")]


def iter_documents(kb_root: Path, scope: str) -> Iterator[LoadedDoc]:
    for path in discover_kb_files(kb_root, scope):
        try:
            yield load_document(path, role=scope, scope=scope)
        except IngestError as exc:
            logger.error("跳过无法加载的文件：%s", exc)


def summarise_documents(docs: Sequence[LoadedDoc]) -> dict[str, Any]:
    return {
        "documents": len(docs),
        "chars": sum(doc.char_len for doc in docs),
        "titles": [doc.title for doc in docs],
    }
