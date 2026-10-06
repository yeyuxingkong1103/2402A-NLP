# -*- coding: utf-8 -*-
"""文件/数据类型识别模块（纯标准库，无第三方依赖）。

识别一个文件或一段字节流属于什么类型，并给出建议的 RAG 解析器，
用于在数据入库前做路由：PDF → MinerU、图片 → OCR、文本/Markdown → 分块等。

识别策略（按优先级）：
1. 空文件：大小为零；
2. 魔数：读取文件头签名（如 ``%PDF-``、``\\x89PNG``、``PK\\x03\\x04`` 等）；
   zip 容器会进一步打开判断是 docx/xlsx/pptx/epub 还是普通压缩包；
3. 文本判定：无魔数时，用 NUL 字节 / 控制字符占比判断是否为文本；
4. 内容嗅探：对文本做编码探测，并按首字符/扩展名识别
   JSON / JSONL / CSV / XML / HTML / Markdown / 代码 / 纯文本；
5. 二进制兜底：既无魔数也无法按文本识别时，回退到扩展名或 binary。

用法示例：

    from data_type import detect, detect_bytes, scan_directory, summarize

    info = detect("农业知识.pdf")
    print(info.kind, info.mime, info.loader)      # pdf application/pdf mineru_parser.parse_pdf

    info = detect_bytes(open("a.json", "rb").read(), name="a.json")
    print(info.kind)                              # json

    infos = scan_directory("data", recursive=True)
    print(summarize(infos))                       # {'pdf': 3, 'markdown': 1, ...}
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag2.data_type")

# 分析时最多读取的字节数（足够魔数与文本嗅探，避免大文件整读）
_MAX_READ = 8 * 1024 * 1024

# 魔数签名表：(前缀字节, 类别, MIME)。RIFF 单独处理（WebP/WAV）。
_MAGIC: list[tuple[bytes, str, str]] = [
    (b"%PDF-", "pdf", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image", "image/png"),
    (b"\xff\xd8\xff", "image", "image/jpeg"),
    (b"GIF87a", "image", "image/gif"),
    (b"GIF89a", "image", "image/gif"),
    (b"BM", "image", "image/bmp"),
    (b"II*\x00", "image", "image/tiff"),
    (b"MM\x00*", "image", "image/tiff"),
    (b"PK\x03\x04", "archive", "application/zip"),   # 含 OOXML / epub
    (b"PK\x05\x06", "archive", "application/zip"),
    (b"PK\x07\x08", "archive", "application/zip"),
    (b"\x1f\x8b", "archive", "application/gzip"),
    (b"7z\xbc\xaf\x27\x1c", "archive", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "archive", "application/vnd.rar"),
    (b"\x7fELF", "binary", "application/x-elf"),
    (b"SQLite format 3\x00", "binary", "application/x-sqlite3"),
    (b"\x00\x00\x01\x00", "image", "image/x-icon"),  # ICO
]

# 扩展名 → (类别, MIME)：作为无魔数时的兜底与文本子类判定的依据
_EXT_MAP: dict[str, tuple[str, str]] = {
    "pdf": ("pdf", "application/pdf"),
    "png": ("image", "image/png"), "jpg": ("image", "image/jpeg"),
    "jpeg": ("image", "image/jpeg"), "gif": ("image", "image/gif"),
    "bmp": ("image", "image/bmp"), "webp": ("image", "image/webp"),
    "tif": ("image", "image/tiff"), "tiff": ("image", "image/tiff"),
    "svg": ("image", "image/svg+xml"),
    "md": ("markdown", "text/markdown"), "markdown": ("markdown", "text/markdown"),
    "txt": ("text", "text/plain"), "log": ("text", "text/plain"),
    "json": ("json", "application/json"), "jsonl": ("jsonl", "application/x-ndjson"),
    "ndjson": ("jsonl", "application/x-ndjson"),
    "csv": ("csv", "text/csv"), "tsv": ("csv", "text/tab-separated-values"),
    "xml": ("xml", "application/xml"),
    "html": ("html", "text/html"), "htm": ("html", "text/html"),
    "docx": ("office", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "xlsx": ("office", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "pptx": ("office", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    "doc": ("office", "application/msword"),
    "xls": ("office", "application/vnd.ms-excel"),
    "ppt": ("office", "application/vnd.ms-powerpoint"),
    "epub": ("archive", "application/epub+zip"),
    "zip": ("archive", "application/zip"),
    "gz": ("archive", "application/gzip"),
    "7z": ("archive", "application/x-7z-compressed"),
    "rar": ("archive", "application/vnd.rar"),
    "yaml": ("text", "text/yaml"), "yml": ("text", "text/yaml"),
    "mp3": ("audio", "audio/mpeg"), "wav": ("audio", "audio/wav"),
    "mp4": ("video", "video/mp4"), "avi": ("video", "video/x-msvideo"),
}

# 代码类扩展名（用于纯文本的进一步归类）
_CODE_EXTS = {
    "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "cc", "h", "hpp",
    "cs", "go", "rs", "php", "rb", "swift", "kt", "sh", "bash", "ps1",
    "css", "scss", "sql", "r", "m", "vue",
}

# 类别 → 建议的 RAG 解析器（loader）与说明
_LOADERS: dict[str, tuple[str | None, str]] = {
    "pdf": ("rag2.mineru_parser.parse_pdf", "用 MinerU 解析"),
    "image": ("rag2.ocr.ocr_image", "用 OCR（PaddleOCR-VL）识别为 txt"),
    "text": ("rag2.pipeline.chunk_text", "直接文本分块"),
    "markdown": ("rag2.pipeline.chunk_text", "Markdown 分块"),
    "json": ("json_loader", "用 json 解析"),
    "jsonl": ("jsonl_loader", "逐行 json 解析"),
    "csv": ("rag2.pdf_table.extract_tables", "用 pdfplumber 抽表格（若是 csv 文件则用 csv 解析）"),
    "xml": ("xml_loader", "用 XML 解析"),
    "html": ("html_loader", "用 HTML 解析 / 抽取文本"),
    "office": ("rag2.mineru_parser.parse_pdf", "docx/pptx 可用 MinerU；xlsx 建议 rag2.pdf_table"),
    "archive": ("unzip", "用 zipfile 读取条目"),
    "code": ("text_loader", "按文本读取"),
    "audio": (None, "音频，非文本"),
    "video": (None, "视频，非文本"),
    "binary": (None, "二进制，需专用工具"),
    "empty": (None, "空文件"),
    "unknown": (None, "未知类型"),
}

# 类别 → 中文标签
_KIND_LABELS: dict[str, str] = {
    "pdf": "PDF 文档", "image": "图片", "text": "纯文本", "markdown": "Markdown",
    "json": "JSON", "jsonl": "JSON Lines", "csv": "CSV/表格", "xml": "XML",
    "html": "HTML", "office": "Office 文档", "archive": "压缩包", "code": "代码",
    "audio": "音频", "video": "视频", "binary": "二进制", "empty": "空文件",
    "unknown": "未知",
}

_TEXT_MIME: dict[str, str] = {
    "text": "text/plain", "markdown": "text/markdown", "json": "application/json",
    "jsonl": "application/x-ndjson", "csv": "text/csv", "xml": "application/xml",
    "html": "text/html", "code": "text/plain",
}


@dataclass
class FileInfo:
    """文件类型识别结果。"""

    name: str
    kind: str
    mime: str
    size: int
    extension: str = ""
    encoding: str | None = None
    loader: str | None = None
    loader_note: str = ""
    confidence: str = "unknown"   # magic / content / extension / size
    detail: str = ""

    @property
    def label(self) -> str:
        """中文类别标签。"""
        return _KIND_LABELS.get(self.kind, self.kind)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "kind": self.kind,
            "label": self.label,
            "mime": self.mime,
            "extension": self.extension,
            "size": self.size,
            "encoding": self.encoding,
            "loader": self.loader,
            "loader_note": self.loader_note,
            "confidence": self.confidence,
            "detail": self.detail,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试打印
        return (f"FileInfo({self.name!r}, kind={self.kind}, mime={self.mime}, "
                f"loader={self.loader})")


# ---------------------------------------------------------------------- 底层判定
def _is_binary(data: bytes) -> bool:
    """粗判二进制：含 NUL 字节或控制字符占比过高。"""
    sample = data[:1024]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    ctrl = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    return ctrl / len(sample) > 0.1


def _decode_text(data: bytes) -> tuple[str | None, str | None]:
    """探测并解码文本，返回 (文本, 编码)；失败返回 (None, None)。"""
    # BOM 优先
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace"), "utf-8-sig"
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", "replace"), "utf-16"
    if data.startswith(b"\xff\xfe\x00\x00") or data.startswith(b"\x00\x00\xfe\xff"):
        return data.decode("utf-32", "replace"), "utf-32"
    # 常见编码逐一尝试
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, None


def _looks_like_markdown(text: str) -> bool:
    """轻量 Markdown 特征判断。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()][:20]
    if not lines:
        return False
    score = 0
    for line in lines:
        if re.match(r"^#{1,6}\s", line):
            score += 2
        elif line.startswith(("- ", "* ", "+ ", "> ", "```", "|")):
            score += 1
        elif re.search(r"\*\*.+\*\*|__.+__|\[.+\]\(.+\)", line):
            score += 1
    return score >= 3


def _sniff_text(text: str, ext: str) -> str:
    """对已解码文本做归类：扩展名优先，内容嗅探兜底。"""
    stripped = text.lstrip()
    if not stripped:
        return "text"

    # 1) 扩展名优先（文本子类的强信号）
    if ext == "json":
        return "json"
    if ext in ("jsonl", "ndjson"):
        return "jsonl"
    if ext in ("csv", "tsv"):
        return "csv"
    if ext in ("md", "markdown"):
        return "markdown"
    if ext in _CODE_EXTS:
        return "code"
    if ext in ("xml",):
        return "xml"
    if ext in ("html", "htm"):
        return "html"
    if ext in ("yaml", "yml", "txt", "log"):
        return "text"

    # 2) 无扩展名 / 未知扩展名 → 内容嗅探
    if stripped[0] in "{[":
        return "json"
    if stripped.startswith("<?xml"):
        return "xml"
    low = stripped[:200].lower()
    if low.startswith("<!doctype html") or "<html" in low:
        return "html"
    if _looks_like_markdown(text):
        return "markdown"
    return "text"


def _classify_zip(data: bytes, ext: str) -> tuple[str, str]:
    """区分 zip 容器：OOXML 文档 / epub / 普通压缩包。"""
    if ext == "docx":
        return "office", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == "xlsx":
        return "office", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if ext == "pptx":
        return "office", "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if ext == "epub":
        return "archive", "application/epub+zip"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
        if "META-INF/container.xml" in names:
            return "archive", "application/epub+zip"
        if any(n.startswith("word/") for n in names):
            return "office", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if any(n.startswith("xl/") for n in names):
            return "office", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if any(n.startswith("ppt/") for n in names):
            return "office", "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        return "archive", "application/zip"
    except Exception:
        return "archive", "application/zip"


def _match_magic(data: bytes, ext: str) -> tuple[str | None, str | None, str]:
    """魔数匹配，返回 (类别, MIME, 置信度)；未命中返回 (None, None, "")。"""
    head = data[:16]
    # RIFF 容器：WebP / WAV
    if head.startswith(b"RIFF") and len(data) >= 12:
        tag = data[8:12]
        if tag == b"WEBP":
            return "image", "image/webp", "magic"
        if tag == b"WAVE":
            return "audio", "audio/wav", "magic"
    for sig, kind, mime in _MAGIC:
        if head.startswith(sig):
            if sig.startswith(b"PK"):  # 仅 zip 容器进一步区分 OOXML/epub
                kind, mime = _classify_zip(data, ext)
            return kind, mime, "magic"
    return None, None, ""


def _build(name: str, ext: str, size: int, kind: str, mime: str,
           encoding: str | None, confidence: str, detail: str) -> FileInfo:
    loader, note = _LOADERS.get(kind, (None, ""))
    return FileInfo(
        name=name, kind=kind, mime=mime, size=size, extension=ext,
        encoding=encoding, loader=loader, loader_note=note,
        confidence=confidence, detail=detail,
    )


def _analyze(data: bytes, name: str, size: int) -> FileInfo:
    ext = Path(name).suffix.lower().lstrip(".") if name else ""

    if size == 0:
        return _build(name, ext, size, "empty", "application/x-empty",
                      None, "size", "空文件")

    kind, mime, confidence = _match_magic(data, ext)
    if kind is not None:
        return _build(name, ext, size, kind, mime, None, confidence, "魔数匹配")

    # 无魔数 → 判断是否文本
    if _is_binary(data):
        kind, mime = "binary", "application/octet-stream"
        if ext in _EXT_MAP:
            ek, em = _EXT_MAP[ext]
            if ek not in ("text", "markdown", "json", "jsonl", "csv", "xml", "html", "code"):
                kind, mime = ek, em
        conf = "extension" if ext in _EXT_MAP else "magic"
        return _build(name, ext, size, kind, mime, None, conf, "无匹配魔数，按二进制处理")

    text, encoding = _decode_text(data)
    if text is None:
        return _build(name, ext, size, "binary", "application/octet-stream",
                      None, "magic", "无法解码为文本")

    kind = _sniff_text(text, ext)
    mime = _TEXT_MIME.get(kind, "text/plain")
    return _build(name, ext, size, kind, mime, encoding, "content", "按文本内容识别")


# ---------------------------------------------------------------------- 对外接口
def detect(path: str | Path) -> FileInfo:
    """识别单个文件的类型。

    参数：
        path: 文件路径。

    返回：
        FileInfo 对象。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    size = p.stat().st_size
    with open(p, "rb") as f:
        data = f.read(_MAX_READ)
    return _analyze(data, name=p.name, size=size)


def detect_bytes(data: bytes | str, name: str | None = None) -> FileInfo:
    """识别一段字节流的类型（如从 zip/数据库中读出的内容）。

    参数：
        data: 字节内容（若传入 str 会按 UTF-8 编码）。
        name: 可选文件名，用于扩展名辅助判定。

    返回：
        FileInfo 对象。
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _analyze(bytes(data), name=name or "", size=len(data))


def scan_directory(path: str | Path, recursive: bool = True,
                   include_hidden: bool = False) -> list[FileInfo]:
    """扫描目录下所有文件的类型，返回按路径排序的结果列表。

    参数：
        path:           目录路径。
        recursive:      是否递归子目录。
        include_hidden: 是否包含隐藏文件（以 ``.`` 开头）。

    返回：
        FileInfo 列表。
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在：{root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    infos: list[FileInfo] = []
    for p in sorted(iterator):
        if not p.is_file():
            continue
        if not include_hidden and p.name.startswith("."):
            continue
        infos.append(detect(p))
    return infos


def summarize(infos: list[FileInfo]) -> dict[str, int]:
    """统计各类别的文件数量。

    参数：
        infos: detect/scan_directory 返回的 FileInfo 列表。

    返回：
        {类别: 数量}，按类别名排序。
    """
    counts: dict[str, int] = {}
    for info in infos:
        counts[info.kind] = counts.get(info.kind, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["FileInfo", "detect", "detect_bytes", "scan_directory", "summarize"]


if __name__ == "__main__":  # pragma: no cover - 自测，无需外部服务
    import tempfile

    demo_dir = Path(tempfile.mkdtemp(prefix="data_type_demo_"))
    samples = {
        "doc.pdf": b"%PDF-1.7 fake pdf content",
        "pic.png": b"\x89PNG\r\n\x1a\n fake png",
        "data.json": '{"name": "测试", "count": 3}'.encode("utf-8"),
        "table.csv": "name,age\n张三,20\n李四,30".encode("utf-8"),
        "note.md": "# 标题\n\n这是 **Markdown** 内容。\n- 项目一\n- 项目二".encode("utf-8"),
        "script.py": b"def hello():\n    print('hi')",
        "plain.txt": "纯文本内容".encode("utf-8"),
        "empty.bin": b"",
    }
    for fname, content in samples.items():
        (demo_dir / fname).write_bytes(content)

    for info in scan_directory(demo_dir, recursive=False):
        d = info.to_dict()
        print(f"{d['name']:14s} -> {d['kind']:8s} ({d['label']}) "
              f"loader={d['loader'] or '-'}")

    print("\n类别统计:", summarize(scan_directory(demo_dir, recursive=False)))
