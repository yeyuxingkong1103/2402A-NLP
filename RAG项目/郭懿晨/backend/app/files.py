import hashlib
from pathlib import Path
from uuid import uuid4


FORBIDDEN_NAME_CHARS = {'\\', '/', ':', '*', '?', '"', '<', '>', '|'}


def pdf_sha256(content: bytes) -> str:
    """计算 PDF 内容的 SHA-256，用于判重。"""
    return hashlib.sha256(content).hexdigest()


def pdf_file_sha256(path: Path) -> str | None:
    """流式计算磁盘 PDF 文件的 SHA-256；文件不可读返回 None。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def sanitize_pdf_name(file_name: str) -> str:
    """清理 PDF 文件名。"""
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("只允许上传 PDF 文件")

    cleaned = "".join("_" if char in FORBIDDEN_NAME_CHARS else char for char in file_name).strip()
    if cleaned in {"", ".pdf"}:
        cleaned = f"{uuid4()}.pdf"
    return cleaned


def save_pdf_bytes(data_dir: Path, file_name: str, content: bytes) -> Path:
    """保存 PDF 字节到 data 目录。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_pdf_name(file_name)
    target = data_dir / f"{uuid4()}-{safe_name}"
    target.write_bytes(content)
    return target
