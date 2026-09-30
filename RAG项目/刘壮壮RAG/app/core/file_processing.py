from pathlib import Path

from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def classify_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"不支持的文件类型: {ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_text(path: str, file_type: str) -> str:
    if file_type == "text":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    if file_type == "pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"无法提取文本的类型: {file_type}")


_ocr = None


def ocr_image(path: str) -> str:
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    result, _ = _ocr(path)
    if not result:
        return ""
    return "".join(item[1] for item in result)
