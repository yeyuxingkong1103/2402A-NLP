"""文档解析与分块：.txt 直接读、.docx 用 python-docx、.pdf 用 pdfplumber（无文字时 OCR 兜底）。

纯同步、无 LLM；OCR 为可选依赖，未启用时对扫描件抛出 DocumentParseError。
"""
from io import BytesIO

from ..config import get_settings

# VARCHAR 字段上限，单块文本不得超过
MAX_CHUNK_LEN = 4096


class DocumentParseError(Exception):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "txt":
        return _read_txt(data)
    if ext == "docx":
        return _read_docx(data)
    if ext == "pdf":
        text = _read_pdf(data)
        if text.strip():
            return text
        return _ocr_text(data)
    raise DocumentParseError(f"不支持的文件类型: {ext or '未知'}")


def _read_txt(data: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_docx(data: bytes) -> str:
    import docx

    doc = docx.Document(BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts)


def _read_pdf(data: bytes) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _ocr_text(data: bytes) -> str:
    s = get_settings()
    if not s.ocr_enabled:
        raise DocumentParseError("扫描件无文字层，需启用 OCR（安装 paddleocr extra 并设置 OCR_ENABLED=true）")
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise DocumentParseError("未安装 paddleocr，无法 OCR 扫描件")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = ocr.ocr(data, cls=True)
    lines = []
    for page in result or []:
        if not page:
            continue
        for item in page:
            # item: [box, (text, confidence)]
            lines.append(item[1][0])
    return "\n".join(lines)


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """按空行分段落，贪心打包到约 size 字符，相邻块重叠 overlap，单块不超过 MAX_CHUNK_LEN。"""
    if size <= overlap:
        overlap = 0
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # 单段过长则直接截断（并尽量避开 MAX 上限）
        while len(para) > size:
            cut = para[:size]
            chunks.append(cut[:MAX_CHUNK_LEN])
            para = para[size - overlap:] if overlap else para[size:]

        if not buf:
            buf = para
            continue
        if len(buf) + 2 + len(para) <= size:
            buf = buf + "\n\n" + para
        else:
            chunks.append(buf[:MAX_CHUNK_LEN])
            # 重叠：新块以上一块尾部开头
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + para) if tail else para

    if buf.strip():
        chunks.append(buf[:MAX_CHUNK_LEN])
    return [c for c in chunks if c.strip()]
