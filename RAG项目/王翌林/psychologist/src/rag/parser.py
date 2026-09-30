"""离线文档解析：PDF(MinerU优先/PyMuPDF/pdfplumber/PaddleOCR回退) / TXT / MD / DOCX，含清洗、去水印、去重。

PDF 解析链路（按优先级）：
  1. MinerU（settings.mineru_enabled）—— 文本型 PDF 高质量解析，默认 tier=flash
  2. PyMuPDF —— 经典 PDF 文本层提取
  3. pdfplumber —— PyMuPDF 失败时回退
  4. PaddleOCR（settings.ocr_fallback_enabled）—— 文本层 < 阈值时对扫描件做 OCR

OCR / MinerU 均通过 src/rag/ocr_loader.py 调用 WSL 中的独立 venv，
依赖缺失时自动降级，不影响主流程。
"""
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from src.core.logging import get_logger

logger = get_logger("rag.parser")

SUPPORTED_TYPES = {"pdf", "txt", "md", "markdown", "docx"}
_WATERMARK_PATTERNS = [
    re.compile(r"^\s*(第\s*\d+\s*页|Page\s*\d+(\s*of\s*\d+)?)\s*$", re.I),
    re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$"),
    re.compile(r"(微信|公众号|扫码关注|免费下载|www\.[a-z0-9\-\.]+\.(com|cn|net|org))", re.I),
    re.compile(r"^\s*(仅供个人学习|请勿传播|版权归.*所有).*$"),
]
# 扫描件常见的不可信元数据标题（PDF 内置 Title 字段）
_JUNK_TITLES = re.compile(
    r"^(ssreader\s*print|untitled|microsoft\s*word|document\d*|camscanner|扫描全能王|image)", re.I
)


def _clean_title(raw: str, path: str) -> str:
    """选择文档标题：元数据标题不可信（扫描件常带 SSReader Print. 之类）时回退到文件名。

    扫描件的 PDF 内置 Title 字段往往是生成软件默认值（如 "SSReader Print"、"Untitled"），
    不可信；此时用文件名兜底，保证标题有实际语义。
    """
    title = re.sub(r"[\x00-\x1f\u200b\ufeff]", "", raw or "").strip()
    if len(title) < 3 or _JUNK_TITLES.match(title):
        title = os.path.splitext(os.path.basename(path))[0]
    return title


@dataclass
class ParsedDocument:
    title: str
    text: str
    file_type: str
    file_path: str
    pages: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------- 清洗 ----------------
def clean_text(text: str) -> str:
    """统一编码、去控制字符、去水印行、合并空白。

    清洗是入库前的必要步骤：扫描件/网页抓取的文本常混入零宽字符、控制字符、
    水印（页码/公众号/版权声明）与多余空白，不清理会污染向量质量与分块效果。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\x00", "")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(p.search(stripped) for p in _WATERMARK_PATTERNS):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"[ \t\u3000]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repeated_lines(pages: List[str], min_ratio: float = 0.6) -> List[str]:
    """去除页眉页脚类重复行（出现频率超过阈值的短行）。

    电子书每页顶部/底部常重复出现书名、章节名、页码等"页眉页脚"；
    它们出现在绝大多数页面且都是短行（≤40 字），用出现次数阈值即可识别剔除，
    避免这些噪声行被反复切块入库、稀释检索质量。
    """
    if len(pages) < 3:
        return pages
    counter: Counter = Counter()
    for page in pages:
        for line in {l.strip() for l in page.splitlines() if 0 < len(l.strip()) <= 40}:
            counter[line] += 1
    threshold = max(3, int(len(pages) * min_ratio))
    repeated = {line for line, cnt in counter.items() if cnt >= threshold}
    if not repeated:
        return pages
    cleaned = []
    for page in pages:
        cleaned.append("\n".join(l for l in page.splitlines() if l.strip() not in repeated))
    logger.debug("已剔除重复页眉页脚行 %d 条", len(repeated))
    return cleaned


def deduplicate_paragraphs(text: str) -> str:
    """段落级去重（保留首次出现）。

    多本书/多来源拼接时可能出现完全相同段落；按"去空白后的段落内容"作 key，
    只保留第一次出现，避免同一段落被反复入库造成检索重复与权重失真。
    """
    seen, result = set(), []
    for para in re.split(r"\n\s*\n", text):
        key = re.sub(r"\s+", "", para)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(para.strip())
    return "\n\n".join(result)


# ---------------- 解析器 ----------------
def _parse_pdf(path: str) -> ParsedDocument:
    """PDF 解析的四级降级链：MinerU → PyMuPDF → pdfplumber → PaddleOCR。

    每一级失败或被判定"结果太差"时才尝试下一级，前级成功即短路。
    前三级都读 PDF 内嵌的"文本层"（快但扫瞄件没有文本层）；最后一级
    PaddleOCR 是对页面图像做识别（慢，但能救扫描件）。
    """
    # pages 是贯穿整条降级链的"接力棒"：哪一级解析出非空结果就写进 pages，
    # 后续各级都用 `if not pages` 判断"要不要接手"，因此前一级成功即自动短路。
    pages: List[str] = []
    meta: dict = {}
    total_pages = 0

    # --- 0. MinerU 优先（更高质量的文本型 PDF 解析）---
    # 降级链排序依据是"质量/成本"权衡：MinerU 质量最高但最重，放最前；
    # PyMuPDF 快且几乎零成本，次之；pdfplumber 作为其兜底；PaddleOCR 最慢最贵，
    # 放最后且仅对"疑似扫描件"触发（见下方 is_scanned 双条件判定）。
    try:
        from src.core.config import settings
        if settings.mineru_enabled:
            from src.rag.ocr_loader import mineru_extract_pdf
            mineru_text = mineru_extract_pdf(path)
            if mineru_text and len(mineru_text) >= settings.ocr_min_text_layer_chars:
                logger.info("MinerU 解析 %s 成功，%d 字符，跳过 OCR 回退",
                            os.path.basename(path), len(mineru_text))
                meta = {"parser": "mineru"}
                # 把 MinerU 文本按 \n\n 切为伪"页"，方便后续去页眉页脚
                pages = [p.strip() for p in mineru_text.split("\n\n") if p.strip()]
                total_pages = len(pages)
    except Exception as exc:
        logger.debug("MinerU 不可用（%s），继续传统解析链路", exc)

    # --- 1. PyMuPDF ---
    if not pages:
        try:
            import fitz  # PyMuPDF
            with fitz.open(path) as doc:
                meta = dict(doc.metadata or {})
                total_pages = len(doc)
                pages = [(p.get_text("text") or "") for p in doc]
            logger.info("PyMuPDF 解析 %s，共 %d 页", os.path.basename(path), len(pages))
        except Exception as exc:
            logger.warning("PyMuPDF 解析失败（%s），尝试 pdfplumber", exc)
            pages = []

    # --- 2. pdfplumber 回退 ---
    text_total = sum(len(p.strip()) for p in pages)
    if text_total < 100 and not pages:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                pages = [(p.extract_text() or "") for p in pdf.pages]
            logger.info("pdfplumber 解析 %s，共 %d 页", os.path.basename(path), len(pages))
        except Exception as exc:
            logger.error("pdfplumber 解析失败：%s", exc)

    # --- 3. PaddleOCR 扫描件回退 ---
    # 扫描件判定（双条件，"或"关系，命中任一即认为是扫描件）：
    #   ① 文本层总字符 < 阈值：整份 PDF 几乎没抽到字，最典型的扫描件特征；
    #   ② 页数 ≥ 10 且有效页比 < 5%：只有极少数页有实质文字（>20 字符），
    #      多半是封面/前言有文字层、正文全是图片。加 total_pages>=10 是为了
    #      避免 1~2 页的短文档因内容少而被误判成扫描件、白白跑一遍昂贵 OCR。
    text_total = sum(len(p.strip()) for p in pages)
    non_empty_pages = sum(1 for p in pages if len(p.strip()) > 20)
    effective_ratio = non_empty_pages / total_pages if total_pages > 0 else 0
    is_scanned = (
        text_total < settings.ocr_min_text_layer_chars
        or (total_pages >= 10 and effective_ratio < 0.05)
    )
    try:
        from src.core.config import settings
        if settings.ocr_fallback_enabled and is_scanned:
            from src.rag.ocr_loader import paddle_ocr_pdf
            logger.info(
                "PDF 疑似扫描件：文本层 %d 字符，有效页比 %.1f%%（%d/%d），触发 PaddleOCR",
                text_total, effective_ratio * 100, non_empty_pages, total_pages,
            )
            # 自适应超时公式：min(max(配置超时, 页数*3), 7200)
            #   - 下限 = 配置值：小文档也至少给配置的兜底时间；
            #   - 中段 = 3 秒/页线性估算：页数越多给越多时间（1000 页约 3000s，
            #     默认 1800s 对 1000+ 页的书明显不够）；
            #   - 上限 = 7200s(2h)：防止损坏文件把进程无限拖住，
            #     超时后由 ocr_loader 杀整个进程组回收资源。
            ocr_timeout = min(max(settings.ocr_timeout, (total_pages or 0) * 3), 7200)
            ocr_text = paddle_ocr_pdf(path, pages="all", timeout=ocr_timeout)
            if ocr_text and len(ocr_text) >= settings.ocr_min_text_layer_chars:
                pages = [ocr_text]  # OCR 输出整体作为一页
                meta["parser"] = "paddleocr"
                total_pages = 1
                logger.info("PaddleOCR 回退成功：%d 字符", len(ocr_text))
            else:
                logger.warning("PaddleOCR 回退未识别到有效文本（%d 字符），跳过", len(ocr_text))
    except Exception as exc:
        logger.warning("OCR 回退不可用（%s），扫描件内容可能缺失", exc)

    text_total = sum(len(p.strip()) for p in pages)
    if text_total < 100:
        logger.warning("PDF 文本层为空且 OCR 未生效：%s", path)

    # ⚠ 清洗管线顺序不可调换：去页眉页脚必须"在合并成单文本之前"完成。
    # remove_repeated_lines 依赖"分页列表"逐页统计同一行跨页出现的频率，
    # 一旦先 "\n\n".join(pages) 合并成一大段，页边界就丢失、无法再按页统计，
    # 页眉页脚会退化成偶发单行而漏删。故固定顺序：先去重复行 → 再合并 → 最后 clean_text。
    pages = remove_repeated_lines(pages)
    text = clean_text("\n\n".join(pages))
    return ParsedDocument(
        title=_clean_title(meta.get("title") or "", path),
        text=text, file_type="pdf", file_path=path,
        pages=total_pages or len(pages), meta=meta,
    )


def _read_text_file(path: str) -> str:
    """多编码尝试读取文本文件（utf-8 → utf-8-sig → gbk → latin-1），最后忽略错误兜底。

    中文环境下的 TXT 常见 GBK 编码，而 python 默认 utf-8 会抛 UnicodeDecodeError；
    按概率从高到低逐个尝试，最后的 errors="ignore" 保证绝不因编码失败而中断。
    """
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_txt(path: str) -> ParsedDocument:
    """解析纯文本/Markdown 文件（整文件作为一个 page）。"""
    raw = _read_text_file(path)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ParsedDocument(
        title=os.path.splitext(os.path.basename(path))[0],
        text=clean_text(raw), file_type=ext or "txt", file_path=path, pages=1,
    )


def _parse_docx(path: str) -> ParsedDocument:
    """解析 DOCX：抽取段落与表格，表格按 " | " 拼接成一行。"""
    try:
        import docx
    except ImportError:
        logger.error("未安装 python-docx，无法解析 DOCX：%s", path)
        return ParsedDocument(title=os.path.basename(path), text="", file_type="docx",
                              file_path=path, meta={"error": "python-docx 未安装"})
    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return ParsedDocument(
        title=os.path.splitext(os.path.basename(path))[0],
        text=clean_text("\n\n".join(parts)), file_type="docx", file_path=path, pages=1,
    )


def parse_file(path: str, do_dedup: bool = True) -> ParsedDocument:
    """解析文件为纯文本，失败时抛异常由上层记录。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"不支持的文件类型：{ext}（支持 {sorted(SUPPORTED_TYPES)}）")

    if ext == "pdf":
        doc = _parse_pdf(path)
    elif ext == "docx":
        doc = _parse_docx(path)
    else:
        doc = _parse_txt(path)

    if do_dedup:
        doc.text = deduplicate_paragraphs(doc.text)
    logger.info("解析完成 %s：%d 字符", os.path.basename(path), doc.char_count)
    return doc


def parse_directory(directory: str, recursive: bool = True) -> List[ParsedDocument]:
    """批量解析目录下的受支持文档。

    单个文件失败只记录错误、继续下一个（不因一本书损坏而中断整批导入）；
    recursive=False 时只扫顶层，避免误扫嵌套子目录。
    """
    results: List[ParsedDocument] = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower().lstrip(".")
            if ext not in SUPPORTED_TYPES:
                continue
            try:
                results.append(parse_file(os.path.join(root, fname)))
            except Exception as exc:
                logger.error("解析失败 %s：%s", fname, exc)
        if not recursive:
            break
    return results