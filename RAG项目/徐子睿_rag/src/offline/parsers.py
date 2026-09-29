"""src/offline/parsers.py —— 多格式文档解析与文本清洗。

在链路中的位置（新架构的离线侧入口）：
    原始文件 → 【本文件】 → src/offline/chunkers.py 分块 → embedder 向量化 → milvus_store 入库
上游调用方：src/offline/pipeline.py 的 parse_directory()

与 backend/pipeline.py 的关系：
    backend 那条主线只吃 PDF（MinerU/pypdf），本文件是新架构的扩展版 ——
    支持 PDF / DOCX / Markdown / HTML / 纯文本五种格式，且每个解析器都可选依赖：
    缺库时不是崩掉，而是给出明确的安装提示或回退到别的解析器。

所有解析器的输出都统一成同一种结构，后面的分块/向量化就不用关心文件来源：
    [{"page": 页码(从1开始), "text": 清洗后的正文, "metadata": {"parser": "用了哪个解析器"}}, ...]
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any


class ParsedPage(dict):
    """带类型标注的解析结果页（等价于 {"page": int, "text": str, "metadata": dict}）。

    继承 dict 而不是用 dataclass：
        因为下游（分块、向量化、Milvus）都按普通字典处理这些数据，
        继承 dict 能直接传给那些接口，不需要额外转换。
        这里的类属性只是给 IDE 和阅读者看的类型提示。
    """

    page: int
    text: str
    metadata: dict[str, Any]


def clean_text(text: str) -> str:
    """清洗解析出来的原始文本。

    参数：
        text: 解析器给出的原始文本
    返回：
        清洗后的文本（首尾空白已去除）。

    做四件事：
        1. html.unescape 还原实体（&amp; -> &），并去掉不间断空格（\\xa0）——
           它长得像普通空格但会破坏关键词匹配，是最容易被忽略的隐藏字符
        2. 多个空格/制表符合并成一个
        3. 3 个以上连续换行压成 2 个（保留段落分隔，去掉大片空行）
        4. 删掉"第 3 页""Page 3"这类页眉页脚 —— 它们在每页都出现，进了索引只会污染检索

    为什么对两类空白分别处理（先合并水平、再压缩垂直）：
        顺序反了的话，先压缩换行会把"行尾空格+换行"这类组合规则弄乱，
        导致段落边界识别不准。
    """
    text = html.unescape(text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*(?:第\s*\d+\s*页|Page\s+\d+)\s*$", "", text, flags=re.I)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    """用 PyMuPDF(fitz) 提取 PDF 的文本层。

    参数：
        path: PDF 路径
    返回：
        按页的解析结果。
    异常：
        未安装 PyMuPDF -> RuntimeError（带安装命令）；
        PDF 没有任何可提取文本（多为扫描件）-> RuntimeError。

    为什么第二个异常很重要：
        "PDF 打开成功但一个字都没提取到"是扫描件的典型特征。
        不报错的话，后面会拿一堆空文本去分块和向量化，
        最终表现为"上传成功但检索不到任何内容"，极难排查。
        这里直接说清是扫描件问题，用户就知道该走 OCR 路径。
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF 解析需要 PyMuPDF：pip install pymupdf") from exc
    document = fitz.open(path)
    pages = [{"page": index + 1, "text": clean_text(page.get_text("text")), "metadata": {"parser": "pymupdf"}} for index, page in enumerate(document)]
    if any(item["text"] for item in pages):
        return pages
    raise RuntimeError(f"PDF 没有可提取文本：{path}")


def parse_pdf_tables(path: Path) -> list[dict[str, Any]]:
    """用 pdfplumber 解析 PDF，并把表格转成"竖线分隔"的文本。

    参数：
        path: PDF 路径
    返回：
        按页的解析结果；未安装 pdfplumber 时返回空列表（表示"这个解析器不可用"）。

    与 parse_pdf 的分工：
        pdfplumber 提表格能力强，但整体速度比 PyMuPDF 慢、纯文本质量略逊；
        PyMuPDF 快但对表格无能为力。
        parse_document 里的策略是"先用 pdfplumber 拿表格、拿不到内容再退回 PyMuPDF"，
        两种解析器各用所长。

    未装库返回 [] 而不是抛异常：
        它只是"增强解析器"，调用方据此判断要不要回退到基础解析器。
        为可选能力抛异常会强行把"没装 pdfplumber"变成致命错误。
    """
    try:
        import pdfplumber
    except ImportError:
        return []
    pages = []
    with pdfplumber.open(path) as document:
        for number, page in enumerate(document.pages, 1):
            parts = [page.extract_text() or ""]
            for table in page.extract_tables() or []:
                parts.append("\n".join(" | ".join(str(cell or "").strip() for cell in row) for row in table))
            pages.append({"page": number, "text": clean_text("\n".join(parts)), "metadata": {"parser": "pdfplumber"}})
    return pages


def parse_docx(path: Path) -> list[dict[str, Any]]:
    """解析 Word 文档（正文段落 + 表格）。

    参数：
        path: .docx 路径
    返回：
        单页的解析结果（Word 没有稳定页码概念，统一记为第 1 页）。
    异常：
        未安装 python-docx -> RuntimeError（带安装命令）。

    Word 没有页码这件事要接受：
        它的分页由渲染器决定、随字体和纸张变化，无法可靠获取。
        编造一个假页码会让 user 引用到"第 1 页"却查不到东西，不如统一标成 1。
        成本是要靠 chunk_id（分块编号）来定位，而不是页码。
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("DOCX 解析需要 python-docx：pip install python-docx") from exc
    document = Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        paragraphs.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
    return [{"page": 1, "text": clean_text("\n".join(paragraphs)), "metadata": {"parser": "python-docx"}}]


def parse_with_optional_ocr(path: Path) -> list[dict[str, Any]]:
    """扫描件 OCR 解析（仅在显式配置时才启用）。

    参数：
        path: PDF 路径
    返回：
        按页的解析结果；未配置或失败时返回空列表。

    通过环境变量 RAG_OCR_PARSER 选择引擎，两种都用 try/except 包住：
        mineru    —— 复用 backend/pipeline.py 的 MinerU 解析（适合复杂版面）
        paddleocr —— 逐页转成 2 倍分辨率位图再 OCR（适合纯扫描件）

    默认关闭（RAG_OCR_PARSER 未设置就不做任何事）：
        OCR 很慢，而且要给每个 PDF 都跑一遍 OCR 是不可接受的默认行为。
        只有用户明确知道自己的文档是扫描件时，才应该打开它。

    2 倍分辨率（fitz.Matrix(2, 2)）：
        原始 DPI 下小字号的汉字 OCR 识别率很差，放大能明显改善 —— 代价是更慢。
    """
    parser = os.getenv("RAG_OCR_PARSER", "").lower()
    if parser == "mineru":
        try:
            from backend.pipeline import parse_with_mineru
            return parse_with_mineru(path)
        except Exception:
            pass  # MinerU 不可用就继续往下试，不中断整个解析流程
    if parser == "paddleocr":
        try:
            from paddleocr import PaddleOCR
            import fitz
            ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 开启方向分类 + 中文模型
            pages = []
            for number, page in enumerate(fitz.open(path), 1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                results = ocr.ocr(pix.tobytes("png"), cls=True)
                # PaddleOCR 返回嵌套结构 [[[框, (文本, 置信度)], ...], ...]，逐层展开只取文本
                text = "\n".join(item[1][0] for line in (results or []) for item in (line or []))
                pages.append({"page": number, "text": clean_text(text), "metadata": {"parser": "paddleocr"}})
            return pages
        except Exception:
            pass
    return []


def parse_markdown(path: Path) -> list[dict[str, Any]]:
    """解析 Markdown / 纯文本类文件。

    参数：
        path: .md / .markdown 路径
    返回：
        单页的解析结果。

    直接读原文不剥 Markdown 标记：
        标题里的 # 和列表的 - 恰好是分块策略（heading_chunks）用来识别章节的锚点，
        提前剥掉反而会让标题分块失效。
        errors="ignore" 容忍编码不干净的文档，不让一个坏字节毁掉整份文件。
    """
    return [{"page": 1, "text": clean_text(path.read_text(encoding="utf-8", errors="ignore")), "metadata": {"parser": "markdown"}}]


def parse_html(path: Path) -> list[dict[str, Any]]:
    """解析 HTML，只取标签之间的可见文本。

    参数：
        path: .html / .htm 路径
    返回：
        单页的解析结果。

    用标准库 HTMLParser 而不是 BeautifulSoup：
        这里只需要"取出所有文本节点"，标准库足够，不必为它引入第三方依赖。
        注意它不会剔除 <script>/<style> 的内容 —— 对本项目的文档类输入可接受，
        换成抓取网页正文的场景则需要额外处理。
    """
    from html.parser import HTMLParser

    class TextParser(HTMLParser):
        """只收集文本节点的极简解析器。"""

        def __init__(self):
            super().__init__()
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            """HTMLParser 的回调：每遇到一段文本节点就收集起来。"""
            self.parts.append(data)

    parser = TextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return [{"page": 1, "text": clean_text("\n".join(parser.parts)), "metadata": {"parser": "html"}}]


def parse_document(path: str | Path) -> list[dict[str, Any]]:
    """按扩展名分派到具体解析器（统一入口）。

    参数：
        path: 文档路径
    返回：
        按页的解析结果。
    异常：
        扩展名不在支持列表内 -> ValueError。

    PDF 的三级降级顺序（这里体现得最清楚）：
        1. pdfplumber  —— 能提表格，优先
        2. PyMuPDF     —— 拿不到内容时退回，纯文本更快更稳
        3. OCR         —— 前两者都提不出文字（扫描件）时，看有没有配 OCR
        每一级都判断"提出来的是不是真有内容"，而不是"函数有没有报错"：
        解析器不报错但返回空文本，是最需要被识别出来的失败模式。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = parse_pdf_tables(path)
        if pages and any(item["text"] for item in pages):
            return pages
        pages = parse_pdf(path)
        if any(item["text"] for item in pages):
            return pages
        return parse_with_optional_ocr(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    if suffix in {".txt", ".text", ".csv"}:
        return [{"page": 1, "text": clean_text(path.read_text(encoding="utf-8", errors="ignore")), "metadata": {"parser": "text"}}]
    raise ValueError(f"不支持的文档格式：{path.suffix}")


def parse_directory(path: str | Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    """递归解析一个目录下所有受支持的文档。

    参数：
        path: 目录路径
    返回：
        [(文件路径, 该文件的按页解析结果), ...]，按文件路径排序。

    为什么递归（rglob）：
        正式数据按角色分目录存放（如 data/raw/lawyer/），
        用户也常把附件放进子目录。只扫一层会静默漏掉这些文件。

    排序是为了结果可复现：
        不加 sorted 的话，入库顺序由文件系统决定，同一批文件两次构建可能得到
        不同的 chunk_id 分配，导致对比评测时无法归因。
    """
    root = Path(path)
    files = [item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in {".pdf", ".docx", ".md", ".markdown", ".html", ".htm", ".txt", ".text", ".csv"}]
    return [(item, parse_document(item)) for item in sorted(files)]
