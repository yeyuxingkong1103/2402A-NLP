"""使用 PyMuPDF、PP-OCRv6 和 PaddleOCR-VL 解析 PDF。"""

import re
import tempfile
from pathlib import Path

import fitz

from app.config import get_settings
from app.rag_ocr_models import _ocr_file, _safe_vl
from app.rag_text_processing import clean_pdf_text, remove_repeated_watermarks

settings = get_settings()


def _render_region(page: fitz.Page, path: Path, clip: fitz.Rect | None = None) -> None:
    """把 PDF 整页或指定矩形区域渲染成 2 倍分辨率图片。"""
    # Matrix(2, 2) 提高 OCR 清晰度；alpha=False 输出普通 RGB 背景。
    page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False).save(path)


def _find_tables(page: fitz.Page) -> list:
    """调用 PyMuPDF 检测当前页表格；不支持或检测失败时返回空列表。"""
    try:
        # .tables 才是真正的表格对象列表。
        return page.find_tables().tables
    except (AttributeError, RuntimeError):
        # 某些特殊 PDF 页面可能不支持表格检测，不能影响正文提取。
        return []


def _table_text(page: fitz.Page, folder: Path, tables: list, use_visual: bool) -> str:
    """把当前页所有表格转换成使用 ``|`` 分列的文本。"""
    # rows 收集这一页所有表格的所有有效行。
    rows: list[str] = []
    # index 用于给临时表格截图命名。
    for index, table in enumerate(tables):
        # table_rows 只保存当前一个表格的行。
        table_rows: list[str] = []
        # table.extract() 返回二维数组：外层是行，内层是单元格。
        for row in table.extract():
            # None 单元格转空串，并清理单元格中的多余空白。
            values = [clean_pdf_text(value or "") for value in row]
            # 整行全空时忽略。
            if any(values):
                # 用竖线保留列边界，便于 BM25 和大模型理解表格关系。
                table_rows.append(" | ".join(values))
        # 合并到页面表格结果。
        rows.extend(table_rows)
        # PyMuPDF 已提取出有效单元格时，无需再让大型视觉模型重复识别。
        # 只有原生表格结果过少时才用 PaddleOCR-VL 补充，CPU 入库会快很多。
        if use_visual and settings.paddleocr_vl_enabled and not table_rows:
            # 原生提取失败时才截取表格区域，避免昂贵的重复视觉识别。
            table_path = folder / f"table_{page.number}_{index}.png"
            # table.bbox 是表格在页面中的矩形坐标。
            _render_region(page, table_path, fitz.Rect(table.bbox))
            # 让视觉模型执行 Table Recognition。
            vl_text = _safe_vl(table_path, "table")
            if vl_text:
                # 加标签方便后续判断内容来源。
                rows.append(f"[视觉表格识别]\n{vl_text}")
    # 表格的不同行使用换行分隔。
    return "\n".join(rows)


def _looks_like_chart(text: str) -> bool:
    """根据 OCR 初识别文字，快速判断图片是否更像统计图表。

    这里不调用大模型，只做成本很低的启发式判断：出现常见图表词，或者图片中
    至少包含两个百分比数据，都说明它更可能需要执行“图表理解”任务。
    """
    # 这些词经常出现在柱状图、折线图、饼图的标题、图例或说明文字中。
    keywords = ("图表", "柱状", "折线", "饼图", "趋势", "占比", "百分比")
    # 百分比正则同时支持整数（20%）和小数（20.5%）；两个以上比例通常表示数据图。
    return any(word in text for word in keywords) or len(re.findall(r"\d+(?:\.\d+)?%", text)) >= 2


def _visual_task(path: Path, basic_text: str) -> str:
    """为一张图片选择 PaddleOCR-VL 任务：图表、表格或普通 OCR。"""
    # 优先使用 OCR 文字判断图表，因为关键词和比例比纯图像线条更能表达语义。
    if _looks_like_chart(basic_text):
        return "chart"
    try:
        # OpenCV 只在这里按需导入；未安装时仍可退回普通 OCR，不阻断 PDF 入库。
        import cv2

        # 转成灰度图，后续只关心线条结构，不需要颜色信息。
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # 反向二值化：浅色背景变 0，深色文字和表格线变 255，便于统计前景像素。
            binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]
            # 30×1 的横向结构元素保留较长横线，尽量滤掉单个汉字和噪点。
            horizontal = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            )
            # 1×30 的纵向结构元素执行同样操作，用于提取表格竖线。
            vertical = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            )
            # 横线像素与竖线像素越多，图片是有框表格的概率越高。
            line_pixels = cv2.countNonZero(horizontal) + cv2.countNonZero(vertical)
            # 线条像素超过整张图的 2.5% 时，将任务切换为表格结构识别。
            if line_pixels > image.size * 0.025:
                return "table"
    # 图像解码失败或 OpenCV 不存在时进行安全降级，不能让可选判断影响主流程。
    except (ImportError, OSError, ValueError):
        pass
    # 既不像图表也没有明显表格网格时，只提取图片文字即可。
    return "ocr"


def _image_text(
    document: fitz.Document, page: fitz.Page, folder: Path, use_visual: bool
) -> str:
    """提取一页 PDF 内嵌图片或扫描整页中的文字与视觉语义。

    ``document`` 用于取出 PDF 内嵌图片，``page`` 是当前页面，``folder`` 是临时
    图片目录；返回值会和本页原生文字、表格文字一起进入后续清洗流程。
    """
    # 同一页可能有多张图片，每种识别结果先收集到列表，最后统一拼接。
    results: list[str] = []
    # 快速索引阶段 use_visual=False，不逐张运行大视觉模型，以缩短首次可检索时间。
    if use_visual:
        # 限制每页最多处理的图片数，防止图片型 PDF 在 CPU 上耗时过长或内存过高。
        images = page.get_images(full=True)[:settings.paddleocr_vl_max_images_per_page]
        for index, image in enumerate(images):
            # image[0] 是 PyMuPDF 的 xref；extract_image 返回字节、扩展名和尺寸。
            extracted = document.extract_image(image[0])
            # 跳过图标、项目符号等小图片，它们通常没有知识价值，识别还会增加噪声。
            if extracted.get("width", 0) < 128 or extracted.get("height", 0) < 64:
                continue
            # 图片只写进 TemporaryDirectory，函数结束后系统会自动删除，不污染 uploads。
            image_path = folder / f"image_{page.number}_{index}.{extracted.get('ext', 'png')}"
            image_path.write_bytes(extracted["image"])
            # 普通 OCR 负责可靠地识别可见文字，速度通常比视觉语言模型快。
            basic_text = _ocr_file(image_path) if settings.ocr_enabled else ""
            # 根据初识别文字和图像线条，决定让 PaddleOCR-VL 理解图表、表格还是文字。
            visual_text = _safe_vl(image_path, _visual_task(image_path, basic_text))
            # 两路结果都保留：OCR 提供原字词，VL 补充结构关系和图表含义。
            results.extend(value for value in (basic_text, visual_text) if value)
    # 页面没有原生文本时，通常是扫描件；此时不能只检查内嵌图片，需要识别整页。
    if not page.get_text("text").strip():
        page_path = folder / f"page_{page.number}.png"
        # 将 PDF 页按统一分辨率渲染成 PNG，供 OCR 模型读取。
        _render_region(page, page_path)
        basic_text = _ocr_file(page_path) if settings.ocr_enabled else ""
        # 视觉增强阶段再补充 VL 识别；快速阶段仅使用基础 OCR。
        visual_text = _safe_vl(page_path, "ocr") if use_visual else ""
        results.extend(value for value in (basic_text, visual_text) if value)
    # 每条识别结果单独成行，便于后续按段落和句子切块。
    return "\n".join(result for result in results if result)


def extract_pdf_text(path: Path, use_visual: bool = True) -> str:
    """完成第 1 步：汇总原生文本、表格、OCR 和可选视觉理解结果。

    - 文字型 PDF：PyMuPDF 的 ``page.get_text`` 直接抽取。
    - 原生表格：PyMuPDF ``find_tables`` 保留行列内容。
    - 扫描页/图片：PP-OCRv6 检测并识别文字。
    - 图表/复杂表格：PaddleOCR-VL 做视觉语义补充。
    - 文字水印：统计跨页重复短文本，在切块前统一删除。
    - use_visual=False：首次快速索引；True：后台视觉增强。
    """
    # 先让 PyMuPDF 验证并打开文件；将底层异常改成用户更容易理解的中文错误。
    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError("PDF 文件已损坏或无法读取") from error
    # 页码和正文分别保存：去水印可能删空某页，但最终仍要保留正确的原 PDF 页码。
    page_numbers: list[int] = []
    page_texts: list[str] = []
    # OCR/VL 所需中间图片放入临时目录，离开 with 块后自动清理。
    with document, tempfile.TemporaryDirectory(prefix="rag_ocr_") as temp_dir:
        folder = Path(temp_dir)
        # start=1 让展示给用户的页码从 1 开始，而 PyMuPDF 内部页号从 0 开始。
        for number, page in enumerate(document, start=1):
            # 先定位本页原生表格，后面抽取时可以保留行列数据。
            tables = _find_tables(page)
            # 一页内容由“原生文字 + 表格 + 图片/扫描 OCR”三部分组成。
            parts = [
                clean_pdf_text(page.get_text("text")),
                _table_text(page, folder, tables, use_visual),
            ]
            # 只有启用至少一种图像识别能力时，才扫描页内图片，避免无意义开销。
            if settings.ocr_enabled or settings.paddleocr_vl_enabled:
                parts.append(_image_text(document, page, folder, use_visual))
            # 合并后再次清洗，统一空白字符并减少重复空行。
            content = clean_pdf_text("\n".join(part for part in parts if part))
            if content:
                page_numbers.append(number)
                page_texts.append(content)
    # 整份文件没有任何可用内容时明确失败，不能建立空索引。
    if not page_texts:
        raise ValueError("未提取到文字、表格或 OCR 内容")
    # 必须等所有页解析完成后再统计重复短行，因为水印的依据就是“跨页反复出现”。
    page_texts, _ = remove_repeated_watermarks(page_texts)
    # 给每页加页码标签，使检索到的块仍能说明内容来自原文哪一页。
    pages = [
        f"[第{number}页]\n{content}"
        for number, content in zip(page_numbers, page_texts, strict=True)
        if content
    ]
    # 极端情况下所有内容都被判定为重复水印，阻止无正文文档进入向量库。
    if not pages:
        raise ValueError("去除重复水印后没有可用正文")
    # 页与页之间留两个换行，下一阶段会优先把它识别为段落边界。
    return "\n\n".join(pages)

