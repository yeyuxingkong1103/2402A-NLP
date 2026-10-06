# -*- coding: utf-8 -*-
"""PDF 处理：文本提取 + 去水印。"""
import fitz
# 解析：PyMuPDF（PDF 解析/编辑）


def extract_text(pdf_bytes: bytes) -> str:
    """提取 PDF 全部页面文本，页间用空行分隔；非法 PDF 抛 ValueError。"""
    try:
        # 解析：尝试打开 PDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # 解析：从字节流打开（filetype 指定格式）
    except Exception as exc:
        # 解析：打开失败（非法文件）
        raise ValueError("无法解析 PDF 文件") from exc
        # 解析：转成友好错误（上层返回 400）
    try:
        # 解析：提取文本
        return "\n\n".join(page.get_text() for page in doc)
        # 解析：逐页提取文本，页间空行分隔
    finally:
        # 解析：无论如何执行
        doc.close()
        # 解析：释放文档资源


DEFAULT_KEYWORDS = ("水印", "机密", "内部资料", "仅供", "请勿", "confidential", "draft", "sample")
# 解析：默认水印关键词表（识别含这些词的文字为水印候选）


def extract_tables(pdf_bytes: bytes) -> list[str]:
    """提取 PDF 中的表格，转成 Markdown 表格文本；无表格返回空列表。"""
    import io
    # 解析：BytesIO（pdfplumber 需要文件对象）
    import pdfplumber
    # 解析：pdfplumber 表格提取库（延迟导入）

    tables = []
    # 解析：结果表格列表
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # 解析：打开 PDF（pdfplumber 不接受裸 bytes，需包 BytesIO）
        for page in pdf.pages:
            # 解析：逐页
            for table in page.extract_tables():
                # 解析：逐表格（依据线条结构识别）
                rows = [[(cell or "").strip() for cell in row] for row in table]
                # 解析：清洗单元格（None 转空串、去空白）
                rows = [r for r in rows if any(r)]  # 去掉全空行
                # 解析：过滤整行空的
                if not rows:
                    # 解析：表格全空
                    continue
                    # 解析：跳过
                header = "| " + " | ".join(rows[0]) + " |"
                # 解析：Markdown 表头行
                divider = "|" + "---|" * len(rows[0])
                # 解析：Markdown 分隔行（每列一个 ---）
                body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
                # 解析：数据行
                tables.append(f"{header}\n{divider}\n{body}" if len(rows) > 1 else header)
                # 解析：拼成完整 Markdown 表格（只有表头时只返回表头行）
    return tables
    # 解析：返回全部表格文本


def remove_watermark(
    # 解析：去水印
    pdf_bytes: bytes,
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS,
    # 解析：文字水印关键词表（可自定义）
    min_pages: int = 2,
    # 解析：文字水印最小出现页数（低于则不认定）
    image_min_pages: int = 3,
    # 解析：图片水印最小出现页数
) -> bytes:
    """返回去水印后的 PDF 字节；无水印时内容保持不变。"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # 解析：打开 PDF
    try:
        # 解析：处理主体
        # ---- 第一遍：收集候选文字水印（跨页统计） ----
        page_spans = []  # [(page_index, bbox, normalized_text)]
        # 解析：候选水印文字列表（页号、位置框、文本）
        for pno, page in enumerate(doc):
            # 解析：逐页
            for block in page.get_text("dict")["blocks"]:
                # 解析：逐文本块
                if block.get("type") != 0:
                    # 解析：非文字块（图片等）
                    continue
                    # 解析：跳过
                for line in block.get("lines", []):
                    # 解析：逐行
                    direction = tuple(line.get("dir", (1, 0)))
                    # 解析：文字方向向量
                    rotated = direction not in ((1, 0), (0, 1))
                    # 解析：非水平/垂直即斜置（斜置是水印典型特征）
                    for span in line.get("spans", []):
                        # 解析：逐文字段
                        text = span.get("text", "").strip()
                        # 解析：文字内容
                        if not text:
                            # 解析：空段
                            continue
                            # 解析：跳过
                        color = span.get("color", 0)
                        # 解析：颜色（整数 RGB）
                        light_gray = (
                            # 解析：浅灰判定
                            ((color >> 16) & 255) > 180
                            # 解析：R 分量 > 180
                            and ((color >> 8) & 255) > 180
                            # 解析：G 分量 > 180
                            and (color & 255) > 180
                            # 解析：B 分量 > 180（三通道都浅 = 灰色水印）
                        )
                        has_keyword = any(k in text for k in keywords)
                        # 解析：含水印关键词
                        if light_gray or rotated or has_keyword:
                            # 解析：三信号任一命中即候选
                            page_spans.append((pno, fitz.Rect(span["bbox"]), text))
                            # 解析：记录候选（页号、位置、文本）

        # 同一文本出现在 ≥ min_pages 页才认定为水印
        page_count: dict[str, int] = {}
        # 解析：文本出现页数统计
        for _, _, text in page_spans:
            # 解析：逐候选
            page_count[text] = page_count.get(text, 0) + 1
            # 解析：计数 +1
        watermark_spans = [
            # 解析：过滤出真水印（多页重复约束——防误伤正文单个词）
            (pno, bbox) for pno, bbox, text in page_spans if page_count[text] >= min_pages
            # 解析：出现页数达标才算水印
        ]

        # ---- 图片水印：同 xref 出现在 ≥ image_min_pages 页 ----
        image_pages: dict[int, list[int]] = {}  # xref -> [页号...]
        # 解析：图片引用统计
        for pno, page in enumerate(doc):
            # 解析：逐页
            for info in page.get_image_info(xrefs=True):
                # 解析：逐图片
                image_pages.setdefault(info["xref"], []).append(pno)
                # 解析：记录该图片出现的页
        watermark_images = [
            # 解析：过滤出真图片水印
            (xref, pages)
            for xref, pages in image_pages.items()
            # 解析：逐图片
            if len(pages) >= image_min_pages
            # 解析：出现 ≥3 页判定为 Logo 水印（1 页的是真实插图不误删）
        ]

        # ---- 涂抹 ----
        for pno, bbox in watermark_spans:
            # 解析：逐文字水印
            doc[pno].add_redact_annot(bbox)
            # 解析：标记涂抹区域
        for _, pages in watermark_images:
            # 解析：逐图片水印
            for pno in pages:
                # 解析：逐出现页
                for info in doc[pno].get_image_info(xrefs=True):
                    # 解析：该页图片
                    if info["xref"] in [x for x, _ in watermark_images]:
                        # 解析：是水印图片
                        doc[pno].add_redact_annot(fitz.Rect(info["bbox"]))
                        # 解析：标记涂抹该图区域
        for page in doc:
            # 解析：逐页
            page.apply_redactions()
            # 解析：执行涂抹（覆盖为白）

        return doc.tobytes()
        # 解析：返回去水印后的 PDF 字节
    finally:
        # 解析：无论如何执行
        doc.close()
        # 解析：释放文档
