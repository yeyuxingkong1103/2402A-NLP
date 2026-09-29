# -*- coding: utf-8 -*-
"""
文档解析模块：支持 txt/md/docx/wps/doc/pdf 六种格式
- docx：python-docx 逐段落提取
- pdf：PyMuPDF (fitz) 逐页提取；pdfplumber 提取表格
- wps/doc：win32com 调本机 WPS COM 另存为 txt
- 扫描件：PaddleOCR 识别图片文字
- 图片提取（注释）：PyMuPDF 抽取 PDF 内嵌图片
- 图表描述（注释）：调用视觉大模型（VLM）把柱状图/饼图转成文字
- 表格整合（注释）：PyMuPDF extract_tables="markdown" 一步到位
- Excel/CSV（注释）：openpyxl/pandas 读取电子表格
- PPT（注释）：python-pptx 提取幻灯片文字
"""

import io  # 字节流处理
import os  # 文件路径操作
from typing import Tuple  # 元组类型标注

from fastapi import HTTPException  # HTTP 异常
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger


def extract_from_docx(file_bytes: bytes) -> str:
    """用 python-docx 读取 .docx：逐段落取文本，空段落当作换行"""
    import docx  # 延迟导入：只在真正需要时加载
    doc = docx.Document(io.BytesIO(file_bytes))  # 从内存字节加载 Word 文档
    paragraphs = []  # 存放各段落
    for para in doc.paragraphs:  # 遍历所有段落
        text = para.text.strip()  # 去首尾空白
        if text:  # 非空段落
            paragraphs.append(text)  # 收集
        else:  # 空段落 → 换行（保留原文档的结构间隔）
            paragraphs.append("")
    return "\n".join(paragraphs)  # 换行拼回完整文本


def extract_from_pdf(file_bytes: bytes) -> str:
    """用 PyMuPDF (fitz) 逐页提取文本，页间加空行"""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # 从字节加载 PDF
    pages = []  # 存放每页文本
    for page in doc:  # 逐页
        pages.append(page.get_text("text"))  # 提取文本
    doc.close()  # 关闭
    text = "\n\n".join(pages)  # 页间空行
    # 如果提取出来是空的，可能是扫描件 → 提示用 OCR
    if not text.strip():
        logger.warning("PDF 文本提取为空，可能是扫描件，需要 OCR 处理")
    return text


def extract_tables_from_pdf(file_bytes: bytes) -> list:
    """用 pdfplumber 提取 PDF 中的表格，返回 Markdown 格式表格文本列表"""
    import pdfplumber  # 延迟导入
    tables_text = []  # 存放各表格的 Markdown 文本
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:  # 打开 PDF
        for page in pdf.pages:  # 逐页
            tables = page.extract_tables()  # 提取本页所有表格
            for table in tables:  # 逐个表格
                if not table:  # 空表跳过
                    continue
                # 把表格转成 Markdown table 格式
                md_lines = []  # 存放 Markdown 行
                for i, row in enumerate(table):  # 逐行
                    cells = [str(cell or "").strip() for cell in row]  # 清理每个单元格
                    md_lines.append("| " + " | ".join(cells) + " |")  # Markdown 行
                    if i == 0:  # 第一行后加分隔行
                        md_lines.append("|" + "|".join(["---"] * len(row)) + "|")
                tables_text.append("\n".join(md_lines))  # 存入列表
    logger.info(f"PDF 表格提取：共 {len(tables_text)} 个表格")  # 日志
    return tables_text  # 返回 Markdown 表格文本列表


def extract_from_wps_doc(file_bytes: bytes, suffix: str) -> str:
    """用 win32com 调本机 WPS 打开 .wps/.doc 另存为 txt 再读文本"""
    import pythoncom  # COM 初始化
    import tempfile  # 临时文件
    pythoncom.CoInitialize()  # 当前线程初始化 COM
    try:
        import win32com.client  # 延迟导入
        tmp_dir = tempfile.mkdtemp()  # 临时目录
        src_path = os.path.join(tmp_dir, f"input{suffix}")  # 临时源文件
        out_path = os.path.join(tmp_dir, "output.txt")  # 临时输出
        with open(src_path, "wb") as f:  # 写入临时文件
            f.write(file_bytes)
        try:  # 先试 WPS
            wps = win32com.client.Dispatch("kwps.Application")  # WPS 文字组件
        except Exception:  # 没装 WPS
            try:  # 再试 Word
                wps = win32com.client.Dispatch("Word.Application")
            except Exception:
                raise HTTPException(500, "本机未安装 WPS 或 Word，无法解析 .wps/.doc 文件")
        wps.Visible = False  # 后台运行
        doc = wps.Documents.Open(src_path)  # 打开
        doc.SaveAs(out_path, 7)  # 另存为 Unicode txt
        doc.Close(False)  # 关闭
        wps.Quit()  # 退出
        with open(out_path, "r", encoding="utf-8") as f:  # 读取 txt
            return f.read()  # 返回文本
    finally:
        pythoncom.CoUninitialize()  # 释放 COM


def extract_from_scanned_pdf(file_bytes: bytes) -> str:
    """用 PaddleOCR 对扫描件/图片型 PDF 做 OCR 识别"""
    try:
        import fitz  # 先用 PyMuPDF 把 PDF 渲染成图片
        from paddleocr import PaddleOCR  # OCR 引擎
    except ImportError:
        raise HTTPException(500, "PaddleOCR 未安装，无法处理扫描件，请 pip install paddleocr")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)  # 初始化 OCR（中文+角度分类）
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # 打开 PDF
    pages_text = []  # 每页 OCR 结果
    for page in doc:  # 逐页
        pix = page.get_pixmap(dpi=300)  # 渲染为 300 DPI 图片（DPI 越高 OCR 越准）
        img_bytes = pix.tobytes("png")  # 转 PNG 字节
        result = ocr.ocr(img_bytes, cls=True)  # OCR 识别
        lines = []  # 本页文本行
        if result and result[0]:  # 有结果
            for line in result[0]:  # 逐行
                lines.append(line[1][0])  # line[1] = (text, confidence)
        pages_text.append("\n".join(lines))  # 本页文本
    doc.close()  # 关闭
    return "\n\n".join(pages_text)  # 页间空行


# ==================== 以下为扩展解析器（当前项目未启用，按需取消注释） ====================

# def extract_images_from_pdf(file_bytes: bytes, output_dir: str = "./extracted_images") -> list:
#     """
#     提取 PDF 内嵌的所有图片（PyMuPDF）
#
#     用途：PDF 里有插图、示意图、照片时，先把图片文件抽出来，
#           再交给 OCR 或视觉大模型（VLM）理解内容。
#     依赖：pip install pymupdf
#     返回：保存的图片路径列表，如 ["extracted_images/p1_img0.png", ...]
#     """
#     import fitz  # PyMuPDF
#     os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
#     doc = fitz.open(stream=file_bytes, filetype="pdf")  # 打开 PDF
#     img_paths = []  # 存放图片路径
#     for page_num, page in enumerate(doc):  # 逐页
#         for img_index, img in enumerate(page.get_images(full=True)):  # 逐张内嵌图
#             xref = img[0]  # 图片在 PDF 内部的引用 ID
#             base_image = doc.extract_image(xref)  # 提取为字节
#             ext = base_image["ext"]  # 图片格式（png/jpeg/...）
#             img_bytes = base_image["image"]  # 图片字节
#             filename = f"p{page_num + 1}_img{img_index}.{ext}"  # 文件名
#             filepath = os.path.join(output_dir, filename)  # 完整路径
#             with open(filepath, "wb") as f:  # 写入文件
#                 f.write(img_bytes)
#             img_paths.append(filepath)  # 记录路径
#     doc.close()
#     logger.info(f"提取图片：共 {len(img_paths)} 张，保存到 {output_dir}")
#     return img_paths  # 返回路径列表，后续可喂给 VLM


# def describe_image_with_vlm(image_bytes: bytes, vlm_api_url: str, vlm_api_key: str,
#                              model: str = "gpt-4o", prompt: str = "请描述这张图片的内容") -> str:
#     """
#     调用视觉大模型（VLM）描述图片内容
#
#     用途：柱状图/饼图/折线图等统计图，或者照片、流程图，
#           纯文本提取器读不出内容，必须靠"看图"的模型转成文字。
#     支持的模型：GPT-4o / Qwen-VL / GLM-4V / Claude 3 等，走 OpenAI 兼容接口
#     依赖：pip install openai
#     返回：VLM 输出的文字描述（可直接当文本块入库）
#     """
#     import base64  # base64 编码图片字节
#     from openai import OpenAI  # OpenAI 兼容客户端
#     client = OpenAI(api_key=vlm_api_key, base_url=vlm_api_url)  # 初始化客户端
#     b64 = base64.b64encode(image_bytes).decode("utf-8")  # 图片字节转 base64
#     response = client.chat.completions.create(  # 调用 VLM
#         model=model,
#         messages=[{
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},  # 文字指令
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},  # 图片
#             ],
#         }],
#         max_tokens=1024,  # 描述不要太长
#     )
#     description = response.choices[0].message.content  # 提取文字
#     logger.info(f"VLM 图片描述完成：{len(description)} 字")
#     return description  # 返回文字描述，可当作普通文本块入库


# def extract_image_with_vlm(file_bytes: bytes, filename: str) -> str:
#     """
#     纯图片文件（png/jpg/webp）→ VLM 描述 → 文字
#
#     用途：知识库里有纯图片文件（不是 PDF 里的内嵌图），直接喂给 VLM 转文字
#     依赖：pip install openai（走 OpenAI 兼容接口）
#     """
#     # VLM 接口配置（按实际环境修改）
#     _VLM_API_URL = "https://api.openai.com/v1"  # 或本地 Qwen-VL 地址
#     _VLM_API_KEY = os.getenv("vlm_api_key", "")
#     _VLM_MODEL = "gpt-4o"  # 或 "qwen-vl-plus"
#     return describe_image_with_vlm(  # 调用上面的通用函数
#         image_bytes=file_bytes,
#         vlm_api_url=_VLM_API_URL,
#         vlm_api_key=_VLM_API_KEY,
#         model=_VLM_MODEL,
#         prompt="请详细描述这张图片的内容，包括图表中的数据、文字、标题等关键信息",  # 通用提示
#     )


# def extract_pdf_with_tables(file_bytes: bytes) -> str:
#     """
#     PDF 文本+表格一次性提取（PyMuPDFLoader 的 extract_tables 模式）
#
#     用途：PDF 里有有线表格（赔偿标准表、量刑表等），不想分开调 pdfplumber，
#           用 PyMuPDFLoader 一步到位，表格直接转 Markdown
#     依赖：pip install langchain-community pymupdf
#     导入：from langchain_community.document_loaders import PyMuPDFLoader
#     返回：纯文本 + Markdown 表格，混合在一起
#     """
#     import tempfile  # 临时文件
#     from langchain_community.document_loaders import PyMuPDFLoader  # LangChain 封装的加载器
#     # PyMuPDFLoader 需要文件路径，不接受字节流，先写临时文件
#     tmp_path = os.path.join(tempfile.mkdtemp(), "input.pdf")  # 临时 PDF
#     with open(tmp_path, "wb") as f:
#         f.write(file_bytes)
#     loader = PyMuPDFLoader(  # 初始化加载器
#         tmp_path,
#         mode="page",  # 按页返回 Document
#         extract_tables="markdown",  # 表格转 Markdown 格式（可选 csv/html）
#         extract_images=False,  # 不提取图片（这里只关心文字+表格）
#     )
#     docs = loader.load()  # 加载
#     text = "\n\n".join(d.page_content for d in docs)  # 拼接所有页
#     logger.info(f"PDF 文本+表格提取完成：{len(docs)} 页")
#     return text


# def extract_from_excel(file_bytes: bytes) -> str:
#     """
#     读取 Excel 文件（.xlsx/.xls），逐 Sheet 转 Markdown 表格
#
#     用途：知识库数据源包含 Excel 表格（如赔偿标准表、量刑对照表）
#     依赖：pip install openpyxl
#     返回：所有 Sheet 的内容拼成 Markdown 表格文本
#     """
#     import openpyxl  # Excel 读取库
#     wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)  # data_only=True 取公式计算后的值
#     sheets_text = []  # 存放各 Sheet 的文本
#     for ws in wb.worksheets:  # 逐 Sheet
#         md_lines = [f"## {ws.title}"]  # Sheet 名做标题
#         for i, row in enumerate(ws.iter_rows(values_only=True)):  # 逐行
#             cells = [str(c or "").strip() for c in row]  # 清理单元格
#             if not any(cells):  # 全空行跳过
#                 continue
#             md_lines.append("| " + " | ".join(cells) + " |")  # Markdown 行
#             if i == 0:  # 第一行后加分隔行
#                 md_lines.append("|" + "|".join(["---"] * len(row)) + "|")
#         sheets_text.append("\n".join(md_lines))  # 拼接
#     logger.info(f"Excel 解析完成：{len(wb.worksheets)} 个 Sheet")
#     return "\n\n".join(sheets_text)  # Sheet 间空行


# def extract_from_pptx(file_bytes: bytes) -> str:
#     """
#     读取 PowerPoint（.pptx），逐幻灯片提取文字
#
#     用途：知识库数据源包含 PPT 课件、汇报材料
#     依赖：pip install python-pptx
#     返回：每页幻灯片的文字，页间空行
#     """
#     from pptx import Presentation  # PPT 读取库
#     prs = Presentation(io.BytesIO(file_bytes))  # 从字节加载
#     slides_text = []  # 每页文本
#     for i, slide in enumerate(prs.slides, start=1):  # 逐页
#         lines = [f"## 第 {i} 页"]  # 页码标题
#         for shape in slide.shapes:  # 逐元素（文本框/表格/图片）
#             if shape.has_text_frame:  # 文本框
#                 text = shape.text_frame.text.strip()  # 提取文字
#                 if text:
#                     lines.append(text)
#             if shape.has_table:  # 表格 → 转 Markdown
#                 table = shape.table
#                 for r, row in enumerate(table.rows):  # 逐行
#                     cells = [cell.text.strip() for cell in row.cells]  # 清理
#                     lines.append("| " + " | ".join(cells) + " |")  # Markdown 行
#                     if r == 0:  # 第一行后加分隔行
#                         lines.append("|" + "|".join(["---"] * len(row.cells)) + "|")
#         slides_text.append("\n".join(lines))  # 拼接本页
#     logger.info(f"PPTX 解析完成：{len(prs.slides)} 页")
#     return "\n\n".join(slides_text)  # 页间空行


def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, bool]:
    """
    根据后缀自动选择解析器

    Returns:
        (纯文本, 是否按 Markdown 标题结构切分)
    """
    suffix = filename.lower()  # 统一小写
    if suffix.endswith(".md"):  # Markdown
        return file_bytes.decode("utf-8", errors="ignore"), True
    elif suffix.endswith(".txt"):  # 纯文本
        return file_bytes.decode("utf-8", errors="ignore"), False
    elif suffix.endswith(".docx"):  # Word/WPS docx
        return extract_from_docx(file_bytes), False
    elif suffix.endswith(".pdf"):  # PDF
        text = extract_from_pdf(file_bytes)  # 先试文本提取
        if not text.strip():  # 空的 → 可能是扫描件
            logger.info("检测到扫描件，启动 OCR ...")
            text = extract_from_scanned_pdf(file_bytes)  # OCR
        return text, False
    elif suffix.endswith(".wps") or suffix.endswith(".doc"):  # WPS/旧版 Word
        return extract_from_wps_doc(file_bytes, os.path.splitext(filename)[1]), False
    # elif suffix.endswith((".xlsx", ".xls")):  # Excel 电子表格
    #     return extract_from_excel(file_bytes), False
    # elif suffix.endswith(".csv"):  # CSV
    #     return file_bytes.decode("utf-8", errors="ignore"), False
    # elif suffix.endswith(".pptx"):  # PowerPoint
    #     return extract_from_pptx(file_bytes), False
    # elif suffix.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):  # 纯图片
    #     return extract_image_with_vlm(file_bytes, filename), False
    else:
        raise HTTPException(400, f"暂不支持 {filename}，请上传 txt/md/docx/wps/doc/pdf")
