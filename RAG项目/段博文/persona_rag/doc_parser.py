# -*- coding: utf-8 -*-
"""
文档解析模块：支持 txt/md/csv/docx/wps/doc/pdf/xlsx/pptx 等格式
- docx：python-docx 逐段落提取
- pdf：PyMuPDF (fitz) 逐页提取；pdfplumber 提取表格
- wps/doc：win32com 调本机 WPS COM 另存为 txt
- 扫描件：PaddleOCR 识别图片文字
- 图片提取：PyMuPDF 抽取 PDF 内嵌图片（工具函数，供上层按需调用）
- 表格整合：PyMuPDFLoader extract_tables="markdown" 一步到位（工具函数）
- Excel/CSV：openpyxl 读取电子表格；CSV 直接按 UTF-8 解码
- PPT：python-pptx 提取幻灯片文字
- VLM 图片理解（未启用）：视觉大模型描述图片/图表，需要 vlm_api_key 与可用的视觉模型服务

在系统中的位置：
    上游是知识库上传接口（FastAPI 路由）——它把上传文件的字节流和文件名交给本模块；
    下游是 text_splitter——本模块只做「字节流 -> 纯文本 + 是否 Markdown」的转换，不负责切分。
    本模块调用的外部库：python-docx / PyMuPDF(fitz) / pdfplumber / PaddleOCR / win32com。

关键设计取舍：
    1. 所有重依赖都写在函数内部惰性 import，因为服务启动时根本不需要解析文档，
       这样既能加快启动，也能让没装这些库的机器先跑起来（不用时不影响）；
    2. 解析器一律接收 bytes 而不是文件路径，这样上传流可以直接在内存里处理、不落临时文件；
       唯一的例外是 win32com（COM 组件只认磁盘路径），所以它先写临时文件再读回。
"""

import io  # 字节流处理：把 bytes 包装成文件对象，给只认文件的库用
import os  # 文件路径操作
from typing import Tuple  # 元组类型标注

from fastapi import HTTPException  # HTTP 异常
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger


def extract_from_docx(file_bytes: bytes) -> str:
    """
    用 python-docx 读取 .docx：逐段落取文本，空段落当作换行

    Args:
        file_bytes: 上传文件的原始字节流（由 extract_text 透传进来）
    Returns:
        纯文本字符串；段落之间用 "\n" 连接，空段落落到结果里就是一个空行，
        用来保持原文档的版式间隔（切分时"空行"是段落策略最重要的分隔符）
    说明:
        只读正文段落，不包含页眉页脚、批注、文本框和图片
    坑:
        .docx 本质是一个 zip 包，python-docx 需要一个"文件对象"而不是字节，
        所以必须用 io.BytesIO 包一层；好处是全程在内存里完成，不用写临时文件
    """
    import docx  # 延迟导入：只在真正需要时加载（python-docx 只在解析 docx 时才用得上）
    doc = docx.Document(io.BytesIO(file_bytes))  # 从内存字节加载 Word 文档
    paragraphs = []  # 存放各段落（最后用换行拼成一整段文本）
    # 1. 逐段落收集文本，把空段落也保留下来当"换行符"用
    for para in doc.paragraphs:  # 遍历所有段落（doc.paragraphs 只含正文段落，不含表格里的单元格）
        text = para.text.strip()  # 去首尾空白（Word 里常带不可见空格/全角空格）
        if text:  # 非空段落
            paragraphs.append(text)  # 收集
        else:  # 空段落 → 换行（保留原文档的结构间隔）
            paragraphs.append("")
    # 2. 用换行把段落拼回完整文本，交回给调用方
    return "\n".join(paragraphs)  # 换行拼回完整文本


def extract_from_pdf(file_bytes: bytes) -> str:
    """
    用 PyMuPDF (fitz) 逐页提取文本，页间加空行

    Args:
        file_bytes: PDF 文件的原始字节流
    Returns:
        纯文本；每页之间用 "\n\n"（空行）分隔，让后续段落切分能感知到页边界
    降级行为:
        提取结果为空（扫描件没有文字层）时**不抛异常**，只打一条 warning，
        把"是否改用 OCR"的决定权交给调用方 extract_text —— 这样本函数保持单一职责
    """
    import fitz  # PyMuPDF 惰性导入：库体积大，只在解析 PDF 时加载，避免拖慢服务启动
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # 从字节加载 PDF（stream= 直接吃内存字节，不落磁盘）
    pages = []  # 存放每页文本
    # 1. 逐页抽取文字层
    for page in doc:  # 逐页
        pages.append(page.get_text("text"))  # "text" 模式按阅读顺序吐纯文本，比 blocks/dict 模式省事
    doc.close()  # 关闭：PyMuPDF 持有底层文件句柄，必须显式释放，否则文件多时句柄会泄漏
    # 2. 页间用空行拼接
    text = "\n\n".join(pages)  # 页间空行
    # 如果提取出来是空的，可能是扫描件 → 提示用 OCR
    if not text.strip():  # 注意用 strip() 判断：只有空白字符也算提取失败
        logger.warning("PDF 文本提取为空，可能是扫描件，需要 OCR 处理")
    return text


def extract_tables_from_pdf(file_bytes: bytes) -> list:
    """
    用 pdfplumber 提取 PDF 中的表格，返回 Markdown 格式表格文本列表

    Args:
        file_bytes: PDF 文件的原始字节流
    Returns:
        list[str]，每个元素是**一个表格**的 Markdown 文本（含 |---|---| 表头分隔行）；
        一个表格都没提取到时返回空列表 []，不会返回 None
    说明:
        与 extract_from_pdf 是互补关系：fitz 负责正文文字，表格数据往往只有 pdfplumber 才提得干净；
        统一转成 Markdown 是为了让表格在切分和检索后仍保留"行列对应"关系，模型才读得懂
    取舍:
        表格不在这里和正文合并，而是分开返回——让调用方决定要不要单独入库、单独切分（表格切碎会彻底失去意义）
    """
    import pdfplumber  # 延迟导入：pdfplumber 依赖较重（内部还带 pdfminer），按需加载即可
    tables_text = []  # 存放各表格的 Markdown 文本
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:  # 打开 PDF（BytesIO 走内存；with 保证异常时也能关闭）
        # 1. 逐页扫表格
        for page in pdf.pages:  # 逐页
            tables = page.extract_tables()  # 返回三维结构：页 -> 表格 -> 行 -> 单元格（无表格则为空列表）
            # 2. 每个表格转成 Markdown 文本
            for table in tables:  # 逐个表格
                if not table:  # 空表跳过（pdfplumber 偶尔会返回空列表，直接处理会下标越界）
                    continue
                # 把表格转成 Markdown table 格式
                md_lines = []  # 存放 Markdown 行
                for i, row in enumerate(table):  # 逐行（i 用来识别第一行=表头）
                    cells = [str(cell or "").strip() for cell in row]  # 单元格可能是 None（合并单元格/空行），统一转字符串再清空白
                    md_lines.append("| " + " | ".join(cells) + " |")  # Markdown 行
                    if i == 0:  # 第一行后加分隔行（Markdown 表格语法要求，否则渲染不出表格样式）
                        md_lines.append("|" + "|".join(["---"] * len(row)) + "|")
                tables_text.append("\n".join(md_lines))  # 一个表格的完整 Markdown 存入列表
    logger.info(f"PDF 表格提取：共 {len(tables_text)} 个表格")  # 日志：出现"表格没提出来"时可从这里先确认数量
    return tables_text  # 返回 Markdown 表格文本列表


def extract_from_wps_doc(file_bytes: bytes, suffix: str) -> str:
    """
    用 win32com 调本机 WPS 打开 .wps/.doc 另存为 txt 再读文本

    Args:
        file_bytes: 文件原始字节流
        suffix:     文件后缀（含点，如 ".wps" / ".doc"），由 extract_text 用 os.path.splitext 取原始大小写；
                    它决定临时文件用什么扩展名 —— COM 组件是靠扩展名识别文档格式的，后缀错了会打不开
    Returns:
        另存为 txt 后读回的纯文本
    异常:
        本机既没装 WPS 也没装 Word 时抛 HTTPException(500)；
        COM 调用本身失败（残留进程占用、文档损坏等）会抛原生异常，由上层兜底
    限制与坑:
        仅 Windows 可用（win32com 是 Windows COM 的 Python 绑定），Linux 部署需换 LibreOffice 方案；
        会真的拉起一个 WPS 进程，比纯 Python 解析慢；高并发时会排队甚至互抢文件，属于已知瓶颈
    """
    import pythoncom  # COM 初始化：win32com 使用前必须 CoInitialize，否则报"尚未调用 CoInitialize"
    import tempfile  # 临时文件：COM 只能按"路径"打开文档，所以必须落盘（本模块唯一落盘的解析器）
    pythoncom.CoInitialize()  # 当前线程初始化 COM（FastAPI 在线程池里执行，每个工作线程都要各自初始化）
    try:
        import win32com.client  # 延迟导入：只有 Windows 装了 pywin32 才存在，放函数里避免非 Windows 启动即崩
        # 1. 把上传字节流写成临时文件
        tmp_dir = tempfile.mkdtemp()  # 临时目录：每次新建一个，避免并发请求互相覆盖同名文件
        src_path = os.path.join(tmp_dir, f"input{suffix}")  # 临时源文件（后缀必须和原文件一致）
        out_path = os.path.join(tmp_dir, "output.txt")  # 临时输出（固定名，因为目录已经是独占的）
        with open(src_path, "wb") as f:  # 写入临时文件（wb=二进制写，不能改文本模式，否则老格式会被破坏）
            f.write(file_bytes)
        # 2. 启动 WPS/Word 应用对象，优先 WPS，失败再退 Word
        try:  # 先试 WPS
            wps = win32com.client.Dispatch("kwps.Application")  # WPS 文字的 COM ProgID
        except Exception:  # 没装 WPS
            try:  # 再试 Word
                wps = win32com.client.Dispatch("Word.Application")
            except Exception:
                raise HTTPException(500, "本机未安装 WPS 或 Word，无法解析 .wps/.doc 文件")
        # 3. 后台打开 -> 另存为 txt -> 关闭
        wps.Visible = False  # 后台运行：不弹窗口，否则会打断正在用电脑的人
        doc = wps.Documents.Open(src_path)  # 打开临时文件
        doc.SaveAs(out_path, 7)  # 另存为 Unicode txt（7 = wdFormatUnicodeText，用别的格式中文会乱码）
        doc.Close(False)  # 关闭（False=不保存修改，避免弹出"是否保存"对话框把流程卡死）
        wps.Quit()  # 退出 WPS 进程（不退出会残留进程、越积越多并占用文件句柄）
        # 4. 读回文本
        with open(out_path, "r", encoding="utf-8") as f:  # 读取 txt（上一步存的是 Unicode，这里用 utf-8 读）
            return f.read()  # 返回文本
    finally:
        pythoncom.CoUninitialize()  # 释放 COM：必须和 CoInitialize 配对，无论成功失败都执行，否则线程里 COM 引用泄漏


def extract_from_scanned_pdf(file_bytes: bytes) -> str:
    """
    用 PaddleOCR 对扫描件/图片型 PDF 做 OCR 识别

    调用时机：extract_text 先用 fitz 提文本，提到空（说明没有文字层 = 真扫描件）才会走到这里。

    Args:
        file_bytes: PDF 文件原始字节流（只能是 PDF，纯图片文件走的是注释里的 VLM 方案）
    Returns:
        识别出的纯文本；页与页之间用空行分隔，页内每行文字用 "\n" 连接
    异常:
        PaddleOCR 未安装时抛 HTTPException(500)，直接告诉用户缺什么依赖（比 ImportError 栈更好定位）
    性能:
        OCR 是整条入库链路最慢的一步（300 DPI 逐页渲染 + 模型推理），
        所以只在 fitz 提不出文字（真扫描件）时才调用，绝不做"顺便试一下"
    版本适配:
        构造参数与结果解析都按 PaddleOCR 3.x 写：2.x 的 use_angle_cls / show_log 已移除，
        返回值也从 "[[框, (文字, 置信度)], ...]" 变成了类字典的 OCRResult
    """
    try:
        import fitz  # 先用 PyMuPDF 把 PDF 渲染成图片
        from paddleocr import PaddleOCR  # OCR 引擎（惰性导入：paddle 依赖体积非常大，不用就不加载）
    except ImportError:
        raise HTTPException(500, "PaddleOCR 未安装，无法处理扫描件，请 pip install paddleocr paddlepaddle")

    # 3.x 的构造参数含义：
    #   use_textline_orientation=True 对应 2.x 的 use_angle_cls=True，处理倒置/旋转的文本行
    #   use_doc_orientation_classify（整页方向分类）和 use_doc_unwarping（去扭曲）默认开启，
    #   但输入是 PDF 直接渲染的规整位图（不是手机拍照），关掉这两个参数能省两次模型推理
    #   enable_mkldnn=False：PaddlePaddle 3.3 在 Windows 上走 oneDNN 加速时会在检测模型上报
    #     NotImplementedError（onednn_instruction.cc:118 ConvertPirAttribute2RuntimeAttribute），
    #     关掉 oneDNN 用朴素 CPU 实现即可绕开；代价是推理稍慢，正确性优先
    ocr = PaddleOCR(
        lang="ch",  # 中文模型（扫描件基本都是中文书）
        # 显式指定「移动端」检测/识别模型：3.x 默认给的是 PP-OCRv6_medium（中量级），
        # CPU 上跑 300 DPI 整页要几分钟一张；换成 mobile 版快一个数量级，印刷体书扫描件精度够用
        text_detection_model_name="PP-OCRv5_mobile_det",  # 文本检测（找文字框）
        text_recognition_model_name="PP-OCRv5_mobile_rec",  # 文本识别（认字）
        use_doc_orientation_classify=False,  # 关：不需要判断整页有没有倒置
        use_doc_unwarping=False,  # 关：不需要矫正拍摄畸变
        use_textline_orientation=True,  # 开：单行文字方向仍可能歪，交给它判
        enable_mkldnn=False,  # 关：绕开上面那个 oneDNN 的 Windows bug
    )

    import shutil  # 清理临时目录
    import tempfile  # 3.x 的 predict 传图片路径最稳妥（传字节流/数组要自己保证 BGR 通道顺序）

    tmp_dir = tempfile.mkdtemp()  # 临时目录：存放逐页渲染出来的 PNG
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # 打开 PDF
    pages_text = []  # 每页 OCR 结果
    try:
        # 1. 逐页渲染成位图再送去识别
        for i, page in enumerate(doc):  # 逐页
            pix = page.get_pixmap(dpi=300)  # 渲染为 300 DPI 图片（DPI 越高 OCR 越准，但耗时和内存也同步上涨）
            img_path = os.path.join(tmp_dir, f"page_{i}.png")  # 每页单独一个文件，避免同名覆盖带来的歧义
            pix.save(img_path)  # 落盘为 PNG（PyMuPDF 原生写盘，不用先在内存里编码再写）
            # 2. 识别整页：3.x 返回 OCRResult 列表，一页多栏排版时可能返回多个结果块
            results = ocr.predict(img_path)
            lines = []  # 本页文本行
            for res in results:  # 逐个结果块
                lines.extend(res["rec_texts"])  # rec_texts 是识别出的文本行列表；坐标/置信度在 rec_polys、rec_scores 里，这里不需要
            pages_text.append("\n".join(lines))  # 本页文本
    finally:
        doc.close()  # 关闭 PDF
        shutil.rmtree(tmp_dir, ignore_errors=True)  # 删临时目录：OCR 耗时较长，异常中断时也不留垃圾
    return "\n\n".join(pages_text)  # 页间空行（交给切分阶段当段落边界用）


# ==================== 扩展解析器（图抽取 / 表格整合 / Excel / PPT 已启用；VLM 见下方注释） ====================

def extract_images_from_pdf(file_bytes: bytes, output_dir: str = "./extracted_images") -> list:
    """
    提取 PDF 内嵌的所有图片（PyMuPDF）

    用途：PDF 里有插图、示意图、照片时，先把图片文件抽出来，
          再交给 OCR 或视觉大模型（VLM）理解内容。
    依赖：pip install pymupdf
    返回：保存的图片路径列表，如 ["extracted_images/p1_img0.png", ...]
    """
    import fitz  # PyMuPDF
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # 打开 PDF
    img_paths = []  # 存放图片路径
    for page_num, page in enumerate(doc):  # 逐页
        for img_index, img in enumerate(page.get_images(full=True)):  # 逐张内嵌图
            xref = img[0]  # 图片在 PDF 内部的引用 ID
            base_image = doc.extract_image(xref)  # 提取为字节
            ext = base_image["ext"]  # 图片格式（png/jpeg/...）
            img_bytes = base_image["image"]  # 图片字节
            filename = f"p{page_num + 1}_img{img_index}.{ext}"  # 文件名
            filepath = os.path.join(output_dir, filename)  # 完整路径
            with open(filepath, "wb") as f:  # 写入文件
                f.write(img_bytes)
            img_paths.append(filepath)  # 记录路径
    doc.close()
    logger.info(f"提取图片：共 {len(img_paths)} 张，保存到 {output_dir}")
    return img_paths  # 返回路径列表，后续可喂给 VLM


def extract_pdf_with_tables(file_bytes: bytes) -> str:
    """
    PDF 文本+表格一次性提取（PyMuPDFLoader 的 extract_tables 模式）

    用途：PDF 里有有线表格（赔偿标准表、量刑表等），不想分开调 pdfplumber，
          用 PyMuPDFLoader 一步到位，表格直接转 Markdown
    依赖：pip install langchain-community pymupdf
    导入：from langchain_community.document_loaders import PyMuPDFLoader
    返回：纯文本 + Markdown 表格，混合在一起
    """
    import tempfile  # 临时文件
    from langchain_community.document_loaders import PyMuPDFLoader  # LangChain 封装的加载器
    # PyMuPDFLoader 需要文件路径，不接受字节流，先写临时文件
    tmp_path = os.path.join(tempfile.mkdtemp(), "input.pdf")  # 临时 PDF
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    loader = PyMuPDFLoader(  # 初始化加载器
        tmp_path,
        mode="page",  # 按页返回 Document
        extract_tables="markdown",  # 表格转 Markdown 格式（可选 csv/html）
        extract_images=False,  # 不提取图片（这里只关心文字+表格）
    )
    docs = loader.load()  # 加载
    text = "\n\n".join(d.page_content for d in docs)  # 拼接所有页
    logger.info(f"PDF 文本+表格提取完成：{len(docs)} 页")
    return text


def extract_from_excel(file_bytes: bytes) -> str:
    """
    读取 Excel 文件（.xlsx/.xls），逐 Sheet 转 Markdown 表格

    用途：知识库数据源包含 Excel 表格（如赔偿标准表、量刑对照表）
    依赖：pip install openpyxl
    返回：所有 Sheet 的内容拼成 Markdown 表格文本
    """
    import openpyxl  # Excel 读取库
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)  # data_only=True 取公式计算后的值
    sheets_text = []  # 存放各 Sheet 的文本
    for ws in wb.worksheets:  # 逐 Sheet
        md_lines = [f"## {ws.title}"]  # Sheet 名做标题
        for i, row in enumerate(ws.iter_rows(values_only=True)):  # 逐行
            cells = [str(c or "").strip() for c in row]  # 清理单元格
            if not any(cells):  # 全空行跳过
                continue
            md_lines.append("| " + " | ".join(cells) + " |")  # Markdown 行
            if i == 0:  # 第一行后加分隔行
                md_lines.append("|" + "|".join(["---"] * len(row)) + "|")
        sheets_text.append("\n".join(md_lines))  # 拼接
    logger.info(f"Excel 解析完成：{len(wb.worksheets)} 个 Sheet")
    return "\n\n".join(sheets_text)  # Sheet 间空行


def extract_from_pptx(file_bytes: bytes) -> str:
    """
    读取 PowerPoint（.pptx），逐幻灯片提取文字

    用途：知识库数据源包含 PPT 课件、汇报材料
    依赖：pip install python-pptx
    返回：每页幻灯片的文字，页间空行
    """
    from pptx import Presentation  # PPT 读取库
    prs = Presentation(io.BytesIO(file_bytes))  # 从字节加载
    slides_text = []  # 每页文本
    for i, slide in enumerate(prs.slides, start=1):  # 逐页
        lines = [f"## 第 {i} 页"]  # 页码标题
        for shape in slide.shapes:  # 逐元素（文本框/表格/图片）
            if shape.has_text_frame:  # 文本框
                text = shape.text_frame.text.strip()  # 提取文字
                if text:
                    lines.append(text)
            if shape.has_table:  # 表格 → 转 Markdown
                table = shape.table
                for r, row in enumerate(table.rows):  # 逐行
                    cells = [cell.text.strip() for cell in row.cells]  # 清理
                    lines.append("| " + " | ".join(cells) + " |")  # Markdown 行
                    if r == 0:  # 第一行后加分隔行
                        lines.append("|" + "|".join(["---"] * len(row.cells)) + "|")
        slides_text.append("\n".join(lines))  # 拼接本页
    logger.info(f"PPTX 解析完成：{len(prs.slides)} 页")
    return "\n\n".join(slides_text)  # 页间空行


def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, bool]:
    """
    根据后缀自动选择解析器（本模块对外唯一入口）

    Args:
        file_bytes: 上传文件的原始字节流（由上传接口读入后原样传入，不做任何预处理）
        filename:   原始文件名，这里只用它的后缀判断类型（大小写不敏感）
    Returns:
        (纯文本, 是否按 Markdown 标题结构切分)
        - 第一个元素：解析出的完整文本，交给 text_splitter；
        - 第二个元素：True 表示是 Markdown，调用方应强制用标题结构策略切分
          （本函数只给"建议"，不做切分，保持职责单一）
    异常:
        不支持的后缀抛 HTTPException(400)；
        .wps/.doc 少了本机 WPS/Word、扫描件少了 PaddleOCR 时，由被调函数抛 500
    说明:
        PDF 走"两级策略"：先试轻量的文本层提取，为空才上 OCR —— 因为 OCR 慢几十倍，
        不能一上来就 OCR；这个判断放在这里而不是 extract_from_pdf 内部，是为了让每个解析函数都保持"纯解析"
    """
    suffix = filename.lower()  # 统一小写，避免 .PDF / .Docx 这类大小写导致漏判
    # 1. 纯文本类：直接在内存里按 UTF-8 解码
    if suffix.endswith(".md"):  # Markdown：结构信息（标题层级）很有价值，所以第二个返回值给 True
        return file_bytes.decode("utf-8", errors="ignore"), True  # errors="ignore" 跳过非法字节，避免个别坏编码直接 500
    elif suffix.endswith(".txt"):  # 纯文本：没有结构，用前缀策略切分即可
        return file_bytes.decode("utf-8", errors="ignore"), False
    # 2. Office 系：交给各自的专用解析器
    elif suffix.endswith(".docx"):  # Word/WPS docx
        return extract_from_docx(file_bytes), False
    elif suffix.endswith(".pdf"):  # PDF
        text = extract_from_pdf(file_bytes)  # 先试文本提取（快，绝大多数电子版 PDF 都能直接出来）
        if not text.strip():  # 空的 → 可能是扫描件（没有文字层）
            logger.info("检测到扫描件，启动 OCR ...")
            text = extract_from_scanned_pdf(file_bytes)  # OCR（慢，只在前一步失败时才走）
        return text, False
    elif suffix.endswith(".wps") or suffix.endswith(".doc"):  # WPS/旧版 Word
        return extract_from_wps_doc(file_bytes, os.path.splitext(filename)[1]), False  # 传原始大小写后缀，COM 靠它识别格式
    elif suffix.endswith(".xlsx"):  # Excel 电子表格（openpyxl 读不了旧版 .xls，故只认 .xlsx）
        return extract_from_excel(file_bytes), False
    elif suffix.endswith(".csv"):  # CSV：本身就是纯文本，直接解码
        return file_bytes.decode("utf-8", errors="ignore"), False
    elif suffix.endswith(".pptx"):  # PowerPoint
        return extract_from_pptx(file_bytes), False
    # elif suffix.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):  # 纯图片
    #     return extract_image_with_vlm(file_bytes, filename), False  # VLM 未启用：需 vlm_api_key 与视觉模型服务
    else:
        raise HTTPException(400, f"暂不支持 {filename}，请上传 txt/md/csv/docx/wps/doc/pdf/xlsx/pptx")  # 400：属于客户端传错文件，不是服务端故障
