"""离线知识库完整实现：PDF解析→清洗→分块→BGE向量化→Milvus同步。

这份文件对应RAG的“离线部分”。离线部分不直接回答用户问题，而是把原始医学PDF
提前加工成可检索的向量数据。答辩时可以把它概括为四步：
1. ``prepare``：从PDF提取文字，清洗后切成知识块；
2. ``embed``：用本地BGE模型把每个知识块转成向量；
3. ``sync``：把正文、来源、块号和向量写入Milvus；
4. ``check``：检查PyMuPDF、PDFPlumber、PaddleOCR和MinerU是否可用。

命令行入口在文件末尾的 :func:`main`，各阶段也都能被单元测试或其他Python代码单独调用。
"""
from __future__ import annotations

# 标准库按职责使用：argparse解析命令；hashlib校验快照；json读写中间数据。
import argparse
import hashlib
import importlib.util
import json
import math  # 用isfinite拒绝NaN、正无穷和负无穷，避免非法向量进入Milvus。
import os  # 读取.env.local加载后的环境变量。
import re  # 清洗正文、判断稀疏页面以及校验SHA-256格式。
import shlex  # 把MINERU_COMMAND安全地拆成“命令+参数”列表。
import shutil  # 用which判断mineru或magic-pdf命令是否存在。
import subprocess  # 以子进程方式调用MinerU命令行程序。
import sys  # 找不到独立mineru命令时，尝试当前Python解释器的-m mineru。
import tempfile  # 原子写文件以及隔离MinerU临时输出，避免留下半成品。
from datetime import datetime  # 为知识更新和报告生成带时区的时间戳。
from pathlib import Path  # 用跨平台Path对象拼接Windows和Ubuntu/WSL路径。


# __file__是当前脚本；parents[1]向上两级得到项目根目录D:/rag-roleplay。
BASE = Path(__file__).resolve().parents[2]
# 在线问答也读取同名collection，因此离线写入和在线检索必须使用同一个名称。
COLLECTION = "doctor_knowledge"
# 默认每块约250个字符，兼顾语义完整和检索粒度；用户仍可用--chunk-size覆盖。
CHUNK_SIZE = 250
# 小于40字符的尾块信息太少，后面会并入相邻块，而不是静默丢弃。
MIN_CHUNK_SIZE = 40
# 教学版限定最多1999条，确保一次查询能完整检查全部数据，并控制本机资源占用。
CORE_MAX_ROWS = 1999
# Milvus查询上限比安全行数多1，用于识别“数据已达到上限、无法完整校验”的情况。
QUERY_LIMIT = 2000
# 必需字段组成最小Milvus schema：稳定ID、向量、原文、文档来源、文档内块序号。
REQUIRED_FIELDS = {"id", "vector", "text", "source", "chunk_index"}
# 旧库不一定有这些扩展字段，所以它们存在时才写入，不存在时仍保持兼容。
OPTIONAL_FIELDS = {"created_at", "updated_at", "summary"}


def load_env(project_root: Path = BASE) -> None:
    """读取项目环境配置，同时保留终端中已经设置的值。

    参数：
        project_root: 项目根目录，默认使用本文件推导出的 ``BASE``。
    返回：
        无返回值；配置会写入当前Python进程的 ``os.environ``。

    ``RAG_ENV_FILE`` 可指定其他配置文件；``setdefault`` 表示终端已有变量优先，
    这样开发、测试和生产可以复用代码，只替换配置。
    """
    # 优先读取RAG_ENV_FILE指定路径，否则使用项目根目录下的.env.local。
    path = Path(os.environ.get("RAG_ENV_FILE", project_root / ".env.local"))
    # 配置文件不是必需品；不存在时保留系统现有环境变量并直接结束。
    if not path.exists():
        return
    # utf-8-sig兼容Windows记事本可能写入的BOM；splitlines逐行处理配置。
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        # 跳过空行、注释和不含等号的行，只接受KEY=VALUE格式。
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            # 只按第一个等号切分，保证API地址或密码中的后续等号仍属于value。
            key, value = line.split("=", 1)
            # 去除首尾空白，再剥掉两端连续的单/双引号字符；它不校验引号是否成对。
            # setdefault不会覆盖终端里已设置的同名变量。
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _pymupdf_text(pdf_path: Path) -> tuple[str, float]:
    """用PyMuPDF读取可复制文字，并计算平均每页字符数。

    ``pdf_path`` 是待解析PDF路径；返回 ``(text, density)``，其中text用换页符分隔，
    density是去掉换行和换页符后的平均字符数，可用于判断扫描版PDF是否需要OCR。
    """
    # 延迟导入使未使用该解析器的命令不必提前加载PyMuPDF。
    import pymupdf

    # with会在读取结束或发生异常时自动关闭PDF文件句柄。
    with pymupdf.open(pdf_path) as document:
        # page.get_text()读取PDF文字层；strip去掉每页首尾无意义空白。
        pages = [page.get_text().strip() for page in document]
    # \f保留页边界，使auto模式能逐页识别哪些页面文字太少。
    text = "\n\f\n".join(pages)
    # max(..., 1)避免空PDF发生除零；density保留是为了兼容调用方和教学展示。
    density = len(text.replace("\n", "").replace("\f", "")) / max(len(pages), 1)
    return text, density


def _paddle_lines(result) -> list[str]:
    """把不同PaddleOCR版本的返回结构统一整理成文字行列表。

    参数 ``result`` 可能是新版结果对象、JSON字符串、字典或旧版嵌套列表；
    返回值只包含去掉首尾空白后的非空文字。递归处理让代码兼容多种版本。
    """
    # 新版结果对象暴露json属性；旧版已经是list/dict，直接保留原值。
    payload = getattr(result, "json", result)
    # 某些版本的json是方法，callable判断后调用；属性形式则无需调用。
    payload = payload() if callable(payload) else payload
    # 如果拿到JSON文本，先尝试解码成Python对象。
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            # 普通文字不是JSON时，把它本身当作一条OCR结果；空字符串返回空列表。
            return [payload.strip()] if payload.strip() else []
    # 新版PaddleOCR常把结果包装在res字典中。
    if isinstance(payload, dict):
        payload = payload.get("res", payload)
        if isinstance(payload, dict):
            # rec_texts是识别出的文本数组；str兼容非字符串值，if过滤空文本。
            return [str(text).strip() for text in payload.get("rec_texts", [])
                    if str(text).strip()]
        # res仍可能是嵌套列表，因此交给同一函数递归展开。
        return _paddle_lines(payload)
    # 旧版结构通常是[文字框坐标, [识别文字, 置信度]]。
    lines = []
    for item in payload or []:
        # 只处理至少包含“坐标+识别结果”两个元素的列表或元组。
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            value = item[1]
            # 标准旧版结果：value[0]就是识别文字。
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                lines.append(value[0].strip())
            # 更深层的分页结果没有直接文字时，继续递归解析item。
            elif isinstance(item[0], (list, tuple)):
                lines.extend(_paddle_lines(item))
    # 最后再次过滤空行，确保后续拼接不会制造无意义段落。
    return [line for line in lines if line]


def _paddleocr_text(pdf_path: Path, dpi: int = 180,
                    page_numbers: list[int] | None = None) -> str:
    """把PDF指定页面渲染成图片，再用PaddleOCR识别中文。

    参数：
        pdf_path: PDF文件路径。
        dpi: 渲染分辨率，默认180；越高越清晰，但耗时和内存也越高。
        page_numbers: 从0开始的页码列表；为None时识别整本PDF。
    返回：
        OCR文字，页面之间以 ``\n\f\n`` 分隔，便于与原PDF页号对应。
    异常：
        依赖缺失时转换为带安装指引的RuntimeError。
    """
    try:
        # numpy把PyMuPDF像素缓冲区转换成PaddleOCR接受的图像数组。
        import numpy as np
        import pymupdf
        from paddleocr import PaddleOCR
    except ImportError as error:
        # 保留原异常作为cause，既方便用户理解，也方便开发者排查具体缺失包。
        raise RuntimeError("PaddleOCR未安装，请先运行deploy/local/setup_parsers.sh") from error
    try:
        # PaddleOCR 3.x参数：中文模型，并关闭本项目不需要的方向/展平预处理。
        ocr = PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                        use_doc_unwarping=False, use_textline_orientation=False,
                        enable_mkldnn=False)
    except TypeError:
        # 旧版不认识3.x参数时退回2.x写法，实现版本兼容。
        ocr = PaddleOCR(lang="ch", use_angle_cls=True, show_log=False)
    # PDF坐标以72 DPI为基准，所以dpi/72得到渲染缩放倍数；pages保存每页文字。
    pages, scale = [], dpi / 72
    # set让“当前页是否需要OCR”的判断接近O(1)；None表示不筛选页码。
    selected_pages = set(page_numbers) if page_numbers is not None else None
    with pymupdf.open(pdf_path) as document:
        # enumerate默认从0开始，正好与page_numbers的约定一致。
        for page_number, page in enumerate(document):
            # auto混合解析时，只对PyMuPDF文字不足的页面做OCR，节省时间。
            if selected_pages is not None and page_number not in selected_pages:
                continue
            # alpha=False输出RGB而非透明通道图，Matrix控制DPI缩放。
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            # 从原始字节零拷贝读取，再还原为“高×宽×通道数”的图像数组。
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            # 新版使用predict，旧版使用ocr；hasattr在运行时选择相应接口。
            raw = ocr.predict(image) if hasattr(ocr, "predict") else ocr.ocr(image, cls=True)
            # 一页可能有多个结果对象，逐个统一成文字行。
            lines = []
            for result in raw or []:
                lines.extend(_paddle_lines(result))
            # 保留空页占位，才能与稀疏页页码一一对应。
            pages.append("\n".join(lines))
    # 用与PyMuPDF相同的页分隔符返回，便于混合替换。
    return "\n\f\n".join(pages)


def _installed(module: str) -> bool:
    """检查Python模块能否被当前解释器找到，不真正导入重量级依赖。

    ``module`` 是模块名（如 ``paddleocr``）；找到返回True，否则返回False。
    ``find_spec`` 比直接import更轻量，适合 ``check`` 命令快速检查环境。
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # 包安装损坏或父模块信息不完整时，也统一视为“当前不可用”。
        return False


def _mineru_command() -> list[str]:
    """寻找可执行的MinerU命令，并以参数列表形式返回。

    配置了 ``MINERU_COMMAND`` 时只探测该命令；未配置时依次探测 ``mineru``、
    ``magic-pdf``。上述候选不可执行时，再查找当前虚拟环境的 ``mineru`` 文件；
    都找不到时返回空列表。
    """
    # strip去掉配置两端空白；空值表示让程序自动探测。
    configured = os.environ.get("MINERU_COMMAND", "").strip()
    # shlex.split支持配置中包含额外参数，并避免把整串误当作可执行文件名。
    candidates = [shlex.split(configured)] if configured else [["mineru"], ["magic-pdf"]]
    for command in candidates:
        # command[0]是可执行程序；which会按当前PATH检查它是否存在。
        if command and shutil.which(command[0]):
            return command
    # MinerU 2.x 提供 CLI，但没有 python -m mineru 入口。
    local_cli = Path(sys.executable).with_name("mineru")
    return [str(local_cli)] if local_cli.is_file() else []


def _mineru_text(pdf_path: Path) -> str:
    """调用MinerU解析复杂排版PDF，并返回它生成的Markdown正文。

    ``pdf_path`` 是输入PDF。MinerU适合公式、标题层级和复杂版面；解析输出放入
    自动清理的临时目录，因此不会污染项目数据目录。
    """
    # 获取经过环境变量和PATH探测后的命令，例如["mineru"]。
    command = _mineru_command()
    if not command:
        raise RuntimeError("MinerU未安装，请先运行deploy/local/setup_parsers.sh")
    # 临时目录离开with后自动删除；prefix便于排障时识别用途。
    with tempfile.TemporaryDirectory(prefix="rag-mineru-") as temp_dir:
        # MinerU会在output_dir内部继续创建层级目录，后面使用rglob搜索。
        output_dir = Path(temp_dir) / "output"
        # 参数-p指定输入PDF，-o指定输出目录；列表传参不会经过shell拼接。
        # pipeline明确使用本机CPU；关闭公式和表格模型可降低答辩电脑的内存占用。
        has_backend = any(item in {"-b", "--backend"} for item in command)
        mineru_args = [] if has_backend else [
            "-b", "pipeline", "-d", "cpu", "--source", "modelscope",
            "-f", "false", "-t", "false",
        ]
        process = subprocess.run(
            [*command, *mineru_args, "-p", str(pdf_path), "-o", str(output_dir)],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        # returncode非0代表外部程序失败，把末尾1200字符带回但避免错误过长。
        if process.returncode:
            detail = (process.stderr or process.stdout)[-1200:].strip()
            raise RuntimeError(f"MinerU解析失败：{detail}")
        # 不假设MinerU具体输出层级，递归寻找所有Markdown文件。
        markdown_files = list(output_dir.rglob("*.md"))
        if not markdown_files:
            detail = (process.stderr or process.stdout)[-1200:].strip()
            message = "MinerU没有生成Markdown结果"
            raise RuntimeError(f"{message}：{detail}" if detail else message)
        # 多个Markdown中通常最大文件是正文，辅助说明文件较小。
        result_file = max(markdown_files, key=lambda path: path.stat().st_size)
        # MinerU输出统一按UTF-8读取，作为后续清洗和分块的原文。
        return result_file.read_text(encoding="utf-8")


def _table_text(pdf_path: Path) -> str:
    """使用PDFPlumber抽取PDF中的表格，并转成可检索的纯文本。

    ``pdf_path`` 是输入文件；返回多个表格拼接后的字符串。每个表格带页码和表号，
    单元格以竖线分隔，便于向量模型保留“这一行有哪些列”的结构信息。
    """
    try:
        # PDFPlumber专门补充PyMuPDF正文提取不擅长的表格结构。
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("pdfplumber未安装，请先运行deploy/local/setup_parsers.sh") from error
    # tables按发现顺序存储每张表格的文本表示。
    tables = []
    with pdfplumber.open(pdf_path) as document:
        # 答辩展示使用从1开始的页码，更符合用户看到的PDF页码习惯。
        for page_number, page in enumerate(document.pages, 1):
            # 一页可能提取出多张表，因此再给每张表从1编号。
            for table_number, table in enumerate(page.extract_tables(), 1):
                # None表示空单元格，用空串替代；strip清除单元格多余空白。
                rows = [" | ".join((cell or "").strip() for cell in row) for row in table]
                # 页码和表号成为可追溯标记，正文行保持换行结构。
                tables.append(f"[第{page_number}页表格{table_number}]\n" + "\n".join(rows))
    # 表格之间留空行，防止两个表末尾和开头粘成一个段落。
    return "\n\n".join(tables)


def parse_document(pdf_path: Path, engine: str = "auto", complex_layout: bool = False,
                   min_chars_per_page: int = 80, include_tables: bool = False) -> tuple[str, str]:
    """选择PDF解析器，返回正文和实际使用的解析器名称。

    参数：
        pdf_path: 输入PDF路径。
        engine: ``auto``、``pymupdf``、``paddleocr`` 或 ``mineru``。
        complex_layout: auto模式下为True时直接选MinerU处理复杂版式。
        min_chars_per_page: auto模式判断文字层是否稀疏的每页字符阈值。
        include_tables: 为True时再用PDFPlumber追加表格文本。
    返回：
        ``(正文, 实际解析器名称)``；混合文档可能返回 ``pymupdf+paddleocr``。
    """
    # 即使调用者传入字符串，也立即转成Path，统一后续路径操作。
    pdf_path = Path(pdf_path)
    # 同时检查“确实是文件”和“.pdf扩展名”，避免把目录或其他文件交给解析器。
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"不是有效PDF文件：{pdf_path}")
    # 白名单阻止拼写错误悄悄走进错误分支。
    if engine not in {"auto", "pymupdf", "paddleocr", "mineru"}:
        raise ValueError(f"不支持的解析器：{engine}")
    # 阈值0代表不把任何页面因“文字少”判为异常；负数没有业务意义。
    if min_chars_per_page < 0:
        raise ValueError("min_chars_per_page不能小于0")
    # complex_layout只影响auto模式；显式指定解析器时尊重用户选择。
    selected = "mineru" if engine == "auto" and complex_layout else engine
    # auto先走速度快的文字层提取；显式pymupdf也进入同一分支。
    if selected in {"auto", "pymupdf"}:
        # density目前用于保留解析统计接口，逐页判定则由下面的pages完成。
        text, density = _pymupdf_text(pdf_path)
        # 先记录实际已执行PyMuPDF，后面如果OCR替换页面再修改名称。
        selected = "pymupdf"
        if engine == "auto":
            # 按保留的换页符恢复页面，避免用全书平均值漏掉夹杂的扫描页。
            pages = text.split("\n\f\n")
            # 删除所有空白后字符仍少于阈值的页面，被视为需要OCR的稀疏页。
            sparse = [index for index, page in enumerate(pages)
                      if len(re.sub(r"\s+", "", page)) < min_chars_per_page]
            if sparse:
                # 只OCR稀疏页，而不是重跑整本PDF；split后与sparse顺序一一对应。
                ocr_pages = _paddleocr_text(pdf_path, page_numbers=sparse).split("\n\f\n")
                # 页数不一致会造成内容放错页，宁可中止也不写入错误知识。
                if len(ocr_pages) != len(sparse):
                    raise RuntimeError("PaddleOCR返回页数与待识别页数不一致")
                # zip把每个原页索引和对应OCR正文配成一对，替换文字层不足的页面。
                for index, ocr_text in zip(sparse, ocr_pages):
                    # OCR为空时保留原文字，避免失败识别反而删除已有内容。
                    pages[index] = ocr_text.strip() or pages[index]
                # 替换完成后重新拼出整本文本，并继续保留页面边界。
                text = "\n\f\n".join(pages)
                # 全部页面OCR与部分页面OCR使用不同名称，报告可如实说明解析方式。
                selected = "paddleocr" if len(sparse) == len(pages) else "pymupdf+paddleocr"
    elif selected == "paddleocr":
        # 用户显式指定OCR时，对全部页面进行图像识别。
        text = _paddleocr_text(pdf_path)
    else:
        # 剩余合法选项只有mineru，用于标题、公式、图片较多的复杂版式。
        text = _mineru_text(pdf_path)
    # 表格提取是附加能力，可与三种正文解析方式组合使用。
    if include_tables:
        tables = _table_text(pdf_path)
        # 只有真正提取到表格才追加，避免正文末尾产生无意义空行。
        text = f"{text}\n\n{tables}" if tables else text
    # 去除整份文本首尾空白；第二项让调用者记录实际解析器，支持数据追溯。
    return text.strip(), selected


def clean_text(raw: str) -> str:
    """执行轻量数据清洗，同时尽量保留医学原意。

    ``raw`` 是解析器输出的原始文字；返回按换行组织的清洗文本。规则会删除空行、
    只含数字/空白/横线的整行、目录连续点和重复空格。注意：像 ``120-130`` 这样
    单独占一行的医学数字区间也会被当前规则删除，建库抽查时必须确认没有误删。
    """
    # output只收集通过规则的有效文字行。
    output = []
    # 换页符转成普通换行，使不同解析器输出都进入相同逐行清洗流程。
    for line in raw.replace("\f", "\n").splitlines():
        # 去掉每行首尾空格，但保留内容内部的正常单空格。
        text = line.strip()
        # 删除空行，以及只包含数字、空白和横线的整行；该规则也可能命中纯数值区间。
        if not text or re.fullmatch(r"[\d\s\-—]+", text):
            continue
        # 三个及以上连续英文句点通常是目录引导符，不是医学语义。
        text = re.sub(r"\.{3,}", "", text)
        # 连续空格或制表符合并成一个空格，减少无意义的向量输入差异。
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text:
            output.append(text)
    # 每个清洗后的原始行仍单独成段，为后续按段落分块提供边界。
    return "\n".join(output)


def split_chunks(text: str, size: int = CHUNK_SIZE,
                 min_size: int = MIN_CHUNK_SIZE) -> list[str]:
    """按段落聚合知识块，并把过短尾块合并到相邻内容。

    参数：
        text: 已清洗文本，每行视作一个自然段落。
        size: 目标块大小，默认250字符；不是绝对上限，因为短尾块可能向前合并。
        min_size: 最小有效块大小，默认40字符。
    返回：
        按原文顺序排列的知识块列表。

    这种分块保留段落语义，同时限制大多数块的长度，便于BGE向量化和精确召回。
    """
    # 三个条件同时确保range步长有效，且最小块不会大于目标块。
    if size <= 0 or min_size <= 0 or min_size > size:
        raise ValueError("分块参数必须满足0 < min_size <= size")
    # 删除空行并保留非空段落顺序，避免生成空知识块。
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    # chunks保存第一轮结果；buffer暂存可继续拼接的短段落。
    chunks, buffer = [], ""
    for paragraph in paragraphs:
        # 单个段落已超过目标大小时，不能继续与其他段落合并。
        if len(paragraph) > size:
            # 先把此前积累的短段落落盘，保证原文先后顺序不被打乱。
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            # range按size移动，把超长段落切成连续片段；末尾可能短于size。
            chunks.extend(paragraph[index:index + size] for index in range(0, len(paragraph), size))
        # 加上当前段落和一个换行后会超出目标大小，就结束当前buffer。
        elif len(buffer) + len(paragraph) + 1 > size:
            if buffer.strip():
                chunks.append(buffer.strip())
            # 当前段落成为下一知识块的起点，换行用于保持段落边界。
            buffer = paragraph + "\n"
        else:
            # 未超长时继续聚合相邻短段落，提高单块语义完整度。
            buffer += paragraph + "\n"
    # 循环结束仍有内容时必须加入，不能丢掉文档末尾。
    if buffer.strip():
        chunks.append(buffer.strip())
    # 第二轮专门处理第一轮可能产生的过短片段。
    merged, pending = [], ""
    for chunk in chunks:
        # pending是上一次暂存的短块；与当前块拼接时保留换行。
        combined = f"{pending}\n{chunk}".strip() if pending else chunk
        # 合并后仍太短就继续等待下一块，而不是把低信息块直接丢弃。
        if len(combined) < min_size:
            pending = combined
        else:
            # 达到最小长度后才作为正式块输出。
            merged.append(combined)
            pending = ""
    # 文档最后可能剩一个短尾块；优先并入最后一个正式块。
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n{pending}"
        else:
            # 整篇文本都不足min_size时仍保留内容，避免小文档完全消失。
            merged.append(pending)
    return merged


def _write_text(path: Path, text: str) -> None:
    """以“临时文件→原子替换”的方式安全写入UTF-8文本。

    ``path`` 是最终路径，``text`` 是完整内容。若写入中断，旧文件仍保留；只有临时
    文件写完后才用 ``replace`` 替换目标，避免后续读取到半个JSONL或半个报告。
    """
    # parents=True递归创建目录；exist_ok=True表示目录已有时不报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标文件位于同一目录，提高replace原子成功的可能性。
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        # 一次写入完整内容；with退出时自动flush并关闭句柄。
        handle.write(text)
        # 保存临时文件名，因为退出with后handle对象不再用于写入。
        temp_path = Path(handle.name)
    # 同文件系统原子替换目标；已有文件会被完整新版本覆盖。
    temp_path.replace(path)


def _write_json(path: Path, payload) -> None:
    """把Python对象格式化为便于人工阅读的UTF-8 JSON文件。"""
    # ensure_ascii=False直接保留中文；indent=2让答辩时可打开检查字段。
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """写JSON Lines文件：一行一条知识块，便于流式读取和定位坏行。"""
    # 每条记录独立序列化并补换行，再由_write_text统一安全落盘。
    _write_text(path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def _sha256(path: Path) -> str:
    """分块计算文件SHA-256，返回64位十六进制摘要。"""
    # 新建独立摘要对象，避免不同文件的内容累计到一起。
    digest = hashlib.sha256()
    # 二进制读取保证Windows和Linux换行处理不会改变哈希。
    with path.open("rb") as handle:
        # 每次读取1 MiB，避免大向量文件一次性占满内存；读到b""时停止。
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    # hexdigest用于写入JSON，也便于与命令行sha256工具核对。
    return digest.hexdigest()


def _embedding_manifest(output_dir: Path, names: list[str]) -> Path:
    """为本次向量输出建立带SHA-256的完整快照清单。

    ``output_dir`` 是embeddings目录，``names`` 只包含本轮生成的JSONL文件名；
    返回清单路径。后续 ``sync --prune`` 必须验证这份清单，防止残缺目录误删知识。
    """
    # 清单固定命名，方便默认同步命令自动发现。
    path = output_dir / "snapshot-manifest.json"
    payload = {
        # version为未来格式升级预留；created_at记录快照生成时间和本地时区。
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        # sorted保证顺序稳定；每个哈希证明同步时文件仍是本轮完整内容。
        "files": [{"name": name, "sha256": _sha256(output_dir / name)}
                  for name in sorted(names)],
    }
    _write_json(path, payload)
    return path


def _prepare_documents(project_root: Path = BASE, input_path: Path | None = None,
                       engine: str = "auto", complex_layout: bool = False,
                       include_tables: bool = False, size: int = CHUNK_SIZE,
                       min_size: int = MIN_CHUNK_SIZE) -> list[dict]:
    """执行“解析→清洗→分块”，并保存三个阶段的中间产物。

    参数：
        project_root: 项目根目录，用于定位 ``.env.local``、输入和中间产物目录。
        input_path: 为None时批量处理 ``data/raw/*.pdf``，否则只处理指定PDF。
        engine: 正文解析器名称，含 ``auto/pymupdf/paddleocr/mineru``。
        complex_layout: 为True时让auto模式选MinerU处理复杂版式。
        include_tables: 为True时追加PDFPlumber提取的表格文字。
        size/min_size: 分块的目标长度和允许保留的最小尾块长度。
    返回：每份文档的来源、实际解析器、字符数和块数组成的字典列表。
    """
    # 先读配置，确保自定义MinerU命令在解析器探测前已经生效。
    load_env(project_root)
    # 项目约定原始资料放在data/raw，不修改用户的原PDF。
    raw_dir = project_root / "data" / "raw"
    # 单文件模式保持用户指定路径；批量模式按文件名排序，保证结果可复现。
    files = [Path(input_path)] if input_path else sorted(raw_dir.glob("*.pdf"))
    if not files:
        raise RuntimeError("data/raw中没有PDF文件")
    # results汇总全部文档，但每份文档的中间文件会分别保存。
    results = []
    for pdf_path in files:
        # 位置参数依次是路径、引擎、复杂版式；表格开关用关键字提高可读性。
        text, selected = parse_document(pdf_path, engine, complex_layout,
                                        include_tables=include_tables)
        if not text:
            raise RuntimeError(f"{pdf_path.name}没有解析出文字")
        # 清洗只删格式噪声，不改写医学内容。
        cleaned = clean_text(text)
        # 按当前命令指定的目标长度和最小长度生成知识块。
        chunks = split_chunks(cleaned, size, min_size)
        if not chunks:
            raise RuntimeError(f"{pdf_path.name}清洗后没有可用知识块")
        # stem是不含.pdf的文件名，用作各阶段文件名和知识来源标识。
        stem = pdf_path.stem
        # parsed保留最接近解析器输出的文本，方便检查OCR或MinerU效果。
        parsed_path = project_root / "data" / "parsed" / f"{stem}.txt"
        _write_text(parsed_path, text)
        # 同名meta文件记录实际引擎和字符数，实现解析结果可追溯。
        _write_json(parsed_path.with_suffix(".meta.json"), {
            "source": pdf_path.name, "engine": selected, "characters": len(text)
        })
        # cleaned保存去噪后、尚未分块的正文，便于说明数据清洗前后差异。
        _write_text(project_root / "data" / "cleaned" / f"{stem}.txt", cleaned)
        # id供文件阶段追踪；source和index组合是同步Milvus时的稳定业务键。
        records = [{"id": f"{stem}_{index:03d}", "text": chunk, "source": stem,
                    "index": index} for index, chunk in enumerate(chunks)]
        # chunks阶段一行一个对象，后续向量化可逐条读取正文和来源。
        _write_jsonl(project_root / "data" / "chunks" / f"{stem}.jsonl", records)
        # 返回与日志都只保存摘要，不把全部正文打印到终端。
        result = {"source": pdf_path.name, "engine": selected, "characters": len(text),
                  "chunks": len(chunks)}
        results.append(result)
        # PREPARED前缀方便部署脚本和人工快速识别成功结果。
        print(f"PREPARED: {pdf_path.name} 引擎={selected} 字符={len(text)} 块={len(chunks)}")
    return results


def embed_chunks(project_root: Path = BASE, model_name: str | None = None, batch_size: int = 16,
                 chunks_dir: Path | None = None, output_dir: Path | None = None,
                 model_factory=None) -> list[dict]:
    """用本地BGE把知识块转成归一化向量，并保存JSONL快照。

    参数：
        project_root: 项目根目录。
        model_name: ``models`` 下的模型目录名；省略时读RAG_EMBED_MODEL。
        batch_size: 单次送入模型的知识块数，越大通常越快但更占内存。
        chunks_dir/output_dir: 测试或自定义数据时覆盖默认输入、输出目录。
        model_factory: 默认是SentenceTransformer；测试可注入假模型而不下载权重。
    返回：
        每个JSONL文件的记录数、向量维度和模型名摘要。
    """
    # 环境变量决定默认模型和运行设备，因此先加载.env.local。
    load_env(project_root)
    # 参数优先于环境变量；都未提供时使用轻量中文BGE默认模型。
    model_name = model_name or os.environ.get("RAG_EMBED_MODEL", "bge-small-zh-v1.5")
    # Path统一处理调用者传入字符串或Path的情况。
    chunks_dir = Path(chunks_dir or project_root / "data" / "chunks")
    output_dir = Path(output_dir or project_root / "data" / "embeddings")
    # 只读取prepare生成的JSONL，并排序保证运行顺序稳定。
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise RuntimeError("data/chunks中没有知识块，请先执行prepare")
    # SentenceTransformer不接受0或负批次，提前给出更清楚的业务错误。
    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")
    if model_factory is None:
        # 延迟导入避免执行prepare/check时也加载PyTorch和Transformer依赖。
        from sentence_transformers import SentenceTransformer
        model_factory = SentenceTransformer
    # 项目要求本地可运行，因此权重必须提前放进models目录。
    model_path = project_root / "models" / model_name
    if not model_path.is_dir():
        raise RuntimeError(f"找不到本地向量模型：{model_path}")
    # device默认cpu保证无显卡也能跑；local_files_only禁止运行时偷偷访问网络。
    model = model_factory(str(model_path), device=os.environ.get("RAG_EMBEDDING_DEVICE", "cpu"),
                          local_files_only=True)
    results = []
    for path in files:
        # 跳过空行并逐行解析JSON；这样坏数据可以定位到具体输入文件。
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        # 每条记录必须有非空字符串text，否则无法形成有意义的向量。
        if not records or any(not isinstance(row.get("text"), str) or not row["text"].strip()
                              for row in records):
            raise ValueError(f"知识块文件为空或格式错误：{path.name}")
        # normalize_embeddings=True生成单位向量，使Milvus余弦相似度可直接比较。
        vectors = model.encode([row["text"] for row in records], batch_size=batch_size,
                               normalize_embeddings=True, show_progress_bar=True)
        # 向量数必须与输入知识块数相等，否则zip会静默截断并丢数据。
        if len(vectors) != len(records):
            raise RuntimeError(f"向量模型返回数量不正确：{path.name}")
        # output复制原记录，并补充可JSON序列化的向量和模型溯源信息。
        output = []
        for record, vector in zip(records, vectors):
            # dict复制避免直接修改刚读取的records，便于调试原始输入。
            item = dict(record)
            # numpy数组用tolist；测试假模型若返回普通序列则用list。
            item["vector"] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            item["embedding_model"] = model_name
            output.append(item)
        # 空向量无法建立Milvus FLOAT_VECTOR字段，必须在落盘前中止。
        if not output[0]["vector"]:
            raise RuntimeError(f"向量模型返回数量不正确：{path.name}")
        # 输出文件沿用输入文件名，保持“一份文档一份快照”的对应关系。
        _write_jsonl(output_dir / path.name, output)
        # dimension读取第一条向量长度；后续同步还会逐条检查维度一致性。
        result = {"source": path.name, "records": len(output),
                  "dimension": len(output[0]["vector"]), "model": model_name}
        results.append(result)
        print(f"EMBEDDED: {path.name} 块={len(output)} 维度={result['dimension']} 模型={model_name}")
    # 清单只列出本轮输入对应文件，为安全删除旧Milvus数据提供完整性证明。
    manifest = _embedding_manifest(output_dir, [path.name for path in files])
    print(f"SNAPSHOT_MANIFEST: {manifest}")
    return results


def _record_key(record: dict) -> tuple[str, int]:
    """从记录提取稳定业务键 ``(source, chunk_index)`` 并校验格式。

    ``record`` 可以来自JSONL或Milvus；早期文件使用 ``index``，数据库使用
    ``chunk_index``，本函数同时兼容。返回值用于更新时找到“同一文档的同一块”。
    """
    # source标识知识来自哪份文档，不使用会随导入变化的数据库ID作为业务身份。
    source = record.get("source")
    # 优先读数据库字段chunk_index；缺失时兼容prepare阶段的index。
    index = record.get("chunk_index", record.get("index"))
    # 来源必须是非空字符串，否则不同文档的块无法稳定区分。
    if not isinstance(source, str) or not source.strip():
        raise ValueError("每条向量记录都必须包含非空source")
    # bool在Python里属于int子类，所以先排除True/False，再检查非负整数。
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("每条向量记录都必须包含非负整数chunk_index或index")
    # strip统一来源两端空格，保证更新时键值稳定。
    return source.strip(), index


def _manifest_files(directory: Path, manifest_path: Path,
                    require_exact: bool) -> tuple[list[Path], list[str]]:
    """验证向量快照清单，返回可信文件及未列入清单的旧文件。

    参数：
        directory: JSONL向量文件所在目录。
        manifest_path: embed阶段生成的snapshot-manifest.json。
        require_exact: prune时为True，要求目录文件与清单完全一致。
    返回：
        ``(清单内文件路径列表, 未列入清单的JSONL文件名列表)``。
    """
    try:
        # 清单必须能以UTF-8读取并解析为JSON。
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取完整快照清单：{manifest_path}") from error
    # 仅当顶层是字典时才尝试读取files，避免对错误类型调用get。
    entries = payload.get("files") if isinstance(payload, dict) else None
    # 版本、文件数组和非空性共同证明它是当前程序认识的完整清单格式。
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("完整快照清单格式错误")
    # files保留已验证路径；names用于重复检查和目录集合比较。
    files, names = [], []
    for entry in entries:
        # 防御性读取：非法条目不会触发AttributeError，而会落入统一业务错误。
        name = entry.get("name") if isinstance(entry, dict) else None
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        # Path(name).name==name禁止../越界路径；正则要求标准64位小写SHA-256。
        if (not isinstance(name, str) or Path(name).name != name or
                not name.endswith(".jsonl") or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash))):
            raise ValueError("完整快照清单包含非法文件名或SHA256")
        # 将清单文件名限定到指定embeddings目录内。
        path = directory / name
        # 文件必须存在且内容哈希与embed时一致，防止同步被截断或篡改的快照。
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"完整快照文件缺失或SHA256不匹配：{name}")
        # 重复文件会掩盖“完整快照”的真实数量，因此明确拒绝。
        if name in names:
            raise ValueError("完整快照清单包含重复文件")
        names.append(name)
        files.append(path)
    # 扫描目录现有JSONL，用集合差找出未参与本轮快照的历史文件。
    actual_names = sorted(path.name for path in directory.glob("*.jsonl"))
    unlisted = sorted(set(actual_names) - set(names))
    # 删除模式要求两边集合完全相同，否则残缺清单可能误删其他文档的数据库记录。
    if require_exact and (sorted(names) != actual_names):
        raise ValueError("完整快照清单与embeddings目录不完全一致，禁止删除旧知识")
    return files, unlisted


def _load_snapshot(directory: Path, files: list[Path] | None = None) -> tuple[list[dict], int, list[str]]:
    """读取并严格校验向量快照。

    ``directory`` 是默认扫描目录；``files`` 非None时只读已由清单验证的文件。
    返回 ``(标准化记录列表, 向量维度, 文件名列表)``。所有记录在连接Milvus前完成
    校验，因此坏快照不会造成数据库只更新一半。
    """
    # 调用方显式给files时尊重清单范围，否则读取目录内全部JSONL；统一排序。
    files = sorted(files if files is not None else directory.glob("*.jsonl"))
    if not files:
        raise RuntimeError("向量快照为空，请先执行embed")
    # seen检测业务键重复；dimension由第一条向量确定并用于检查其余记录。
    records, seen, dimension = [], set(), None
    for path in files:
        # 保留原始行号，错误信息能精确定位到“文件:行”。
        lines = path.read_text(encoding="utf-8").splitlines()
        # 空文件不能代表一份完整文档快照，防止prune误认为文档应被清空。
        if not any(line.strip() for line in lines):
            raise RuntimeError("向量快照包含空JSONL文件")
        for line_number, line in enumerate(lines, 1):
            # 允许文件中偶尔有空行，不把它当作知识记录。
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL格式错误：{path.name}:{line_number}") from error
            # JSON数组、字符串等即使语法合法，也不符合知识块对象结构。
            if not isinstance(record, dict):
                raise ValueError(f"JSONL记录必须是对象：{path.name}:{line_number}")
            # 一次提取并复用业务键、正文和向量，减少后续重复查字典。
            key, text, vector = _record_key(record), record.get("text"), record.get("vector")
            # 同一来源同一块号只能有一条，否则无法判断哪条是最终版本。
            if key in seen:
                raise ValueError("向量快照中source与chunk_index组合重复")
            seen.add(key)
            # 空正文即使有向量也不能给用户提供引用依据。
            if not isinstance(text, str) or not text.strip():
                raise ValueError("每条向量记录都必须包含非空text")
            # 与Milvus VARCHAR schema限制一致，提前报错比数据库写入失败更清楚。
            if len(text) > 8000 or len(key[0]) > 256:
                raise ValueError("text或source超过教学版schema长度")
            # 向量必须是非空JSON数组，不能是字符串或单个数字。
            if not isinstance(vector, list) or not vector:
                raise ValueError("每条向量记录都必须包含非空vector数组")
            # 排除布尔值、非数字、NaN和无穷值，保证余弦距离可正常计算。
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in vector):
                raise ValueError("vector只能包含有限数值")
            # 第一条建立维度基准，例如BGE-small通常为512维。
            if dimension is None:
                dimension = len(vector)
            # Milvus一个FLOAT_VECTOR字段要求collection内所有向量维度相同。
            elif len(vector) != dimension:
                raise ValueError("快照内部向量维度不一致")
            # 复制记录并补充统一的chunk_index；旧record若含index字段仍会原样保留。
            normalized = dict(record)
            normalized["source"], normalized["chunk_index"] = key
            records.append(normalized)
    # 按来源和块号排序使写入计划、报告和测试结果稳定可复现。
    records.sort(key=lambda item: (item["source"], item["chunk_index"]))
    # 教学版查询验证最多安全覆盖1999条，超出时拒绝不完整验证。
    if len(records) > CORE_MAX_ROWS:
        raise RuntimeError(f"知识块超过教学版安全上限{CORE_MAX_ROWS}")
    # 此处records必非空，所以dimension已经由第一条设置；int明确返回类型。
    return records, int(dimension), [path.name for path in files]


def _field_map(description: dict) -> dict[str, dict]:
    """把不同pymilvus版本的collection描述统一成“字段名→字段信息”。"""
    # 有的版本直接返回fields，有的嵌在schema.fields中；or依次选择可用结构。
    fields = description.get("fields") or description.get("schema", {}).get("fields") or []
    # 字段名也存在name和field_name两种写法，只保留确实有名字的字段。
    return {str(field.get("name") or field.get("field_name")): field for field in fields
            if field.get("name") or field.get("field_name")}


def _vector_dimension(fields: dict[str, dict]) -> int:
    """从Milvus字段描述中兼容读取vector字段的正整数维度。"""
    # vector不存在时用空字典，随后统一抛出“无法读取维度”。
    field = fields.get("vector", {})
    # 不同SDK把维度放在params或type_params中。
    params = field.get("params") or field.get("type_params") or {}
    # 先收集字段顶层可能使用的dim/dimension两种名称。
    values = [field.get("dim"), field.get("dimension")]
    if isinstance(params, dict):
        # 再补充嵌套参数中的两种名称，按顺序尝试。
        values.extend((params.get("dim"), params.get("dimension")))
    for value in values:
        try:
            # SDK可能返回字符串数字，因此先int转换并要求大于0。
            if value is not None and int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            # 当前候选无法转换时继续检查其他兼容位置。
            pass
    raise RuntimeError("无法读取现有collection的向量维度")


def _create_collection(client, dimension: int, data_type=None) -> None:
    """创建医生知识库collection及余弦向量索引。

    参数：
        client: 已连接的 ``MilvusClient`` 或测试假客户端。
        dimension: BGE输出向量的维度，必须与快照一致。
        data_type: 默认使用pymilvus.DataType；测试可注入假枚举以隔离外部服务。
    返回：
        无返回值；通过client在Milvus中创建 ``doctor_knowledge``。
    """
    if data_type is None:
        # 延迟导入使纯解析和纯向量化阶段不强制安装/连接Milvus。
        from pymilvus import DataType
        data_type = DataType
    # auto_id=False表示ID由本程序分配，更新时才能保持同一业务块ID稳定。
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    # id是Milvus主键；INT64既能排序也能用ids列表精确删除。
    schema.add_field("id", data_type.INT64, is_primary=True)
    # vector存BGE浮点向量，dim必须等于本次快照维度。
    schema.add_field("vector", data_type.FLOAT_VECTOR, dim=dimension)
    # text保存原文，在线回答会把它作为上下文和引用；长度与快照校验一致。
    schema.add_field("text", data_type.VARCHAR, max_length=8000)
    # source保存文档名，用于引用来源和稳定业务键。
    schema.add_field("source", data_type.VARCHAR, max_length=256)
    # chunk_index保存该文档内从0开始的块序号。
    schema.add_field("chunk_index", data_type.INT64)
    # created_at记录该业务块首次进入知识库的时间，后续更新尽量保留。
    schema.add_field("created_at", data_type.VARCHAR, max_length=40)
    # 创建独立索引参数对象，再为vector字段配置检索索引。
    indexes = client.prepare_index_params()
    # FLAT执行精确搜索，适合教学版小数据；COSINE与归一化BGE向量匹配。
    indexes.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    # 把固定名称、schema和索引一次提交给Milvus创建collection。
    client.create_collection(COLLECTION, schema=schema, index_params=indexes)


def _existing_rows(client, fields: dict[str, dict]) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    """读取现有Milvus记录，并建立业务键到旧记录的映射。

    ``client`` 是Milvus客户端，``fields`` 是现有schema字段映射；返回
    ``(去重后的旧记录列表, (source, chunk_index)到记录的字典)``。
    这一步使知识更新可以复用旧ID和created_at，而不是每次全部新增。
    """
    # row_count为0时无需执行查询，直接返回两个空容器。
    if int(client.get_collection_stats(COLLECTION).get("row_count", 0)) <= 0:
        return [], {}
    # 必查ID、正文、来源和块号；可选字段仅在旧schema实际存在时请求。
    output_fields = sorted((REQUIRED_FIELDS - {"vector"}) | (OPTIONAL_FIELDS & fields.keys()))
    # 更新规划不需要取大体积向量；id>=0覆盖本程序生成的全部非负主键。
    rows = client.query(COLLECTION, filter="id >= 0", output_fields=output_fields,
                        limit=QUERY_LIMIT)
    # 统计说非空而查询为空说明数据不可见或服务异常，此时不能安全更新。
    if not rows:
        raise RuntimeError("collection统计非空但查询不到记录")
    # 达到limit时无法证明已读完全部数据，也就不能正确找出旧ID和过期块。
    if len(rows) >= QUERY_LIMIT:
        raise RuntimeError("知识块达到查询上限，无法安全建立完整ID映射")
    # by_id检测主键冲突；by_key用于新快照匹配旧业务记录。
    by_id, by_key = {}, {}
    for row in rows:
        # Milvus返回值可能是数值兼容类型，int统一为Python整数。
        row_id, key = int(row["id"]), _record_key(row)
        # 同一数据库ID若对应两个不同业务键，说明现有数据已损坏。
        if row_id in by_id and _record_key(by_id[row_id]) != key:
            raise RuntimeError("现有collection同一ID对应多个业务键")
        # 同一业务键若对应两个ID，增量更新无法安全选择保留哪一个。
        if key in by_key and int(by_key[key]["id"]) != row_id:
            raise RuntimeError("现有collection存在重复source与chunk_index")
        # 正常记录同时放入两种索引；重复且完全相同的返回行会自然覆盖。
        by_id[row_id], by_key[key] = row, row
    # by_id.values去除可能的完全重复查询行；by_key留给_build_plan快速查找。
    return list(by_id.values()), by_key


def _build_plan(records: list[dict], existing: list[dict], by_key: dict,
                fields: dict[str, dict], now: str, prune: bool) -> tuple[list[dict], list[int], list[dict]]:
    """比较新快照与旧库，生成待写行、过期ID和可读更新明细。

    参数：
        records: 已校验的新向量记录。
        existing/by_key: 现有Milvus记录及业务键映射。
        fields: 现有或计划创建的schema字段。
        now: 本次同步共用的ISO时间，保证报告时间一致。
        prune: 是否按完整快照删除旧库中已不存在的块。
    返回：
        ``(待upsert记录, 过期ID列表, 报告明细列表)``。
    """
    # 新ID从当前最大ID加1开始；空库default=-1，因此第一条ID是0。
    next_id = max((int(row["id"]) for row in existing), default=-1) + 1
    # desired_ids表示新快照需要存在的ID；rows写数据库；manifest写更新报告。
    desired_ids, rows, manifest = set(), [], []
    for record in records:
        # 用来源+块号查旧记录；同一业务块更新时沿用其数据库ID。
        key, old = _record_key(record), by_key.get(_record_key(record))
        # old存在表示更新，不存在表示新增并分配next_id。
        row_id = int(old["id"]) if old else next_id
        if old is None:
            # 只在真正新增时递增，已有记录不会消耗新ID。
            next_id += 1
        # 记录最终应保留的ID集合，稍后与现有集合做差。
        desired_ids.add(row_id)
        # 只挑schema必需业务字段，避免把JSONL中的临时id等未知字段写入Milvus。
        row = {name: record[name] for name in ("text", "source", "chunk_index", "vector")}
        row["id"] = row_id
        # created_at字段只有schema支持时才写；旧值优先，体现“首次创建时间”。
        if "created_at" in fields:
            row["created_at"] = (old or {}).get("created_at") or record.get("created_at") or now
        # updated_at若存在，每次同步都使用快照提供值或本次时间。
        if "updated_at" in fields:
            row["updated_at"] = str(record.get("updated_at") or now)
        # summary是可选摘要；缺失时写空串以满足VARCHAR字段类型。
        if "summary" in fields:
            row["summary"] = str(record.get("summary") or "")
        rows.append(row)
        # 报告不保存大向量和整段正文，只记录可审阅的身份、动作和时间。
        item = {"id": row_id, "source": key[0], "chunk_index": key[1],
                "action": "updated" if old else "added", "updated_at": now}
        # 摘要与向量模型若存在，可用于追溯；它们不一定是Milvus schema字段。
        for name in ("summary", "embedding_model"):
            if name in record:
                item[name] = record[name]
        manifest.append(item)
    # 集合差得到“旧库有、当前快照没有”的ID，也就是潜在过期知识块。
    existing_ids = {int(row["id"]) for row in existing}
    stale = sorted(existing_ids - desired_ids)
    # prune/初始化时最终只剩新快照；普通更新会保留其他旧来源的数据。
    final_count = len(desired_ids) if prune else len(existing_ids | desired_ids)
    # 确保同步后仍可在QUERY_LIMIT内进行完整安全校验。
    if final_count > CORE_MAX_ROWS:
        raise RuntimeError(f"更新后将有{final_count}个知识块，超过安全上限{CORE_MAX_ROWS}")
    return rows, stale, manifest


def _verify_snapshot(client, expected: list[dict]) -> None:
    """回读Milvus并逐条核对业务键、ID和正文。

    ``client`` 是已连接的Milvus客户端；``expected`` 是需要继续保留的完整行。
    函数不返回值。向量不回读以降低传输量，但正文必须核对，防止只验证ID存在却
    实际保留了旧文本。失败时抛异常：删除前的第一次校验可阻止删除，删除后的第二次
    校验用于报告写入失败，不能撤销已经完成的Milvus删除。
    """
    # 回读所有教学版记录，只请求验证所需的小字段。
    visible = client.query(COLLECTION, filter="id >= 0",
                           output_fields=["id", "source", "chunk_index", "text"],
                           limit=QUERY_LIMIT)
    # 以业务键索引实际值，value同时包含数据库ID和原文。
    actual = {(row["source"], int(row["chunk_index"])): (int(row["id"]), row.get("text"))
              for row in visible}
    # 任何缺失、ID不符或正文不符都会进入missing列表。
    missing = [row for row in expected
               if actual.get((row["source"], int(row["chunk_index"]))) !=
               (int(row["id"]), row["text"])]
    if missing:
        # 明确说明不会进入后续删除阶段，保护已有知识。
        raise RuntimeError("Milvus写入校验失败，未执行后续删除")


def sync_milvus(project_root: Path = BASE, embeddings_dir: Path | None = None,
                 report_path: Path | None = None, *, prune: bool = False,
                 dry_run: bool = False, reset: bool = False, client_factory=None,
                 data_type=None, snapshot_manifest: Path | None = None) -> dict:
    """首次创建或按来源+块号稳定更新Milvus；删除必须显式指定。

    参数：
        project_root: 项目根目录，用于定位配置、默认快照和默认报告。
        embeddings_dir: 向量JSONL目录；省略时使用 ``data/embeddings``。
        report_path: 更新报告路径；省略时写入 ``outputs/knowledge_update_report.json``。
        prune: 删除旧库中不在完整快照里的块；必须显式提供并验证清单。
        dry_run: 只生成更新计划和报告，不修改Milvus。
        reset: 删除同名collection再重建，会清空旧数据。
        client_factory/data_type: 生产环境使用pymilvus默认值，单元测试可注入假实现。
        snapshot_manifest: embed阶段生成的完整快照清单路径。
    返回：
        包含同步状态、快照摘要、增删改数量及每条动作的报告字典。

    星号 ``*`` 表示它后面的危险开关必须写成关键字参数，防止位置传参误触删除。
    """
    # 无论调用者传字符串还是Path，都转换成Path以统一路径拼接行为。
    project_root = Path(project_root)
    # or让None使用项目默认目录，同时允许测试传入隔离的临时目录。
    embeddings_dir = Path(embeddings_dir or project_root / "data" / "embeddings")
    report_path = Path(report_path or project_root / "outputs" / "knowledge_update_report.json")
    # Milvus URI等配置在创建客户端前加载。
    load_env(project_root)
    # 记录用户是否“明确”提供清单；prune不能只依赖目录里碰巧存在的默认文件。
    explicit_manifest = Path(snapshot_manifest) if snapshot_manifest else None
    # 普通同步可自动使用embed生成的默认清单来限定可信文件。
    default_manifest = embeddings_dir / "snapshot-manifest.json"
    # 删除是不可逆操作，必须同时出现--prune和显式--snapshot-manifest。
    if prune and explicit_manifest is None:
        raise ValueError("--prune必须同时提供--snapshot-manifest完整快照清单")
    # 显式路径优先；否则仅在默认清单确实存在时使用，缺失则读取全部JSONL。
    manifest_path = explicit_manifest or (default_manifest if default_manifest.is_file() else None)
    # 没有清单时传None让_load_snapshot自行扫描；stale_artifacts默认无未列文件。
    snapshot_files, stale_artifacts = (None, [])
    if manifest_path:
        # prune要求目录与清单完全一致；普通更新则允许历史文件保留但不导入。
        snapshot_files, stale_artifacts = _manifest_files(
            embeddings_dir, manifest_path, require_exact=prune
        )
    # 在连接数据库前完整读取并校验快照，得到记录、统一维度和文件名。
    records, snapshot_dimension, files = _load_snapshot(embeddings_dir, snapshot_files)
    if client_factory is None:
        # 延迟导入使PDF处理和单元测试不必启动Milvus依赖。
        from pymilvus import MilvusClient
        client_factory = MilvusClient
    # URI默认指向本机Docker Milvus；client_factory注入使测试不访问真实服务。
    client = client_factory(uri=os.environ.get("RAG_MILVUS_URI", "http://127.0.0.1:19530"))
    try:
        # exists决定本次是首次初始化还是增量更新。
        exists = client.has_collection(COLLECTION)
        # reset即使旧库存在，也按空库规划；后面真正写入前才执行drop。
        initialized = not exists or reset
        if initialized:
            # 新schema固定包含必需字段和created_at，供_build_plan构造写入行。
            fields = {name: {} for name in REQUIRED_FIELDS | {"created_at"}}
            existing, by_key = [], {}
        else:
            # 增量更新必须先了解现有schema，兼容不同历史版本。
            fields = _field_map(client.describe_collection(COLLECTION))
            # 缺任一必需字段都无法支持在线检索或稳定更新，直接中止。
            missing_fields = REQUIRED_FIELDS - fields.keys()
            if missing_fields:
                raise RuntimeError(f"现有collection缺少字段：{','.join(sorted(missing_fields))}")
            # 新快照和旧collection维度不同不能upsert到同一FLOAT_VECTOR字段。
            current_dimension = _vector_dimension(fields)
            if snapshot_dimension != current_dimension:
                raise RuntimeError(
                    f"向量维度不匹配：快照为{snapshot_dimension}维，现有collection为{current_dimension}维"
                )
            # 读取旧记录，用业务键建立稳定ID映射和过期记录集合。
            existing, by_key = _existing_rows(client, fields)
        # 本次全部记录共用一个精确到秒且带时区的时间戳。
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        # 初始化等价于“新快照就是全库”；普通更新是否删除由prune决定。
        rows, stale, manifest = _build_plan(records, existing, by_key, fields, now,
                                             prune or initialized)
        # 状态优先报告dry_run，其次区分重建、首次初始化和普通更新。
        status = "dry_run" if dry_run else ("reset" if reset and exists else
                                             "initialized" if not exists else "updated")
        # report既给程序读取，也作为答辩中的知识库动态更新证据。
        report = {
            "status": status, "collection": COLLECTION, "created_at": now,
            # snapshot说明本次数据来源、规模、维度和完整性验证情况。
            "snapshot": {"files": files, "records": len(records),
                          "dimension": snapshot_dimension,
                          "manifest": str(manifest_path) if manifest_path else None,
                          "unlisted_embedding_files_retained": stale_artifacts,
                          "validated_complete_for_prune": bool(prune and explicit_manifest)},
            # added/updated来自业务键匹配；stale默认保留，只有prune才真正删除。
            "changes": {"added": sum(item["action"] == "added" for item in manifest),
                        "updated": sum(item["action"] == "updated" for item in manifest),
                        "stale_retained": 0 if prune or initialized else len(stale),
                        "pruned": len(stale) if prune and not dry_run else 0,
                        "planned_prune": len(stale) if prune else 0},
            # 旧schema没有可选字段时，元数据仍进入报告，不冒险修改在线库schema。
            "optional_metadata_storage": {
                name: "collection_and_report" if name in fields else "report_only"
                for name in ("summary", "updated_at")
            },
            # 在线应用启动时加载索引和模型，因此知识更新后提示重启服务。
            "restart_required": True, "items": manifest, "stale_ids": stale,
        }
        # dry_run到此只写报告，下面所有Milvus变更都被跳过。
        if not dry_run:
            # reset显式要求先删旧collection；更新exists变量供随后创建判断使用。
            if reset and exists:
                client.drop_collection(COLLECTION)
                exists = False
            # 首次运行或reset后，根据快照真实维度创建schema和索引。
            if not exists:
                _create_collection(client, snapshot_dimension, data_type)
            # upsert按主键实现“存在则更新、不存在则新增”。
            client.upsert(COLLECTION, rows)
            # flush等待数据持久化，load_collection使其可立即查询验证。
            client.flush(COLLECTION)
            client.load_collection(COLLECTION)
            # 先确认所有新正文已正确可见，再考虑删除任何旧知识。
            _verify_snapshot(client, rows)
            if prune and stale:
                # 只有显式prune、完整清单已验证且确有旧ID时才执行删除。
                client.delete(collection_name=COLLECTION, ids=stale)
                client.flush(COLLECTION)
                client.load_collection(COLLECTION)
                # 删除后再次确认新快照仍完整，防止删除条件误伤目标行。
                _verify_snapshot(client, rows)
                # 最后只回读ID，检查任何stale ID是否仍残留。
                remaining = client.query(COLLECTION, filter="id >= 0", output_fields=["id"],
                                         limit=QUERY_LIMIT)
                if {int(row["id"]) for row in remaining} & set(stale):
                    raise RuntimeError("过期知识块删除校验失败")
        # 报告最后写入；使用原子写避免下次读取半份同步结果。
        _write_json(report_path, report)
        return report
    finally:
        # 成功或任意异常都关闭客户端连接，避免资源泄漏和测试相互影响。
        client.close()


def _parser_status() -> dict[str, bool]:
    """返回四类解析能力的就绪状态，键是工具名、值是布尔值。"""
    # MinerU既可能是Python包，也可能是独立命令，因此使用专门的命令探测函数。
    return {
        "pymupdf": _installed("pymupdf"),
        "pdfplumber": _installed("pdfplumber"),
        "paddleocr": _installed("paddleocr"),
        "mineru": bool(_mineru_command()),
    }


def main() -> None:
    """解析命令行参数，并把四个子命令分发到对应离线阶段。"""
    # description会显示在python -m app.offline_pipeline --help顶部。
    parser = argparse.ArgumentParser(description="医学RAG离线知识库流水线")
    # dest把子命令名保存到args.command；required=True要求用户必须选择一个阶段。
    commands = parser.add_subparsers(dest="command", required=True)
    # check只检查环境，不读取或修改知识库。
    check = commands.add_parser("check", help="检查PDF解析器")
    # store_true表示参数出现时为True；部署脚本用它要求四种解析能力全部就绪。
    check.add_argument("--require-all", action="store_true", help="缺少任一解析器时返回失败")
    # prepare负责PDF解析、文本清洗和知识分块。
    prepare = commands.add_parser("prepare", help="PDF解析、清洗并分块")
    # type=Path让argparse直接把字符串路径转成Path；不传则批量处理data/raw。
    prepare.add_argument("--input", type=Path, help="单个PDF；省略则处理data/raw全部PDF")
    # choices在进入业务逻辑前拒绝非法引擎名；auto是普通资料默认选项。
    prepare.add_argument("--engine", choices=["auto", "pymupdf", "paddleocr", "mineru"],
                         default="auto")
    # 复杂版式开关只在auto引擎下触发MinerU。
    prepare.add_argument("--complex-layout", action="store_true")
    # --tables额外运行PDFPlumber，把表格文本追加到正文。
    prepare.add_argument("--tables", action="store_true")
    # 两个整数参数控制目标块长和最小块长，默认值来自文件顶部常量。
    prepare.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    prepare.add_argument("--min-size", type=int, default=MIN_CHUNK_SIZE)
    # embed读取prepare生成的JSONL，并使用本地BGE输出向量快照。
    embed = commands.add_parser("embed", help="用本地BGE生成向量")
    # --model是models目录下的文件夹名；None时交给环境变量/默认值选择。
    embed.add_argument("--model", default=None)
    # batch-size控制一次推理条数，不改变向量结果，只影响速度和内存。
    embed.add_argument("--batch-size", type=int, default=16)
    # sync负责把完整向量记录首次创建或增量写入Milvus。
    sync = commands.add_parser("sync", help="首次创建或安全更新Milvus")
    # 自定义目录和报告路径主要用于测试、迁移或多套环境。
    sync.add_argument("--embeddings-dir", type=Path)
    sync.add_argument("--report", type=Path)
    # dry-run只展示计划，适合正式更新前检查added/updated/pruned数量。
    sync.add_argument("--dry-run", action="store_true")
    # prune是危险开关：删除完整快照中不存在的旧数据库块。
    sync.add_argument("--prune", action="store_true", help="删除完整快照中已不存在的旧块")
    # 清单必须来自embed且每个SHA-256匹配，作为prune的完整性门禁。
    sync.add_argument("--snapshot-manifest", type=Path,
                      help="--prune必填：由embed生成且SHA256完全匹配的完整快照清单")
    # reset删除并重建整个collection，适合schema或向量维度变化后的显式重建。
    sync.add_argument("--reset", action="store_true", help="删除并重建collection，旧知识会被清空")
    # parse_args读取sys.argv，并生成带上述字段的Namespace对象。
    args = parser.parse_args()
    # check也要读取.env.local中的MINERU_COMMAND，所以分支前统一加载配置。
    load_env(BASE)
    if args.command == "check":
        # status形如{"pymupdf": true, ...}，JSON输出方便部署脚本读取。
        status = _parser_status()
        print(json.dumps(status, ensure_ascii=False))
        # require-all下任一False都让argparse以非零状态结束，阻止错误部署继续。
        if args.require_all and not all(status.values()):
            missing = "、".join(name for name, ready in status.items() if not ready)
            parser.error(f"解析器未就绪：{missing}")
    elif args.command == "prepare":
        # 参数顺序对应根目录、输入、引擎、复杂版式、表格、目标块长、最小块长。
        _prepare_documents(BASE, args.input, args.engine, args.complex_layout, args.tables,
                           args.chunk_size, args.min_size)
    elif args.command == "embed":
        # CLI只暴露模型名和批大小，输入输出使用项目标准目录。
        embed_chunks(BASE, args.model, args.batch_size)
    else:
        # 剩余必然是sync；危险开关使用关键字传参，降低顺序传错风险。
        result = sync_milvus(BASE, args.embeddings_dir, args.report, prune=args.prune,
                             dry_run=args.dry_run, reset=args.reset,
                             snapshot_manifest=args.snapshot_manifest)
        # 从报告提取计数，打印一行适合部署日志查看的同步摘要。
        changes = result["changes"]
        print(f"MILVUS_{result['status'].upper()}: added={changes['added']} "
              f"updated={changes['updated']} retained={changes['stale_retained']} "
              f"pruned={changes['pruned']} restart_required=true")


if __name__ == "__main__":
    # 直接运行或python -m调用时进入CLI；被测试import时不会自动执行命令。
    main()
