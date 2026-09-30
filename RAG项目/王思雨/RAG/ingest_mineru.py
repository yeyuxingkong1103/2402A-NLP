# -*- coding: utf-8 -*-
"""MinerU 解析模块（第 12 步新增）：真实调用 MinerU 并与 PyMuPDF 对比。

**版本说明**：本机装的是 **mineru 3.4.5**（不是提示词里说的 2.x），因此下面的 API
按 3.4.5 的实际情况写。3.4.5 里 `mineru.cli.common.do_parse` 仍然存在，
签名与 2.x 基本一致（多了 backend / effort / image_analysis 等参数），
所以调用方式变化不大，但**模型来源环境变量只有 `MINERU_MODEL_SOURCE`**——
3.4.5 全包搜不到 `MINERU_MODEL_DIR`，模型实际路径由 `~/mineru.json` 的
`models-dir` 决定（本机指向 `C:\\mineru_models`）。

**两个实跑踩出来的坑**（都在下面用代码绕开了）：

1. **缺 torchvision**：pipeline 后端要 `torchvision.transforms.v2`。
   直接 `pip install torchvision` 会把 **torch 2.13.0 升到 2.14.0**，
   而 torch 是 BGE-m3 与 reranker 的地基，升级可能弄坏整条 RAG 链路。
   实测 `torchvision==0.28.0` 只装自己、不动 torch，已按此版本安装。
2. **中文路径**：`fast_langdetect` 的原生 FastText 加载器打不开含中文的路径
   （本机用户目录是 `C:\\Users\\雨子\\`）。实测同一份模型文件放在
   `C:/fasttext_models/` 能加载、放在 `D:/模型/` 就失败——**必须纯 ASCII 路径**。
   这与项目里 PaddleOCR 需要 `OCR_MODEL_DIR` 是同一类问题。
   下面的 `_fix_fasttext_path()` 把它的模型路径常量在运行时指到 ASCII 副本。
"""

import json                                    # 写对比结果 JSON
import shutil                                  # 复制 FastText 模型到 ASCII 路径
import time                                    # 统计解析耗时
from pathlib import Path                       # 路径处理

import config                                  # 读 PDF 目录等配置
from logger import get_logger                  # 日志工具

logger = get_logger("ingest_mineru")           # 创建本模块的 logger 实例

BASE_DIR = Path(__file__).resolve().parent             # 项目根目录
MINERU_OUTPUT_DIR = BASE_DIR / "data" / "mineru_output"   # MinerU 输出目录
COMPARE_JSON = BASE_DIR / "data" / "mineru_compare.json"  # 对比结果文件
FASTTEXT_ASCII_DIR = Path("C:/fasttext_models")        # FastText 模型的纯 ASCII 副本目录
FASTTEXT_FILE = "lid.176.ftz"                          # fast_langdetect 用的小模型文件名


def _fix_fasttext_path() -> str:
    """把 fast_langdetect 的模型路径指到纯 ASCII 副本，绕开中文路径打不开的问题。

    返回最终生效的路径字符串；加载器本来就指向 ASCII 路径时直接返回。
    """
    import fast_langdetect.ft_detect.infer as fli          # 导入其内部模块以改常量
    current = Path(fli.LOCAL_SMALL_MODEL_PATH)             # 当前生效的模型路径
    if str(current).isascii():                             # 已经是纯 ASCII
        return str(current)                                # 无需处理
    FASTTEXT_ASCII_DIR.mkdir(parents=True, exist_ok=True)  # 确保 ASCII 目录存在
    target = FASTTEXT_ASCII_DIR / FASTTEXT_FILE            # 目标文件路径
    if not target.exists():                                # 副本不存在才复制
        shutil.copy(current, target)                       # 从原位置复制一份
    fli.LOCAL_SMALL_MODEL_PATH = target                    # 改掉模块级常量（只影响本进程）
    fli._model_cache = type(fli._model_cache)() if hasattr(fli, "_model_cache") else None   # 清缓存
    logger.info("FastText 模型路径已改到纯 ASCII 路径：%s", target)   # 记录一次
    return str(target)                                     # 返回生效路径


def parse_with_mineru(pdf_path: str, max_pages: int = None) -> dict:
    """用 MinerU 把 PDF 解析成 Markdown，失败时返回 ok=False，绝不抛异常。

    :param pdf_path: PDF 文件路径
    :param max_pages: 只解析前 N 页（None 表示全部），调试时用来省时间
    :return: 成功 ``{"ok": True, "markdown", "pages", "elapsed", "output_dir"}``；
             失败 ``{"ok": False, "reason": ...}``
    """
    pdf = Path(pdf_path)                                   # 转成 Path 便于处理
    if not pdf.exists():                                   # 文件不存在
        return {"ok": False, "reason": f"PDF 不存在：{pdf}"}   # 直接返回失败
    try:                                                   # 导入与解析都要兜住异常
        from mineru.cli.common import do_parse             # MinerU 3.x 的解析入口
    except Exception as exc:                               # 环境没装好
        logger.warning("MinerU 不可用：%s", exc)            # 记录告警
        return {"ok": False, "reason": f"MinerU 未安装或依赖不全：{exc}"}   # 返回失败原因

    _fix_fasttext_path()                                   # 先修 FastText 路径，否则解析必失败

    try:                                                   # 开始解析
        pages = _pdf_page_count(pdf)                        # 先量一下总页数
        end_page = None if max_pages is None else min(max_pages - 1, pages - 1)   # 结束页（0 基）
        MINERU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # 输出目录不存在就创建
        start = time.time()                                # 计时起点
        do_parse(                                          # 调用 MinerU（3.4.5 签名）
            output_dir=str(MINERU_OUTPUT_DIR),             # 输出目录
            pdf_file_names=[pdf.stem],                     # 待解析文件名
            pdf_bytes_list=[pdf.read_bytes()],              # 文件内容
            p_lang_list=["ch"],                            # 文档语言：中文
            backend="pipeline",                            # 传统 pipeline 后端：模型小，8G 显存够
            parse_method="auto",                           # 自动判断走文本层还是 OCR
            formula_enable=False,                          # 标准文档没有公式，关掉省时间
            table_enable=True,                             # 开启表格识别
            f_dump_md=True,                                # 产出 markdown（MinerU 的强项）
            f_dump_content_list=True,                      # 产出结构化内容列表
            start_page_id=0,                               # 从第 1 页开始
            end_page_id=end_page,                          # 到指定页为止
        )                                                  # 解析调用结束
        elapsed = round(time.time() - start, 2)             # 本次解析耗时（秒）
        md_file = MINERU_OUTPUT_DIR / pdf.stem / "auto" / f"{pdf.stem}.md"   # markdown 产物路径
        markdown = md_file.read_text(encoding="utf-8") if md_file.exists() else ""   # 读 markdown
        parsed_pages = pages if max_pages is None else min(max_pages, pages)   # 实际解析页数
        logger.info("MinerU 解析完成：%s，%d 页，耗时 %.2f 秒",
                    pdf.name, parsed_pages, elapsed)        # 记录结果
        return {"ok": True, "markdown": markdown, "pages": parsed_pages,   # 页数
                "elapsed": elapsed, "output_dir": str(MINERU_OUTPUT_DIR)}   # 耗时与输出目录
    except Exception as exc:                                # 解析过程出错
        logger.warning("MinerU 解析失败：%s", exc)           # 记录告警
        return {"ok": False, "reason": f"MinerU 解析失败：{exc}"}   # 返回失败原因


def _pdf_page_count(pdf: Path) -> int:
    """用 PyMuPDF 取 PDF 总页数；取不到时返回 0。"""
    try:                                                   # 只是量页数，失败不该中断
        import fitz                                        # PyMuPDF
        with fitz.open(str(pdf)) as doc:                   # 打开文档
            return doc.page_count                          # 返回页数
    except Exception as exc:                               # 读取失败
        logger.warning("读取 PDF 页数失败：%s", exc)         # 记录告警
        return 0                                           # 返回 0


def _md_table_count(markdown: str) -> int:
    """统计 MinerU 产出里的表格数量。

    **实测注意**：MinerU 3.4.5 的 pipeline 后端把表格输出成 **HTML `<table>` 块**
    （如 `<table><tr><td>...</td></tr></table>`），**不是** markdown 管道表。
    一开始按 `|` 数，5 个 PDF 全返回 0，是计数方式错了。
    这里两种形式都数：`<table` 标签优先，同时把管道表按连续行分组计入。
    """
    html_tables = markdown.count("<table")                 # HTML 形式的表格数
    pipe_tables, in_table = 0, False                       # 管道表计数与"当前是否在表内"
    for line in markdown.splitlines():                     # 逐行扫描管道表
        stripped = line.strip()                            # 去掉两侧空白
        is_row = stripped.startswith("|") and stripped.endswith("|")   # 是否是管道表行
        if is_row and not in_table:                        # 进入一张新表
            pipe_tables += 1                               # 表数加一
            in_table = True                                # 标记已在表内
        elif not is_row:                                   # 遇到非表行
            in_table = False                               # 离开表
    return html_tables + pipe_tables                       # 返回两种形式之和


def _normalize(text: str) -> str:
    """归一化文本用于比对：去掉空白、markdown 标记与常见标点，便于判断"内容是否出现过"。"""
    import re                                          # 局部导入，避免模块顶层多一个依赖
    text = re.sub(r"[#*`>|\-\[\]()<>]", "", text)      # 去掉 markdown 标记
    text = re.sub(r"\s+", "", text)                    # 去掉所有空白
    return text                                        # 返回归一化结果


def compare_mineru_vs_pymupdf(pdf_path: str) -> dict:
    """把一个 PDF 分别用 MinerU 与 PyMuPDF 解析一遍并对比。

    独立对比工具，**不参与主链路**，也不修改 `parse_pdf` 的行为。
    :return: 对比结果字典，字段见第 12 步规格
    """
    import ingest                                       # 延迟导入，避免与 ingest 循环依赖
    pdf = Path(pdf_path)                                # 转成 Path
    logger.info("对比解析：%s", pdf.name)                # 记录开始

    m = parse_with_mineru(str(pdf))                     # MinerU 解析（全部页）
    pages_pymupdf = ingest.parse_text_with_pymupdf(str(pdf))       # PyMuPDF 逐页文本
    pymupdf_text = "\n".join(p["text"] for p in pages_pymupdf)     # 拼成整篇文本
    try:                                                # 表格提取可能失败
        tables_pymupdf = len(ingest.parse_tables_with_pdfplumber(str(pdf)))   # pdfplumber 表格数
    except Exception as exc:                            # 提取失败
        logger.warning("pdfplumber 表格提取失败：%s", exc)   # 记录告警
        tables_pymupdf = 0                              # 记为 0

    if not m.get("ok"):                                 # MinerU 没跑成功
        return {"pdf": pdf.name, "mineru_ok": False, "reason": m.get("reason", ""),   # 如实记录失败
                "mineru_chars": 0, "pymupdf_chars": len(pymupdf_text),                # 字符数
                "mineru_tables": 0, "pymupdf_tables": tables_pymupdf,                  # 表格数
                "pages_mineru": 0, "pages_pymupdf": len(pages_pymupdf),                # 页数
                "mineru_only": ""}                                                     # 无独有内容

    markdown = m["markdown"]                            # MinerU 的 markdown 正文
    norm_pymupdf = _normalize(pymupdf_text)             # PyMuPDF 文本的归一化形式
    # 找出 MinerU 有、PyMuPDF 里找不到的整行内容（前若干条，拼成 200 字样例）
    only_lines = []                                     # 收集 MinerU 独有行
    for line in markdown.splitlines():                  # 逐行比对
        key = _normalize(line)                          # 归一化
        if len(key) < 8:                                # 太短的行（标题碎片、页码）不参与
            continue                                    # 跳过
        if key not in norm_pymupdf:                     # PyMuPDF 里找不到这一行
            only_lines.append(line.strip())             # 记为 MinerU 独有
        if len(only_lines) >= 5:                        # 取前 5 条即可
            break                                       # 够了就停
    mineru_only = "\n".join(only_lines)[:200]           # 拼成样例并截到 200 字

    result = {                                          # 组装对比结果
        "pdf": pdf.name,                                # 文件名
        "mineru_ok": True,                              # MinerU 是否成功
        "mineru_chars": len(markdown),                  # MinerU markdown 字符数
        "pymupdf_chars": len(pymupdf_text),             # PyMuPDF 文本字符数
        "mineru_tables": _md_table_count(markdown),     # MinerU 表格数（按 md 表行分组）
        "pymupdf_tables": tables_pymupdf,               # pdfplumber 表格数
        "pages_mineru": m["pages"],                     # MinerU 解析页数
        "pages_pymupdf": len(pages_pymupdf),            # PyMuPDF 页数
        "mineru_elapsed": m["elapsed"],                 # MinerU 耗时
        "mineru_only": mineru_only,                     # MinerU 独有内容样例
    }                                                   # 结果组装结束
    logger.info("对比完成：%s，MinerU %d 字符 / PyMuPDF %d 字符",
                pdf.name, result["mineru_chars"], result["pymupdf_chars"])   # 记录结果
    return result                                       # 返回对比结果


def compare_all(file_names: tuple = None) -> list:
    """对语料里的 PDF 逐个做对比，返回结果列表。

    默认只处理 `ingest.TARGET_PDF_NAMES` 里登记的 5 个 GB/T 标准，
    **不扫描整个 PDF 目录**——PDF_DIR 下还混着与语料无关的文档（讲义、论文等），
    全扫会白跑一堆。
    """
    import ingest                                       # 延迟导入，取目标文件名单
    pdf_dir = Path(config.PDF_DIR)                      # PDF 目录来自配置
    names = list(file_names) if file_names else list(ingest.TARGET_PDF_NAMES)   # 默认只做 5 个标准
    results = []                                        # 结果收集
    for i, name in enumerate(names, 1):                 # 逐个处理
        logger.info("对比进度 %d/%d：%s", i, len(names), name)   # 打印进度
        results.append(compare_mineru_vs_pymupdf(str(pdf_dir / name)))   # 对比并收集
    return results                                      # 返回全部结果


def run_compare_command() -> None:
    """命令行子流程：逐个对比 MinerU 与 PyMuPDF，打印对比表并落盘 JSON。

    从 `python -m ingest --compare-mineru` 调用，整个对比流程都收在本模块，
    这样 `ingest.py` 不必背着 MinerU 的依赖与输出逻辑。
    """
    start = time.time()                                # 记录开始时间
    print("正在用 MinerU 与 PyMuPDF 逐个对比，首次加载模型会慢一些……")   # 提示
    results = compare_all()                            # 逐个对比（失败也记在结果里，不抛异常）
    head = "{:<40}{:>6}{:>13}{:>12}{:>11}{:>10}"       # 表格各列宽度格式
    print("=" * 92)                                    # 分隔线
    print(head.format("PDF", "页数", "PyMuPDF字符", "MinerU字符", "PyMuPDF表", "MinerU表"))   # 表头
    print("-" * 92)                                    # 分隔线
    for r in results:                                  # 逐条打印对比行
        name = r["pdf"] if len(r["pdf"]) <= 38 else r["pdf"][:35] + "..."   # 长文件名截断
        print(head.format(name, r["pages_pymupdf"], r["pymupdf_chars"], r["mineru_chars"],   # 前四列
                          r["pymupdf_tables"], r["mineru_tables"]))         # 后两列
    print("-" * 92)                                    # 分隔线
    for r in results:                                  # 逐个打印 MinerU 独有内容样例
        print()                                        # 空一行
        print("[%s] MinerU 独有内容样例：" % r["pdf"])   # 标题
        for line in (r["mineru_only"] or "（无——PyMuPDF 已覆盖 MinerU 的全部内容）").splitlines():
            print("  " + line)                          # 逐行缩进打印
    COMPARE_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")   # 落盘
    print()                                            # 空一行
    print("对比结果已写入：%s" % COMPARE_JSON)           # 打印输出路径
    print("总耗时：%.1f 秒" % (time.time() - start))     # 打印耗时
