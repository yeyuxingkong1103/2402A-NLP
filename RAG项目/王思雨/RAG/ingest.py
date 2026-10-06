# -*- coding: utf-8 -*-
"""资料解析模块：PDF 解析（MinerU / PyMuPDF / pdfplumber / PaddleOCR）+ 语义切分，第 2 步不含入库。"""

import json                                   # 导入 json，用于把解析结果写成 JSON 文件
import sys                                    # 导入 sys，用于读命令行参数（--compare-mineru）
import time                                   # 导入 time，用于记录耗时与时间戳
from pathlib import Path                      # 导入 Path，用于路径拼接与目录创建

import pymupdf                                # 导入 PyMuPDF（即 fitz）用于提取 PDF 文本层
import pdfplumber                             # 导入 pdfplumber，用于提取 PDF 表格
import config                                 # 导入配置模块，PDF 目录与模型路径从这里读
from logger import get_logger                 # 导入日志工具，用于记录解析进度
from ingest_chunk import (                    # 从文本处理模块导入清洗与切分函数
    find_repeated_lines,                      # 跨页重复行识别函数
    remove_watermark_and_header_footer,       # 去水印与页眉页脚函数
    semantic_chunk,                           # 语义切分函数
)                                             # 导入结束

logger = get_logger("ingest")                 # 创建本模块的 logger 实例

PDF_SUFFIX = ".pdf"                           # 只处理扩展名为 pdf 的文件
DATA_DIR = Path(__file__).resolve().parent / "data"      # 解析结果输出目录
CHUNK_JSON = DATA_DIR / "parsed_chunks.json"  # 解析结果默认输出文件

ENABLE_MINERU = True                          # 是否启用 MinerU：第 12 步起模型已就绪，可调用对比
USE_MINERU_AS_MAIN = config.USE_MINERU_AS_MAIN   # 是否把 MinerU 当主链路：默认 False，主链路仍走 PyMuPDF
#                                             # （原因：MinerU 吃显存，解析 15 秒会涨到几分钟）
ENABLE_OCR = True                             # 是否启用 PaddleOCR：模型仅 18MB，已开启用于补扫描页
OCR_DPI = 200                                 # 扫描页渲染成图片时的分辨率
BLANK_CHECK_DPI = 80                          # 判断空白页时的低渲染分辨率，够用且快
BLANK_RATIO = 0.001                           # 深色像素占比低于此值判定为空白页
BLANK_SAMPLE_STEP = 397                       # 像素采样步长，避免逐像素统计过慢
CHUNK_MAX_LEN = 500                           # 单个语义块最大长度
CHUNK_MIN_LEN = 80                            # 单个语义块最小长度


def parse_with_mineru(pdf_path: str) -> dict:          # MinerU 解析入口（实现在 ingest_mineru.py）
    """MinerU 解析：开关关闭时返回未启用，开启时委托给 ingest_mineru 的真实实现。"""
    if not ENABLE_MINERU:                              # 开关关闭时
        return {"ok": False, "reason": "MinerU 开关未开启（ENABLE_MINERU=False）"}   # 返回未启用
    import ingest_mineru                               # 延迟导入：MinerU 依赖重，不拖慢主链路
    return ingest_mineru.parse_with_mineru(pdf_path)   # 委托给真实实现


def parse_text_with_pymupdf(pdf_path: str) -> list:    # PyMuPDF 文本提取入口
    """用 PyMuPDF 提取每一页的文本层，返回 [{page, text}]，页码从 1 开始。"""
    pages = []                                         # 保存每页文本
    with pymupdf.open(pdf_path) as doc:                 # 打开 PDF，退出时自动关闭
        for index in range(doc.page_count):             # 逐页遍历
            page = doc[index]                           # 取出当前页对象
            text = page.get_text()                      # 提取该页文本层
            pages.append({"page": index + 1, "text": text})   # 页码从 1 开始记录
    logger.info("PyMuPDF 提取完成：%s，共 %d 页", Path(pdf_path).name, len(pages))   # 记录日志
    return pages                                       # 返回逐页文本


def parse_tables_with_pdfplumber(pdf_path: str) -> list:   # pdfplumber 表格提取入口
    """用 pdfplumber 提取每一页的表格，返回 [{page, tables: [[行], ...]}]。"""
    result = []                                        # 保存每页表格
    try:                                               # 表格提取容易出错，做异常保护
        with pdfplumber.open(pdf_path) as pdf:          # 打开 PDF
            for index, page in enumerate(pdf.pages):    # 逐页遍历，index 从 0 开始
                tables = page.extract_tables()          # 提取该页所有表格
                cleaned = []                            # 保存清洗后的表格
                for table in tables:                    # 逐个表格处理
                    rows = [[(cell or "").strip() for cell in row] for row in table]   # 空格转空串
                    if rows:                            # 只保留有内容的表格
                        cleaned.append(rows)            # 收进结果
                if cleaned:                             # 该页确实有表格才记录
                    result.append({"page": index + 1, "tables": cleaned})   # 页码从 1 开始
        logger.info("pdfplumber 表格提取完成：%s，共 %d 页有表格",
                    Path(pdf_path).name, len(result))   # 记录日志
    except Exception as exc:                            # 提取失败时
        logger.warning("表格提取失败：%s", exc)          # 记录告警
    return result                                      # 返回表格结果


_OCR_ENGINE = None                                    # OCR 识别器单例缓存，避免每页重复加载模型


def _get_ocr_engine():                                # 内部函数：获取 OCR 识别器
    """懒加载 PaddleOCR 识别器并缓存为单例；模型加载一次约十几秒，不能每页都重建。"""
    global _OCR_ENGINE                                # 声明使用模块级变量
    if _OCR_ENGINE is not None:                       # 已经加载过
        return _OCR_ENGINE                            # 直接返回缓存
    from paddleocr import PaddleOCR                   # 延迟导入，避免未安装时影响模块加载
    root = Path(config.OCR_MODEL_DIR)                 # OCR 模型根目录（必须是纯英文路径）
    kwargs = {                                        # 组装创建参数
        "use_angle_cls": True,                        # 开启方向分类，纠正倒置文字
        "lang": "ch",                                 # 识别语言为中文
        "show_log": False,                            # 关闭 PaddleOCR 冗余日志
    }                                                 # 参数组装结束
    if (root / "det").exists():                       # 模型已放到纯英文路径时
        kwargs["det_model_dir"] = str(root / "det")   # 指定检测模型目录
        kwargs["rec_model_dir"] = str(root / "rec")   # 指定识别模型目录
        kwargs["cls_model_dir"] = str(root / "cls")   # 指定方向分类模型目录
    _OCR_ENGINE = PaddleOCR(**kwargs)                 # 创建识别器并缓存
    logger.info("PaddleOCR 模型加载完成，模型目录：%s", root)   # 记录日志
    return _OCR_ENGINE                                # 返回识别器


def ocr_with_paddleocr(image_path: str) -> str:       # PaddleOCR 识别入口
    """用 PaddleOCR 识别图片里的文字，环境不可用时返回空字符串，不抛异常。"""
    try:                                              # 尝试导入并识别
        engine = _get_ocr_engine()                    # 获取识别器单例
        outcome = engine.ocr(image_path, cls=True)    # 执行识别
        lines = []                                    # 保存识别出的文本行
        for page in outcome or []:                    # 遍历识别结果
            for item in page or []:                   # 遍历每个文字块
                lines.append(item[1][0])              # item[1][0] 是识别出的文字
        return "\n".join(lines)                       # 拼成多行文本返回
    except Exception as exc:                          # 未安装或识别失败
        logger.warning("PaddleOCR 不可用：%s", exc)    # 记录告警
        return ""                                     # 返回空字符串，不阻塞主流程


def _has_ink(page) -> bool:                            # 内部函数：判断页面是否画了东西
    """判断页面是否有内容：有图片、有矢量绘制，或渲染后存在足够多的深色像素。"""
    if page.get_images(full=True) or page.get_drawings():   # 有图片或有矢量图形
        return True                                    # 说明不是空白页
    samples = page.get_pixmap(dpi=BLANK_CHECK_DPI).samples  # 低分辨率渲染后取像素
    step = BLANK_SAMPLE_STEP                           # 采样步长，避免逐像素统计太慢
    total = len(samples) // step + 1                   # 采样点总数
    dark = sum(1 for i in range(0, len(samples), step) if samples[i] < 200)   # 深色采样点
    return dark / total >= BLANK_RATIO                 # 深色比例超过阈值才算有内容


def _ocr_pdf_page(pdf_path: str, page_no: int) -> str:  # 把无文本页渲染成图片再 OCR
    """把 PDF 某页渲染成 PNG 后交给 PaddleOCR；页面本身是空白页时直接返回空串。"""
    with pymupdf.open(pdf_path) as doc:                # 打开 PDF
        page = doc[page_no - 1]                        # 取出目标页
        if not _has_ink(page):                         # 该页是真正的空白页
            return ""                                  # 没有可识别内容，直接返回空串
        image_dir = DATA_DIR / "ocr_images"            # OCR 临时图片目录
        image_dir.mkdir(parents=True, exist_ok=True)   # 目录不存在就创建
        image_path = image_dir / f"{Path(pdf_path).stem}_p{page_no}.png"   # 拼出图片路径
        page.get_pixmap(dpi=OCR_DPI).save(str(image_path))   # 渲染并保存成 PNG
    return ocr_with_paddleocr(str(image_path))         # 交给 OCR 识别并返回文本


def clean_pages(pages: list) -> list:                  # 对逐页文本做清洗
    """对所有页文本做去水印与去页眉页脚处理，返回 [{page, text}]。"""
    raw_texts = [item["text"] for item in pages]       # 先取出每页原始文本
    repeated = find_repeated_lines(raw_texts)          # 统计跨页重复的页眉页脚行
    if repeated:                                       # 有重复行时记录一条日志
        logger.info("识别到 %d 条跨页重复页眉页脚", len(repeated))   # 记录数量
    cleaned = []                                       # 保存清洗后的结果
    for item in pages:                                 # 逐页清洗
        text = remove_watermark_and_header_footer(     # 调用清洗函数
            item["text"], item["page"], len(pages), repeated)   # 传入原文、页码、总页数、重复行
        cleaned.append({"page": item["page"], "text": text})    # 保存清洗结果
    return cleaned                                     # 返回清洗后的逐页文本


def build_chunks(source: str, cleaned_pages: list) -> list:   # 把页文本切成块
    """把清洗后的逐页文本做语义切分，返回带来源与页码的块列表。"""
    chunks = []                                        # 保存所有块
    pending = ""                                       # 上一页遗留的过短片段，拼到下一页开头
    for item in cleaned_pages:                         # 逐页处理
        text = pending + item["text"]                  # 把上一页遗留片段接到本页开头
        pending = ""                                   # 拼接后清空遗留缓冲
        pieces = semantic_chunk(text, CHUNK_MAX_LEN, CHUNK_MIN_LEN)   # 做语义切分
        if pieces and len(pieces[-1]) < CHUNK_MIN_LEN:  # 本页尾块过短，跨页切断不自然
            pending = pieces.pop()                     # 取出来留给下一页再合并
        for piece in pieces:                           # 逐个块记录
            chunks.append({                            # 每个块记录来源、页码与正文
                "chunk_id": len(chunks),               # 块序号，从 0 开始递增
                "source": source,                      # 文件名，供第 3 步写入 Milvus 的 source 字段
                "page": item["page"],                  # 来源页码
                "text": piece,                         # 块正文，供写入 Milvus 的 text 字段
                "length": len(piece),                  # 块长度，便于统计
            })                                         # 块字典结束
    if pending and chunks:                             # 全部页处理完仍有遗留片段
        merged = chunks[-1]["text"] + pending          # 尝试并回最后一个块
        if len(merged) <= CHUNK_MAX_LEN:               # 合并后不超长
            chunks[-1]["text"] = merged                # 更新正文
            chunks[-1]["length"] = len(merged)         # 更新长度
    return chunks                                      # 返回块列表


def parse_pdf(pdf_path: str) -> dict:                  # 单个 PDF 的完整解析入口
    """解析单个 PDF：PyMuPDF 取文本、pdfplumber 取表格、清洗后语义切分。"""
    path = Path(pdf_path)                              # 转成 Path 便于取文件名
    logger.info("开始解析：%s", path.name)              # 记录开始日志
    pages = parse_text_with_pymupdf(str(path))         # 第一步：提取逐页文本
    empty_pages = [p for p in pages if not p["text"].strip()]   # 找出没有文本层的页
    if empty_pages and ENABLE_OCR:                     # 有无文本页且 OCR 开关打开
        for item in empty_pages:                       # 逐页尝试 OCR
            item["text"] = _ocr_pdf_page(str(path), item["page"])   # 用识别结果填充该页文本
        still_empty = [p["page"] for p in empty_pages if not p["text"].strip()]   # 补不回来的页
        if still_empty:                                # 剩下的都是真空白页
            logger.info("%s 有 %d 页为空白页，OCR 无内容可识别：%s",
                        path.name, len(still_empty), still_empty)   # 记录说明
    elif empty_pages:                                  # 有无文本页但 OCR 未开启
        logger.info("%s 有 %d 页无文本层，OCR 未开启", path.name, len(empty_pages))   # 记录说明
    tables = parse_tables_with_pdfplumber(str(path))   # 第二步：提取表格
    cleaned_pages = clean_pages(pages)                 # 第三步：去水印与页眉页脚
    chunks = build_chunks(path.name, cleaned_pages)    # 第四步：语义切分
    result = {                                         # 组装解析结果
        "source": path.name,                           # 文件名
        "pages": len(pages),                           # 总页数
        "empty_pages": [p["page"] for p in empty_pages],   # 无文本层的页码
        "text_blocks": cleaned_pages,                  # PyMuPDF 的逐页文本（已清洗）
        "tables": tables,                              # pdfplumber 提取的表格
        "chunks": chunks,                              # 语义切分后的块
    }                                                  # 结果字典结束
    logger.info("解析完成：%s，%d 页 %d 块 %d 页含表格",
                path.name, result["pages"], len(chunks), len(tables))   # 记录完成日志
    return result                                      # 返回解析结果


def parse_all_pdfs(file_names: list = None) -> list:   # 批量解析入口
    """按 .env 的 PDF_FILES 解析 PDF；清单为空（None）时扫描 PDF_DIR 下全部 PDF。

    单个文件不存在、或单个文件解析抛异常，都只记日志并跳过，**不中断整批**。
    file_names 显式传入时优先用它（测试与临时调用用得上），否则读配置。
    """
    pdf_dir = Path(config.PDF_DIR)                     # 从配置读取 PDF 目录
    if not pdf_dir.exists():                           # 目录不存在时
        logger.error("PDF 目录不存在：%s", pdf_dir)     # 记录错误日志
        return []                                      # 返回空列表
    names = file_names if file_names is not None else config.get_pdf_files()   # 参数优先，其次读配置
    if names:                                          # 配置了清单
        targets = [Path(config.resolve_pdf_path(n)) for n in names]   # 逐条解析（支持相对与绝对路径）
    else:                                              # 清单为空表示扫全目录
        targets = sorted(pdf_dir.glob(f"*{PDF_SUFFIX}"))   # 扫描目录下全部 PDF
    logger.info("待解析文件共 %d 个", len(targets))     # 记录文件数量
    results = []                                       # 保存所有解析结果
    for path in targets:                               # 逐个解析
        if not path.exists():                          # 文件不存在
            logger.warning("文件不存在，跳过：%s", path)   # 记录告警后跳过
            continue                                   # 处理下一个
        try:                                           # 单个文件失败不影响整批
            results.append(parse_pdf(str(path)))       # 收集解析结果
        except Exception as exc:                       # 该文件解析出错
            logger.error("解析失败，跳过 %s：%s", path.name, exc)   # 记录错误后跳过
    return results                                     # 返回全部成功解析的结果


def save_chunks_to_json(all_results: list, out_path: str = None) -> str:   # 结果落盘
    """把解析结果写入 JSON 文件，data 目录不存在会自动创建，返回文件路径。"""
    target = Path(out_path) if out_path else CHUNK_JSON   # 未指定路径时用默认路径
    target.parent.mkdir(parents=True, exist_ok=True)      # 确保输出目录存在
    payload = {                                        # 组装要写出的内容
        "generated_at": int(time.time()),              # 生成时间戳
        "pdf_count": len(all_results),                 # PDF 个数
        "chunk_count": sum(len(r["chunks"]) for r in all_results),   # 块总数
        "results": all_results,                        # 每个 PDF 的解析明细
    }                                                  # 内容字典结束
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")   # 写文件
    logger.info("解析结果已写入：%s", target)           # 记录日志
    return str(target)                                 # 返回文件路径


def main() -> None:                                    # 命令行入口
    """命令行入口：默认解析 PDF 并落盘；--reindex 跑一键重建，--compare-mineru 跑对比。"""
    if "--reindex" in sys.argv:                        # 走一键重建分支
        import reindex                                 # 延迟导入：三步编排都在 reindex.py
        reindex.main()                                 # 解析 → 增强 → 入库
        return                                         # 重建结束即返回，不做默认解析
    if "--compare-mineru" in sys.argv:                 # 走对比分支
        import ingest_mineru                           # 延迟导入：MinerU 依赖重
        ingest_mineru.run_compare_command()            # 整个对比流程都在该模块里
        return                                         # 对比结束即返回，不做默认解析
    start = time.time()                                # 记录开始时间
    logger.info("开始批量解析，PDF 目录：%s", config.PDF_DIR)   # 记录目录
    results = parse_all_pdfs()                          # 批量解析
    print("=" * 78)                                     # 打印分隔线
    print(f"{'文件名':<52}{'页数':>6}{'块数':>7}{'表格页':>7}")   # 打印表头
    print("-" * 78)                                     # 打印分隔线
    for item in results:                                # 逐个打印结果
        name = item["source"]                           # 文件名
        if len(name) > 50:                              # 文件名太长时截断显示
            name = name[:47] + "..."                    # 截断并加省略号
        print(f"{name:<52}{item['pages']:>6}{len(item['chunks']):>7}{len(item['tables']):>7}")   # 打印一行
    print("-" * 78)                                     # 打印分隔线
    total_chunks = sum(len(r["chunks"]) for r in results)   # 统计块总数
    print(f"合计：{len(results)} 个 PDF，{total_chunks} 个语义块")   # 打印汇总
    print("=" * 78)                                     # 打印分隔线
    out_path = save_chunks_to_json(results)             # 结果写入 JSON
    print(f"输出文件：{out_path}")                       # 打印输出路径
    print(f"耗时：{time.time() - start:.1f} 秒")         # 打印耗时
    logger.info("批量解析结束，共 %d 块，输出 %s", total_chunks, out_path)   # 记录结束日志


if __name__ == "__main__":                             # 直接运行本文件时
    main()                                             # 执行命令行入口
