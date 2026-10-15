# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：修复型 PDF 解析器。判定文字型/扫描型/混合型，对扫描页补 OCR，
#           识别图纸页并导出图像供下游生成图形描述，解决图片型 PDF "零可提取文本" 的信息丢失。

import os                                                        # 路径处理
import re                                                        # 图号正则
import json                                                      # 解析结果缓存
import numpy as np                                               # 位图转数组
import fitz                                                      # PyMuPDF
import config                                                    # 全局配置
from logger import get_logger                                    # 日志

logger = get_logger("rag15.parser")

_OCR_ENGINE = None                                               # RapidOCR 单例（首次调用才加载）

FIGURE_PAGE_HINT = "说明书附图"                                   # 专利附图页的固定页眉
# 图号：图1 / 图 2 / 图10；前后不能再跟数字，避免把 "图01813417.3" 之类的编号误判为图号
_FIG_LABEL_RE = re.compile(r"(?<![0-9])图\s*([0-9]{1,2})(?![0-9])")


def get_ocr_engine():
    """惰性加载 RapidOCR 引擎并缓存，避免重复初始化模型。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR                 # 延迟导入
        logger.info("首次加载 RapidOCR 引擎 ...")
        _OCR_ENGINE = RapidOCR()
        logger.info("RapidOCR 引擎加载完成")
    return _OCR_ENGINE


def classify_document(doc):
    """整份文档分流：逐页统计可提取字符数，文字型 / 扫描型 / 混合型。"""
    chars = [len(p.get_text().strip()) for p in doc]              # 每页可提取字符数
    is_scan = [c < config.SCAN_PAGE_CHAR_THRESHOLD for c in chars]  # 低于阈值 → 扫描页
    scan_cnt = sum(is_scan)                                       # 扫描页数量
    ratio = scan_cnt / max(len(chars), 1)                         # 扫描页占比
    if ratio >= config.SCAN_RATIO_THRESHOLD:                      # 超阈值 → 扫描型
        kind = "扫描型"
    elif scan_cnt == 0:                                           # 无扫描页 → 文字型
        kind = "文字型"
    else:                                                         # 其余 → 混合型
        kind = "混合型"
    return {"kind": kind, "page_count": len(chars),               # 类型 / 总页数
            "scan_pages": scan_cnt, "scan_ratio": round(ratio, 3),  # 扫描页数 / 占比
            "page_chars": chars}                                  # 逐页字符数


def ocr_page(page, dpi=None):
    """把单页渲染为位图后做 OCR，返回按行拼接的文本与平均置信度。"""
    dpi = dpi or config.RENDER_DPI                                # 渲染分辨率
    pix = page.get_pixmap(dpi=dpi)                                # 渲染位图
    img = np.frombuffer(pix.samples, dtype=np.uint8)              # 原始字节
    img = img.reshape(pix.height, pix.width, pix.n)               # 还原 H×W×C
    if pix.n == 4:                                                # RGBA 去 alpha
        img = img[:, :, :3]
    result, _ = get_ocr_engine()(img)                             # 执行 OCR
    if not result:                                                # 无识别结果
        return "", 0.0, []
    lines, scores, boxes = [], [], []                             # 文本行 / 置信度 / 文本框
    for box, text, score in result:                               # 逐条结果
        if score is not None and score < config.OCR_MIN_SCORE:    # 低置信度丢弃
            continue
        lines.append(text)                                        # 文本
        scores.append(score or 0.0)                               # 分数
        boxes.append({"text": text, "score": round(score or 0.0, 3),
                      "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box]})
    avg = round(sum(scores) / len(scores), 3) if scores else 0.0  # 平均置信度
    return "\n".join(lines), avg, boxes


def largest_image_bbox(page):
    """返回页面中面积最大的嵌入图像 bbox；无图像时返回整页矩形。"""
    page_area = max(page.rect.width * page.rect.height, 1.0)      # 页面面积
    best, best_ratio = None, 0.0                                  # 最优图像与占比
    for img in page.get_images(full=True):                        # 遍历嵌入图像
        try:
            bbox = page.get_image_bbox(img)                       # 图像位置
        except Exception:                                         # 无有效 bbox 跳过
            continue
        ratio = (bbox.width * bbox.height) / page_area            # 面积占比
        if ratio > best_ratio:                                    # 记录最大者
            best, best_ratio = bbox, ratio
    if best is None:                                              # 无嵌入图像 → 整页
        return list(page.rect), 1.0
    return [best.x0, best.y0, best.x1, best.y1], round(best_ratio, 3)


def figure_labels(text):
    """从页面文本中提取图号（图1 / 图2 …），保持出现顺序去重。"""
    seen, out = set(), []                                         # 去重集合与结果
    for m in _FIG_LABEL_RE.finditer(text):                        # 正则扫描
        label = f"图{m.group(1)}"                                  # 规范化
        if label not in seen:                                     # 首次出现才保留
            seen.add(label)
            out.append(label)
    return out


def detect_figure_page(page, page_no, text, image_bbox, image_ratio):
    """判定该页是否为图纸页，并给出图纸的裁剪区域与图号。

    判定规则（对文字型/扫描型统一适用，不依赖图像尺寸）：
      1) 文本含专利附图页页眉"说明书附图"；
      2) 文本很短且出现"图N"字样（附图页正文极少，只有图号）。
    """
    labels = figure_labels(text)                                  # 页面出现的图号
    is_fig = (FIGURE_PAGE_HINT in text) or (len(text) < 200 and bool(labels))
    if not is_fig:
        return None
    bbox = image_bbox if image_ratio >= 0.30 else list(page.rect)  # 大图用图框，否则整页
    return {"page_no": page_no,                               # 图纸所在页码
            "labels": labels or ["图1"],                       # 图号列表（无则默认图1）
            "bbox": [round(float(v), 2) for v in bbox],        # 裁剪区域
            "image_ratio": image_ratio}                        # 图像面积占比


def export_figure(page, fig, out_path, dpi=None):
    """按图纸 bbox 裁剪渲染并保存为 PNG，返回路径。"""
    dpi = dpi or config.RENDER_DPI                                # 渲染分辨率
    pix = page.get_pixmap(dpi=dpi, clip=fitz.Rect(*fig["bbox"]))  # 只渲染图纸区域
    pix.save(out_path)                                            # 落盘
    return out_path


def _cache_path(pdf_path):
    """解析缓存文件路径；按文档名区分，存于 data/ 下。"""
    name = os.path.splitext(os.path.basename(pdf_path))[0]        # 文档名
    return os.path.join(config.DATA_DIR, f"parse_{name}.json")    # 缓存路径


def parse_document(pdf_path, figure_dir=None):
    """修复型解析入口：逐页取文本，扫描页自动补 OCR，识别并导出图纸页。

    返回 (pages, meta)：
        pages —— 每页 {page_no, text, source, chars, confidence, figure, labels, ocr_boxes}
        meta  —— 文档级分流信息
    """
    cache = _cache_path(pdf_path)                                 # 解析缓存
    if os.path.exists(cache):                                     # 命中缓存：OCR 代价高，直接复用
        with open(cache, "r", encoding="utf-8") as fh:             # 读取
            saved = json.load(fh)
        if saved.get("mtime") == os.path.getmtime(pdf_path):      # 源文件未变才复用
            logger.info("命中解析缓存 %s，跳过逐页 OCR", os.path.basename(cache))
            pages = saved["pages"]                                # 页面结果
            if figure_dir:                                        # 补导出图纸图片（缓存中只存裁剪框）
                os.makedirs(figure_dir, exist_ok=True)            # 确保目录存在
                doc = fitz.open(pdf_path)                         # 重新打开用于渲染
                for p in pages:
                    if p.get("figure"):
                        name = f"{os.path.splitext(saved['meta']['file'])[0]}_p{p['page_no']}.png"
                        p["figure_file"] = export_figure(doc[p["page_no"] - 1], p["figure"],
                                                         os.path.join(figure_dir, name))
                doc.close()                                       # 关闭文档
            return pages, saved["meta"]                           # 直接返回

    doc = fitz.open(pdf_path)                                     # 打开 PDF
    meta = classify_document(doc)                                 # 整体分流
    meta["file"] = os.path.basename(pdf_path)                     # 文件名
    logger.info("文档 %s：%s，共 %d 页，扫描页 %d 页（占比 %.1f%%）",
                meta["file"], meta["kind"], meta["page_count"],
                meta["scan_pages"], meta["scan_ratio"] * 100)

    pages = []                                                    # 逐页结果
    for idx, page in enumerate(doc):                              # 遍历页面
        page_no = idx + 1                                         # 页码（1 起）
        raw_chars = meta["page_chars"][idx]                       # 可提取字符数
        if raw_chars < config.SCAN_PAGE_CHAR_THRESHOLD:           # 扫描页 → OCR 补文本
            text, conf, boxes = ocr_page(page)                    # 识别
            source = "ocr"                                        # 来源标记
            logger.info("  第 %d 页 扫描页 → OCR 得 %d 字（置信度 %.3f）",
                        page_no, len(text), conf)
        else:                                                     # 文字页 → 直接提取
            text, conf, boxes = page.get_text().strip(), 1.0, []  # 文本、满分、无 OCR 框
            source = "text"                                       # 来源标记
        bbox, ratio = largest_image_bbox(page)                    # 最大图像区域
        fig = detect_figure_page(page, page_no, text, bbox, ratio)  # 图纸页判定
        entry = {"page_no": page_no, "text": text,                # 页码 / 正文
                 "source": source, "chars": len(text),            # 来源 / 字数
                 "confidence": conf,                               # 置信度
                 "figure": fig,                                    # 图纸信息或 None
                 "labels": fig["labels"] if fig else [],           # 图号
                 "ocr_boxes": boxes}                               # OCR 文本框（供图纸标注分析）
        if fig and figure_dir:                                    # 是图纸页且要求导出
            name = f"{os.path.splitext(meta['file'])[0]}_p{page_no}.png"
            entry["figure_file"] = export_figure(page, fig, os.path.join(figure_dir, name))
            logger.info("  第 %d 页 图纸页 %s → 导出 %s", page_no, fig["labels"],
                        os.path.basename(entry["figure_file"]))
        pages.append(entry)                                       # 收集
    doc.close()                                                   # 关闭
    with open(cache, "w", encoding="utf-8") as fh:                # 落盘解析缓存，避免重复 OCR
        json.dump({"mtime": os.path.getmtime(pdf_path), "pages": pages, "meta": meta},
                  fh, ensure_ascii=False)
    return pages, meta


if __name__ == "__main__":                                        # 自检入口
    import json                                                   # 打印结果
    _pages, _meta = parse_document(config.TARGET_PDF)             # 解析目标文档
    print(json.dumps({k: v for k, v in _meta.items() if k != "page_chars"},
                     ensure_ascii=False, indent=2))
    for _p in _pages:                                             # 逐页概要
        _fig = ",".join(_p["labels"]) if _p["figure"] else "-"
        print(f"p{_p['page_no']:>2} [{_p['source']:>4}] {_p['chars']:>4}字 图号={_fig}")
