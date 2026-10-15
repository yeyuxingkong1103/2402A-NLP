# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：PDF 页面类型识别（文字型/扫描型/混合型）。按页判断——单页可提取字符数低于
#           阈值即视为扫描页；扫描页占比超过阈值即标记整个文件为扫描型。所有阈值来自配置，
#           落在阈值附近的文件额外进入“待确认”列表。另记录图像覆盖率作为独立交叉验证信号。

import os                                                          # 路径处理
import fitz                                                        # PyMuPDF

# 交叉验证用：图像面积占比达到该值即认为该页“靠图承载信息”
IMAGE_PAGE_RATIO = 0.30


def _largest_image_ratio(page):
    """返回单页中最大嵌入图像的面积占页面面积之比；无图像返回 0.0。"""
    page_area = max(page.rect.width * page.rect.height, 1.0)       # 页面面积
    best = 0.0                                                     # 最大占比
    for img in page.get_images(full=True):                         # 遍历嵌入图像
        try:
            bbox = page.get_image_bbox(img)                        # 图像位置
        except Exception:                                          # 部分损坏图像取不到位置
            continue
        if bbox.is_empty or bbox.is_infinite:
            continue
        area = abs(bbox.width * bbox.height)                       # 图像面积
        best = max(best, area / page_area)                         # 取最大
    return min(best, 1.0)                                          # 归一化到 0~1


def classify_kind(scan_ratio, scan_th, text_th):
    """按扫描页占比给出文件级判定：scan / text / mixed。"""
    if scan_ratio >= scan_th:                                      # 扫描页占绝对多数
        return "scan"
    if scan_ratio <= text_th:                                      # 扫描页很少
        return "text"
    return "mixed"                                                 # 介于两者之间


def probe_pdf(path, cfg):
    """探测单个 PDF：返回 (全文可提取文本, 页面类型信息 dict)。

    抛出的异常由上层 scanner 捕获并记入“解析失败”，不影响批处理继续。
    """
    char_th = cfg.get("pdf_triage.char_threshold", 30)             # 单页字符阈值
    scan_th = cfg.get("pdf_triage.scan_ratio_threshold", 0.70)     # 扫描型占比阈值
    text_th = cfg.get("pdf_triage.text_ratio_threshold", 0.30)     # 文字型占比阈值
    margin = cfg.get("pdf_triage.borderline_margin", 0.10)         # 待确认带宽
    max_probe = cfg.get("pdf_triage.max_pages_probe", 0)           # 逐页探测上限（0=全部）

    doc = fitz.open(path)                                          # 打开 PDF
    try:
        if doc.needs_pass:                                         # 加密文档无法解析
            raise ValueError("PDF 已加密，需要密码")
        page_count = doc.page_count                                # 总页数
        limit = page_count if not max_probe else min(page_count, max_probe)
        texts, scan_pages, image_ratios = [], 0, []                # 文本 / 扫描页数 / 图像占比
        for idx in range(limit):                                   # 逐页探测
            page = doc[idx]                                        # 当前页
            t = page.get_text(cfg.get("pdf_triage.text_extract_flags", "text")) or ""
            texts.append(t)                                        # 收集文本
            if len(t.strip()) < char_th:                           # 字符数低于阈值 → 扫描页
                scan_pages += 1
            image_ratios.append(_largest_image_ratio(page))        # 图像覆盖率
        full_text = "\n".join(texts)                               # 全文（供长度统计/敏感检测）
        if limit < page_count:                                     # 未探测完的页按比例外推
            full_text = full_text + ""                             # 长度统计按已探测部分计
        ratio = round(scan_pages / float(limit), 4) if limit else 0.0
        kind = classify_kind(ratio, scan_th, text_th)              # 文件级判定
        img_mean = round(sum(image_ratios) / len(image_ratios), 4) if image_ratios else 0.0
        img_pages = sum(1 for r in image_ratios if r >= IMAGE_PAGE_RATIO)
        img_ratio = round(img_pages / float(limit), 4) if limit else 0.0
        # 启发式与图像覆盖率信号不一致时也要人工确认（例如极少文字但也没大图）
        disagree = (kind == "scan" and img_ratio < text_th) or \
                   (kind == "text" and img_ratio > scan_th)
        borderline = (abs(ratio - scan_th) <= margin) or (abs(ratio - text_th) <= margin)
        info = {"page_count": page_count,                          # 总页数
                "pages_probed": limit,                             # 实际探测页数
                "scan_pages": scan_pages,                          # 扫描页数
                "scan_ratio": ratio,                               # 扫描页占比
                "pdf_kind": kind,                                  # text / scan / mixed
                "avg_chars_per_page": round(len(full_text) / max(limit, 1), 1),
                "image_ratio_mean": img_mean,                      # 图像覆盖率均值
                "image_page_ratio": img_ratio,                     # 大图页占比（交叉验证）
                "need_confirm": bool(borderline or disagree),      # 是否进“待确认”列表
                "confirm_reason": _reason(borderline, disagree, ratio,
                                          scan_th, text_th, img_ratio)}
    finally:
        doc.close()                                                # 确保释放句柄
    return full_text, info


def _reason(borderline, disagree, ratio, scan_th, text_th, img_ratio):
    """生成进入“待确认”列表的原因说明；无需确认时返回空串。"""
    items = []                                                     # 原因条目
    if borderline:
        items.append("扫描页占比 %.2f 落在阈值 %.2f/%.2f 附近"
                     % (ratio, text_th, scan_th))
    if disagree:
        items.append("字符级启发式与图像覆盖率 %.2f 的判断不一致" % img_ratio)
    return "；".join(items)
