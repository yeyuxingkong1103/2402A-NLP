# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：解析器注册表。每种文档分类标签对应一个解析器适配器，工作流按标签把文档
#           分发到这里。接口统一为 parse(path, cfg) -> dict，便于替换为 RAGFlow 的 DeepDoc。

import os                                                          # 路径
import fitz                                                        # PyMuPDF

_OCR_ENGINE = None                                                 # RapidOCR 单例


def _ocr_engine():
    """惰性加载 RapidOCR，避免非扫描分支付出加载代价。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR                  # 延迟导入
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def text_parser(path, cfg, max_pages=0):
    """文字型 PDF / 文本类：直接抽取文本层。"""
    doc = fitz.open(path)                                          # 打开
    try:
        limit = doc.page_count if not max_pages else min(doc.page_count, max_pages)
        text = "\n".join(doc[i].get_text() for i in range(limit))  # 逐页抽取
        return {"parser": "text_parser", "ok": True, "pages": limit,
                "chars": len(text), "preview": text[:200].replace("\n", " ")}
    finally:
        doc.close()


def ocr_parser(path, cfg, max_pages=1):
    """扫描型 PDF：渲染后做 OCR。演示流程只处理前若干页，避免耗时过长。"""
    dpi = cfg.get("runtime.ocr_demo_dpi", 200)                     # 渲染分辨率
    engine = _ocr_engine()                                         # OCR 引擎
    doc = fitz.open(path)                                          # 打开
    try:
        limit = doc.page_count if not max_pages else min(doc.page_count, max_pages)
        chunks = []                                                # 逐页 OCR 结果
        for i in range(limit):
            pix = doc[i].get_pixmap(dpi=dpi)                       # 渲染位图
            import numpy as np                                     # 延迟导入
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:                                         # 去 alpha 通道
                img = img[:, :, :3]
            result, _ = engine(img)                                # 识别
            chunks.append("".join(t for _b, t, _s in (result or [])) if result else "")
        text = "\n".join(chunks)                                   # 合并
        return {"parser": "ocr_parser", "ok": True, "pages": limit,
                "chars": len(text), "preview": text[:200].replace("\n", " ")}
    finally:
        doc.close()


def hybrid_parser(path, cfg, max_pages=1):
    """混合型 PDF：逐页判断，有文本层直接抽，没有才 OCR。"""
    char_th = cfg.get("pdf_triage.char_threshold", 30)             # 单页字符阈值
    doc = fitz.open(path)                                          # 打开
    try:
        limit = doc.page_count if not max_pages else min(doc.page_count, max_pages)
        texts, ocr_pages = [], []                                  # 文本与 OCR 页
        for i in range(limit):
            t = doc[i].get_text()                                  # 先试文本层
            if len(t.strip()) >= char_th:                          # 有文本层
                texts.append(t)
            else:
                ocr_pages.append(i + 1)                            # 记为需要 OCR 的页
        if ocr_pages:                                              # 有扫描页才调 OCR
            res = ocr_parser(path, cfg, max_pages=len(ocr_pages))
            texts.append(res.get("preview", ""))
        text = "\n".join(texts)
        return {"parser": "hybrid_parser", "ok": True, "pages": limit,
                "ocr_pages": ocr_pages, "chars": len(text),
                "preview": text[:200].replace("\n", " ")}
    finally:
        doc.close()


def docx_parser(path, cfg, max_pages=0):
    """Word 文档：复用 Skill 的 OOXML 文本抽取。"""
    from document_quality_assessment.scanner import _ooxml_text  # 复用已有实现
    text = _ooxml_text(path)                                       # 抽取
    return {"parser": "docx_parser", "ok": True, "pages": None,
            "chars": len(text), "preview": text[:200].replace("\n", " ")}


def passthrough_parser(path, cfg, max_pages=0):
    """默认透传：不解析，仅登记，交由后续人工或专用解析器处理。"""
    return {"parser": "passthrough_parser", "ok": True, "pages": None,
            "chars": 0, "preview": "", "note": "未匹配到专用解析器，已登记待人工处理"}


# 解析器名 → 实现函数（键与配置 routing.label_to_parser 的取值一致）
REGISTRY = {"text_parser": text_parser, "ocr_parser": ocr_parser,
            "hybrid_parser": hybrid_parser, "docx_parser": docx_parser,
            "markdown_parser": text_parser, "table_parser": passthrough_parser,
            "pptx_parser": docx_parser, "passthrough_parser": passthrough_parser}


def get_parser(name):
    """按名字取解析器；未注册的返回透传解析器。"""
    return REGISTRY.get(name, passthrough_parser)
