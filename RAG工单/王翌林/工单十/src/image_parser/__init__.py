# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/__init__.py —— 工单四图像解析子包入口

模块构成（见 docs/01_图像解析方案.md）：
  - image_extractor ：PDF 图像提取（L1 位图 + L2 矢量区域渲染）
  - image_captioner ：Qwen2-VL 图像描述 + 解析流水线编排（caption→VQA→OCR）
  - image_vqa       ：图表/结构图/流程图预置问题模板 VQA
  - image_ocr       ：PaddleOCR 优先 + VLM 转录兜底的图内文字提取
  - 后续：image_embedding（Chinese-CLIP 嵌入）、image_store / image_retriever
"""
from src.image_parser.image_extractor import ImageExtractor      # 工单四：图像提取器
from src.image_parser.image_captioner import ImageCaptioner       # 工单四：解析编排器
from src.image_parser.image_vqa import ImageVQAEngine             # 工单四：VQA 引擎
from src.image_parser.image_ocr import ImageOCREngine             # 工单四：OCR 引擎

__all__ = ["ImageExtractor", "ImageCaptioner", "ImageVQAEngine", "ImageOCREngine"]
