# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_ocr.py —— 工单四图像 OCR 模块

职责（见 docs/01_图像解析方案.md §5.3）：
  提取图内文字，输出 ocr_text 字段。

引擎降级链：
  1. PaddleOCR（工单四首选，PP-OCRv4 中英文）—— 未安装则跳过；
  2. Qwen2-VL 文字转录（vlm 引擎，延迟注入避免循环依赖）；
  3. 均不可用时返回空串并记 status，不阻塞流水线。
"""
import io
from typing import Callable, List, Optional

from loguru import logger
from PIL import Image

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"


class ImageOCREngine:
    """工单四：图像 OCR 引擎（PaddleOCR 优先，Qwen2-VL 转录兜底）"""

    def __init__(self, vlm_transcribe_fn: Optional[Callable[[Image.Image], str]] = None):
        # 工单四：PaddleOCR 惰性初始化（首次调用才加载，避免拖慢无 OCR 场景）
        self._paddle = None
        self._paddle_tried = False
        self.engine_name = "none"
        # 工单四：Qwen2-VL 转录回调（由 captioner 注入，模型常驻时零额外显存）
        self._vlm_transcribe_fn = vlm_transcribe_fn

    # ------------------------------------------------------------------
    def _get_paddle(self):
        """工单四：惰性加载 PaddleOCR；不可用返回 None（容错规范）"""
        if self._paddle_tried:
            return self._paddle
        self._paddle_tried = True
        try:
            from paddleocr import PaddleOCR  # 工单四：PaddleOCR 可选依赖
            # 工单四：PaddleOCR 3.x API（use_textline_orientation 替代 use_angle_cls，
            #         show_log 已移除；enable_mkldnn=False 规避 PIR/oneDNN 推理崩溃）
            try:
                self._paddle = PaddleOCR(lang="ch", use_textline_orientation=True,
                                         enable_mkldnn=False)
            except (TypeError, ValueError):
                self._paddle = PaddleOCR(use_angle_cls=True, lang="ch",
                                         show_log=False)
            self.engine_name = "paddleocr"
            logger.info("[image_ocr] PaddleOCR 就绪")
        except Exception as e:                      # 工单四：未安装/初始化失败→降级
            logger.warning(f"[image_ocr] PaddleOCR 不可用，降级 VLM 转录: {e}")
            self._paddle = None
        return self._paddle

    # ------------------------------------------------------------------
    def recognize(self, image: Image.Image) -> dict:
        """工单四：识别图内文字，返回 {ocr_text, engine, status}"""
        # 1) 工单四：PaddleOCR 主通道
        paddle = self._get_paddle()
        if paddle is not None:
            try:
                import numpy as np
                arr = np.array(image.convert("RGB"))
                result = paddle.predict(arr)        # 工单四：3.x 统一 predict 接口
                lines: List[str] = []
                for page in (result or []):
                    # 3.x 返回 OCRResult(dict)：rec_texts 为文本列表；兼容 2.x 嵌套结构
                    texts = page.get("rec_texts") if isinstance(page, dict) else None
                    if texts:
                        lines.extend(t.strip() for t in texts if t and t.strip())
                    elif isinstance(page, (list, tuple)):
                        for item in page:           # 2.x：[[box, (text, score)], ...]
                            txt = item[1][0] if isinstance(item, (list, tuple)) else str(item)
                            if txt and txt.strip():
                                lines.append(txt.strip())
                return {"ocr_text": "\n".join(lines), "engine": "paddleocr",
                        "status": "ok" if lines else "empty"}
            except Exception as e:                  # 工单四：单图失败不阻塞
                logger.warning(f"[image_ocr] PaddleOCR 识别失败，降级: {e}")

        # 2) 工单四：Qwen2-VL 文字转录兜底
        if self._vlm_transcribe_fn is not None:
            try:
                text = self._vlm_transcribe_fn(image)
                return {"ocr_text": text or "", "engine": "qwen2vl",
                        "status": "ok" if text else "empty"}
            except Exception as e:
                logger.warning(f"[image_ocr] VLM 转录失败: {e}")

        # 3) 工单四：全部不可用——优雅降级
        return {"ocr_text": "", "engine": "none", "status": "unavailable"}


def transcribe_prompt() -> str:
    """工单四：Qwen2-VL 图内文字转录指令（OCR 替代通道）"""
    return "逐字转录图中出现的全部文字，按原始阅读顺序输出，不要添加解释。"
