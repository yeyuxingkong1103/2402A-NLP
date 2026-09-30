# -*- coding: utf-8 -*-
"""图片文字识别：RapidOCR（PaddleOCR 模型的 ONNX 轻量版，CPU 友好）。

引擎懒加载：首次识别图片时才初始化（约 1-2 秒），之后复用。
"""
import logging
# 解析：日志模块

logger = logging.getLogger("rag-roleplay.ocr")
# 解析：本模块 logger（记录引擎初始化等事件）

_engine = None
# 解析：模块级引擎单例（惰性初始化）


class RapidOCREngine:
    """RapidOCR 引擎包装：调用返回按坐标排序后的文字行拼接文本。"""

    def __init__(self):
        # 解析：构造——不加载模型（懒加载）
        self._engine = None
        # 解析：底层 RapidOCR 实例（首次使用时创建）

    def __call__(self, image_bytes: bytes) -> str:
        # 解析：让包装对象可调用（engine(image) 语法）
        if self._engine is None:
            # 解析：首次调用才加载模型
            from rapidocr import RapidOCR
            # 解析：延迟导入——避免未安装时 import 本模块失败

            logger.info("初始化 RapidOCR 引擎（首次）...")
            # 解析：记录初始化日志（首次约 1-2 秒）
            self._engine = RapidOCR()
            # 解析：创建引擎实例（自动加载检测+识别模型）
        result = self._engine(image_bytes)
        # 解析：执行 OCR 识别
        # rapidocr 3.x：txts 为按检测顺序（自上而下、自左而右）的文本元组
        txts = getattr(result, "txts", None)
        # 解析：取识别文本元组（rapidocr 3.x 返回 RapidOCROutput 对象）
        if not txts:
            # 解析：没有识别到文字
            return ""
            # 解析：返回空串
        return "\n".join(str(t) for t in txts if t)
        # 解析：每行文字用换行拼接（保持阅读顺序）


def get_engine():
    """进程内单例（测试可 monkeypatch）。"""
    global _engine
    # 解析：引用模块级单例变量
    if _engine is None:
        # 解析：未初始化
        _engine = RapidOCREngine()
        # 解析：创建包装引擎
    return _engine
    # 解析：返回引擎（进程内只创建一次）


def recognize_image(image_bytes: bytes) -> str:
    """识别图片上的文字，返回按阅读顺序拼接的文本。"""
    return get_engine()(image_bytes)
    # 解析：取引擎并调用识别（测试通过 monkeypatch get_engine 注入假引擎）
