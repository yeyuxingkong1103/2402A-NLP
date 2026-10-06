"""OCR：扫描件 / 图片兜底。

默认 RapidOCR（ONNX 后端，det/rec/cls 模型随 wheel 打包，免联网、免 GPU）。
provider 抽象预留：rapid | paddle | multimodal——与 parser.py 的注释呼应，
哪天要换 PaddleOCR / 多模态大模型，只改这里的引擎装配，parser 无感知。

惰性加载：RapidOCR 首次初始化约 0.6s、模型约 14MB，只在真遇到扫描件/图片时才
import + 建实例，之后复用；避免 import 时加载模型拖慢启动。
"""
from __future__ import annotations

import threading

_engine = None
# 上传接口是同步路由，FastAPI 把它丢进线程池并发执行，n 个请求同时上传扫描件时
# 会真的并发走到这里。锁保护的是**首次**实例化：没有它就会各自 import + 建一份
# RapidOCR（多份 ONNX 会话，白吃内存，也可能撞上 onnxruntime 的初始化竞态）。
_lock = threading.Lock()


def _get_engine():
    """惰性单例：首次调用才导入并实例化 RapidOCR（线程安全）。

    双重检查是为了让稳态调用不付锁的代价——只有 _engine 仍为 None 时才去抢锁。
    """
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR()
    return _engine


def ocr_image(image) -> str:
    """识别一张图（numpy 数组或 PIL Image），按阅读顺序拼成多行文本。

    RapidOCR 返回 ``[box, text, score]``，score 在 1.2.3 里是字符串、这里不解析
    （引擎内部已过滤低置信框）。逐行用换行拼接，交给下游 chunker 继续处理。
    """
    engine = _get_engine()

    # numpy 在这里才 import：本模块会被 parser 顶层导入，而 parser 又被包 __init__ 导出，
    # 顶层 import numpy 等于把它的加载挂在服务启动路径上。
    import numpy as np

    # 传 PIL Image 时**必须已经是 RGB**（调用方负责 convert("RGB") / alpha=False）：
    # RGBA 会转出 4 通道、调色板图会转出索引值，喂给识别模型要么报错要么识别出乱码。
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # 第二个返回值是引擎的耗时统计，用不上；result 为 None 表示一个文本框都没检出
    # （空图、纯色图），这不是异常，下面按空串处理。
    result, _ = engine(image)
    if not result:
        return ""
    return "\n".join(text for _box, text, _score in result if text)
