"""离线文档处理链的门面：解析 → 清洗 → 分块。

把这条链路上游要用的东西（DocumentParser / Chunker / 分块参数常量 / 清洗函数）
从各子模块聚合成一个稳定的导入面。调用方（app/api/knowledge.py、scripts/ingest.py、
scripts/seed.py）只从这里 import，子模块改名或挪位置时不会连带漂移。

enhance / ingest 刻意不在这里再导出：它们属于**入库**链路而不是解析链路，调用方
直接 `from ..core.document.enhance import ...`。ocr 是被 parser 内部使用的实现细节。

本包对重依赖（fitz / pdfplumber / rapidocr / PIL / numpy）全都在函数体内惰性 import，
所以 import 这个包不会加载任何模型，启动开销可以忽略——别在模块顶层加这些导入。
"""
from .chunker import (
    CHUNK_OVERLAP_MAX,
    CHUNK_OVERLAP_MIN,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    STRATEGIES,
    Chunker,
)
from .cleaner import clean_text, remove_watermark_lines
from .parser import DocumentParser

__all__ = [
    "Chunker",
    "CHUNK_SIZE_MIN",
    "CHUNK_SIZE_MAX",
    "CHUNK_OVERLAP_MIN",
    "CHUNK_OVERLAP_MAX",
    "STRATEGIES",
    "clean_text",
    "remove_watermark_lines",
    "DocumentParser",
]
