# -*- coding: utf-8 -*-
"""工单3 统一异常体系（设计/接口设计.md §1.5、§3.3 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

纪律：
    1. 禁止 ``except: pass``、禁止空 ``except`` 分支；
    2. 允许的降级必须在日志中留下 ``*.degrade`` 事件（含 reason）；
    3. 异常对象必须携带 ``code`` / ``stage`` / ``detail`` 三个可序列化字段，便于 API 回传与留痕。
"""

from __future__ import annotations

import traceback
from typing import Any

# 工单编号常量：所有日志与异常载荷都会带上，便于跨工单检索
WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 错误码 → 中文说明（设计 §3.3）
ERROR_CODES: dict[str, str] = {
    "RAG-0000": "未分类错误",
    "RAG-1000": "配置非法（fail fast）",
    "RAG-1001": "必需目录不存在",
    "RAG-1002": "环境变量类型/取值非法",
    "RAG-2000": "PDF 打不开或已损坏",
    "RAG-2001": "PDF 页数异常",
    "RAG-2002": "单页解析失败（已降级为空文本）",
    "RAG-2100": "单表归一化失败（显式降级为文本块）",
    "RAG-2200": "分块元数据缺失或页码越界",
    "RAG-2201": "chunk_id 重复",
    "RAG-3000": "嵌入服务不可用且本地降级失败",
    "RAG-3100": "索引文件缺失",
    "RAG-3200": "向量维度不匹配（不静默截断）",
    "RAG-4000": "检索异常",
    "RAG-5000": "LLM 后端探测失败（0.5 s 超时、不重试）",
    "RAG-5100": "生成超时",
    "RAG-5200": "返回体无可用文本",
    "RAG-6000": "引用非法（页码越界/块不存在/无支撑原文）",
    "RAG-6100": "生成或闸门阶段致命错误",
    "RAG-7000": "SQLite 异常",
    "RAG-8000": "评估脚本异常",
    "RAG-9000": "日志初始化失败",
}


class RagError(Exception):
    """工单3 全部可预期异常的基类。"""

    code: str = "RAG-0000"
    stage: str = "core"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        stage: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        if code:
            self.code = code
        if stage:
            self.stage = stage
        self.detail: dict[str, Any] = dict(detail or {})

    def to_payload(self) -> dict[str, Any]:
        """转成 API 可回传的载荷：{"code","stage","message","detail"}。"""
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
        }

    def __str__(self) -> str:  # noqa: D105 —— 保持简短可读
        return f"[{self.code}/{self.stage}] {self.message}"


class ConfigError(RagError):
    """配置非法/缺失（fail fast）。"""

    code = "RAG-1000"
    stage = "config"


class PdfParseError(RagError):
    """PDF 打不开/损坏/页数异常。"""

    code = "RAG-2000"
    stage = "parse"


class TableParseError(RagError):
    """单表归一化失败：上层必须显式降级为该页文本块。"""

    code = "RAG-2100"
    stage = "table"


class ChunkError(RagError):
    """页码越界、元数据缺失、chunk_id 重复。"""

    code = "RAG-2200"
    stage = "chunk"


class EmbeddingError(RagError):
    """嵌入服务不可用且本地降级也失败。"""

    code = "RAG-3000"
    stage = "embed"


class IndexMissingError(RagError):
    """索引文件缺失。"""

    code = "RAG-3100"
    stage = "index"


class IndexDimensionError(RagError):
    """向量维度不匹配（不静默截断）。"""

    code = "RAG-3200"
    stage = "index"


class RetrievalError(RagError):
    """检索异常（上抛后由上层转「不清楚」）。"""

    code = "RAG-4000"
    stage = "retrieve"


class LLMProbeError(RagError):
    """后端探测失败（0.5 s 超时、不重试）。"""

    code = "RAG-5000"
    stage = "generate"


class LLMTimeoutError(RagError):
    """生成超时。"""

    code = "RAG-5100"
    stage = "generate"


class LLMResponseError(RagError):
    """返回体无可用文本。"""

    code = "RAG-5200"
    stage = "generate"


class CitationError(RagError):
    """引用非法（页码越界/块不存在/无支撑原文）。"""

    code = "RAG-6000"
    stage = "citation"


class AnswerError(RagError):
    """生成/闸门阶段致命错误。"""

    code = "RAG-6001"
    stage = "generate"


class StorageError(RagError):
    """SQLite 异常。"""

    code = "RAG-7000"
    stage = "storage"


class EvalError(RagError):
    """评估脚本异常。"""

    code = "RAG-8000"
    stage = "eval"


class LogSetupError(RagError):
    """日志初始化失败。"""

    code = "RAG-9000"
    stage = "config"


def wrap(exc: BaseException, *, code: str, stage: str, **detail: Any) -> RagError:
    """把任意异常包装成 RagError（已是 RagError 时合并 detail 后原样返回）。"""
    if isinstance(exc, RagError):
        merged = dict(exc.detail)
        merged.update(detail)
        exc.detail = merged
        return exc
    payload = dict(detail)
    payload.setdefault("cause_type", type(exc).__name__)
    payload.setdefault("cause_message", str(exc))
    return RagError(str(exc), code=code, stage=stage, detail=payload)


def error_payload(exc: BaseException) -> dict[str, Any]:
    """任意异常 → {"code","stage","message","detail"}（含 traceback 摘要）。"""
    if isinstance(exc, RagError):
        payload = exc.to_payload()
        payload["detail"].setdefault("traceback_tail", _tb_tail(exc))
        return payload
    return {
        "code": "RAG-0000",
        "stage": "core",
        "message": str(exc),
        "detail": {"cause_type": type(exc).__name__, "traceback_tail": _tb_tail(exc)},
    }


def _tb_tail(exc: BaseException, *, lines: int = 6) -> str:
    """取 traceback 末尾若干行，避免日志被超长堆栈淹没。"""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    parts = [ln for ln in tb.strip().splitlines() if ln.strip()]
    return "\n".join(parts[-lines:])
