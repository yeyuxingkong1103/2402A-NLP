"""统一业务异常层级。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 基础设施（对应 设计/接口设计.md §1.2、§9）

纪律：业务失败一律抛具体子类，禁止裸 ``Exception``；每个 ``except`` 分支
必须 ``logger.exception(...)`` 记录堆栈或返回结构化 ``error`` 字段（禁止 ``except: pass``）。
错误码与用户可见文案的映射见 ``config.Settings.app`` 与 §9。
"""

from __future__ import annotations

from typing import Any


class RAGError(Exception):
    """工单2 全部业务异常的基类：人工智能NLP-RAG-基于PDF文档的问答系统优化。"""

    code: str = "RAG_ERROR"
    #: 用户可见兜底文案（子类可覆盖，也可由 config 动态覆盖）
    user_message: str = "系统繁忙，请稍后再试"

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})

    def as_dict(self) -> dict[str, Any]:
        """结构化错误（供 HTTP 层与日志使用）。"""
        return {"code": self.code, "message": self.message, "detail": self.detail}


class PDFParseError(RAGError):
    """PDF 不存在 / 损坏 / 解析中断。"""

    code = "PDF_PARSE_ERROR"
    user_message = "PDF 解析失败，请检查文件是否损坏"


class IndexNotReadyError(RAGError):
    """索引缺失或维度不一致。"""

    code = "INDEX_NOT_READY"
    user_message = "索引未就绪，请先执行 研发/scripts/build_index.py"


class RetrievalError(RAGError):
    """嵌入失败、索引为空、检索内部异常。"""

    code = "RETRIEVAL_ERROR"
    user_message = "检索服务异常，请稍后再试"


class LLMUnavailableError(RAGError):
    """LLM 服务探测失败或生成异常。"""

    code = "LLM_UNAVAILABLE"
    user_message = "服务暂时不可用，请稍后再试"


class StorageError(RAGError):
    """SQLite / 文件读写失败。"""

    code = "STORAGE_ERROR"
    user_message = "数据保存失败，请稍后再试"


class ConfigError(RAGError):
    """配置非法（未知后端名、维度非法、路径不可创建等）。"""

    code = "CONFIG_ERROR"
    user_message = "系统配置错误，请联系管理员"


#: 错误码 -> 用户可见文案（HTTP 层与 UI 共用）
ERROR_MESSAGES: dict[str, str] = {
    PDFParseError.code: PDFParseError.user_message,
    IndexNotReadyError.code: IndexNotReadyError.user_message,
    RetrievalError.code: RetrievalError.user_message,
    LLMUnavailableError.code: LLMUnavailableError.user_message,
    StorageError.code: StorageError.user_message,
    ConfigError.code: ConfigError.user_message,
    "NO_EVIDENCE": "不清楚",
    "PAGE_FILTER_EMPTY": "指定页码未检索到内容",
}


def user_message_for(code: str, default: str = "服务暂时不可用，请稍后再试") -> str:
    """按错误码返回用户友好文案（不暴露堆栈）。"""
    return ERROR_MESSAGES.get(code, default)
