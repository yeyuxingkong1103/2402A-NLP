"""Streamlit 网页界面：基于 PDF 文档的 RAG 问答系统（工单1 · 阶段 3）。

本模块**只负责界面层**，不实现任何解析、检索、生成逻辑：

1. 通过契约接口 ``app.core.qa_engine.QAEngine`` 调用问答引擎，
   使用的接口仅限：``ask`` / ``stream`` / ``new_conversation`` / ``list_conversations``
   / ``switch_conversation`` / ``clear_conversation`` / ``last_retrieved`` / ``stats``
   / ``submit_feedback``。
2. 问答引擎（``app/core/qa_engine.py``）可能尚未完成、或其依赖未就绪，
   因此所有**导入与调用**都做了容错处理：失败时在界面上给出中文提示，
   绝不因为异常而白屏（界面健壮性是本工单的第一要求）。
3. 按工单第 5.4 节要求，界面**只展示最终答案与引用来源**；
   意图识别、Query 改写、检索中间步骤等中间信息只写日志（``logs/`` 下）。

启动方式::

    # 方式一（推荐）：项目根目录下
    streamlit run app/ui/streamlit_app.py

    # 方式二：模块方式运行
    python -m app.ui.streamlit_app
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

# ==========================================================================
# 0. 路径引导：无论用哪种方式启动，都要保证 ``app`` 包可以被导入
#    （``streamlit run 研发/app/ui/streamlit_app.py`` 时 sys.path 里可能没有源码根）
#
#    本文件位于 <root>/研发/app/ui/streamlit_app.py：
#      parents[1] = app    parents[2] = 研发（加入 sys.path 才能 import app.*）
#      parents[3] = 项目根（定位 data / logs / models）
# ==========================================================================
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入，运行时不强依赖
    from app.core.qa_engine import QAEngine
    from app.models.schemas import Answer, Citation, Conversation, RetrievedChunk

F = TypeVar("F", bound=Callable[..., Any])
UI_MODULE = "app.ui.streamlit_app"


# ==========================================================================
# 1. 依赖容错：streamlit / 配置 / 日志
#    —— 任何一项缺失都不能让模块导入直接崩溃，否则界面会白屏
# ==========================================================================
try:  # streamlit 是界面硬依赖
    import streamlit as st

    HAS_STREAMLIT = True
    STREAMLIT_IMPORT_ERROR = ""
except Exception as _exc:  # pragma: no cover - 环境缺依赖分支
    st = None  # type: ignore[assignment]
    HAS_STREAMLIT = False
    STREAMLIT_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


class _StderrLogger:
    """``app.core.logging_conf`` 不可用时的降级日志器（写 stderr，不静默失败）。"""

    def _emit(self, level: str, module: str, message: str, **extra: Any) -> None:
        print(f"[{level}] {module} | {message} | {extra}", file=sys.stderr)

    def debug(self, module: str, message: str, **extra: Any) -> None:
        self._emit("DEBUG", module, message, **extra)

    def info(self, module: str, message: str, **extra: Any) -> None:
        self._emit("INFO", module, message, **extra)

    def warning(self, module: str, message: str, **extra: Any) -> None:
        self._emit("WARNING", module, message, **extra)

    def error(self, module: str, message: str, **extra: Any) -> None:
        self._emit("ERROR", module, message, **extra)

    def exception(self, module: str, message: str, **extra: Any) -> None:
        self._emit("ERROR", module, message, traceback=traceback.format_exc(), **extra)


def _identity_trace(func: F) -> F:
    """``@trace`` 不可用时的空装饰器，保证接口一致。"""
    return func


try:  # 项目内日志模块（loguru 缺失时其内部已自动降级到标准库 logging）
    from app.core.logging_conf import logger, trace
except Exception as _log_exc:  # pragma: no cover - 环境缺依赖分支
    logger = _StderrLogger()  # type: ignore[assignment]
    trace = _identity_trace  # type: ignore[assignment]
    _LOGGING_IMPORT_ERROR = f"{type(_log_exc).__name__}: {_log_exc}"
else:
    _LOGGING_IMPORT_ERROR = ""

try:  # 项目内配置模块
    from app.core.config import get_settings

    HAS_SETTINGS = True
    _SETTINGS_IMPORT_ERROR = ""
except Exception as _cfg_exc:  # pragma: no cover - 环境缺依赖分支
    get_settings = None  # type: ignore[assignment]
    HAS_SETTINGS = False
    _SETTINGS_IMPORT_ERROR = f"{type(_cfg_exc).__name__}: {_cfg_exc}"


def _fallback_settings() -> Any:
    """``app.core.config`` 不可用时的最小默认配置（保证界面仍能渲染）。"""
    root = _PROJECT_ROOT
    return SimpleNamespace(
        app=SimpleNamespace(
            app_name="基于 PDF 文档的 RAG 问答系统",
            version="1.0.0",
            unknown_answer="不清楚",
            first_token_budget_seconds=3.0,
            expose_intermediate_steps=False,
        ),
        paths=SimpleNamespace(
            project_root=root,
            data_raw=root / "data" / "raw",
            logs=root / "logs",
            default_pdf=root / "data" / "raw" / "招股说明书1.pdf",
        ),
    )


_SETTINGS_CACHE: Any | None = None


def _settings() -> Any:
    """获取全局配置（带缓存与降级），任何异常都不会抛出到界面。"""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        if not HAS_SETTINGS:
            logger.warning(UI_MODULE, "配置模块不可用，使用内置默认配置", error=_SETTINGS_IMPORT_ERROR)
            _SETTINGS_CACHE = _fallback_settings()
        else:
            try:
                _SETTINGS_CACHE = get_settings()
            except Exception:
                logger.exception(UI_MODULE, "加载 app.core.config 失败，使用内置默认配置")
                _SETTINGS_CACHE = _fallback_settings()
    return _SETTINGS_CACHE


def _app_setting(name: str, default: Any) -> Any:
    """读取 ``settings.app.<name>``，失败时返回默认值。"""
    try:
        return getattr(_settings().app, name, default)
    except Exception:
        return default


def _path_setting(name: str, default: Path) -> Path:
    """读取 ``settings.paths.<name>``，失败时返回默认值。"""
    try:
        value = getattr(_settings().paths, name, default)
        return Path(value)
    except Exception:
        return default


def _project_name() -> str:
    """项目名（侧边栏与标题展示）。"""
    return str(_app_setting("app_name", "基于 PDF 文档的 RAG 问答系统"))


def _project_version() -> str:
    """版本号。"""
    return str(_app_setting("version", "1.0.0"))


def _unknown_answer() -> str:
    """工单约定的兜底回复文案（默认“不清楚”）。"""
    return str(_app_setting("unknown_answer", "不清楚"))


def _first_token_budget_seconds() -> float:
    """首字响应预算（秒），工单验收要求 3 秒。"""
    try:
        return float(_app_setting("first_token_budget_seconds", 3.0))
    except Exception:
        return 3.0


def _default_pdf() -> Path:
    """默认加载的 PDF：``data/raw/招股说明书1.pdf``。"""
    return _path_setting("default_pdf", _PROJECT_ROOT / "data" / "raw" / "招股说明书1.pdf")


def _raw_dir() -> Path:
    """原始 PDF 存放目录。"""
    return _path_setting("data_raw", _PROJECT_ROOT / "data" / "raw")


# ==========================================================================
# 2. Streamlit 兼容层
# ==========================================================================
def cache_resource(**cache_kwargs: Any) -> Callable[[F], F]:
    """``st.cache_resource`` 的兼容包装。

    - streamlit 可用：等价于 ``@st.cache_resource(**cache_kwargs)``；
    - streamlit 不可用：原样返回函数，保证模块仍可被导入做语法/契约自检。
    """

    def _decorator(func: F) -> F:
        if not HAS_STREAMLIT:
            return func
        try:
            return st.cache_resource(**cache_kwargs)(func)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - 极端情况下退化为普通函数
            logger.exception(UI_MODULE, "st.cache_resource 不可用，已退化为普通函数")
            return func

    return _decorator


def _has_streamlit_runtime() -> bool:
    """当前是否运行在 Streamlit 运行时中（``streamlit run`` 或 AppTest）。

    注意：``streamlit run`` 会以 ``__name__ == "__main__"`` 执行本文件，
    此时**不能**抛 ``SystemExit``（它是 ``BaseException``，会直接终止
    Streamlit 的脚本线程，导致页面卡住不再重跑）。
    """
    try:
        from streamlit import runtime  # noqa: PLC0415 - 惰性导入，避免无 streamlit 时报错

        return bool(runtime.exists())
    except Exception:
        return False


def _clear_engine_cache() -> None:
    """清空引擎缓存，用于“重新初始化问答引擎”。"""
    try:
        loader = globals().get("load_engine")
        clear = getattr(loader, "clear", None)
        if callable(clear):
            clear()
        elif HAS_STREAMLIT:
            st.cache_resource.clear()  # type: ignore[union-attr]
    except Exception:
        logger.exception(UI_MODULE, "清理引擎缓存失败")


# ==========================================================================
# 3. 会话状态
# ==========================================================================
# 引擎 stats() 常见字段的中文标签（未收录的字段按原样展示）
STAT_LABELS: dict[str, str] = {
    "pages": "文档页数",
    "tables": "表格数",
    "chunks": "片段数",
    "vector_count": "向量条数",
    "bm25_documents": "BM25 文档数",
    "conversations": "会话数",
    "messages": "消息数",
    "feedback": "反馈数",
}

STATE_DEFAULTS: dict[str, Any] = {
    "conversation_id": None,  # 当前会话 ID
    "messages": [],  # 当前会话的消息列表（界面侧记录）
    "messages_by_conv": {},  # conversation_id -> 消息列表（用于切换会话时回放）
    "conv_meta": {},  # conversation_id -> 引擎侧消息条数
    "pdf_path": None,  # 当前选中的 PDF 绝对路径
    "doc_id": None,  # 传给 QAEngine 的 doc_id
    "doc_id_override": "",  # 侧边栏手填的 doc_id（优先级最高）
    "pending_question": None,  # 由“演示问题”按钮触发的待处理提问
    "_pdf_initialized": False,
}


def _init_state() -> None:
    """初始化 ``st.session_state``（幂等）。"""
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # 首次运行：默认加载《招股说明书1.pdf》
    if not st.session_state.get("_pdf_initialized"):
        st.session_state["_pdf_initialized"] = True
        default = _default_pdf()
        if default.exists():
            st.session_state["pdf_path"] = str(default)
            st.session_state["doc_id"] = _doc_id_for(default)
        else:
            logger.warning(UI_MODULE, "默认 PDF 不存在，等待用户上传", path=str(default))


def _doc_id_for(pdf_path: Path) -> str | None:
    """推导传给 ``QAEngine`` 的 doc_id。

    规则：
    1. 侧边栏手填的“文档 ID”优先级最高；
    2. 选择默认文档《招股说明书1.pdf》时返回 ``None``，交由引擎解析默认知识库
       —— 索引可能是用自动生成的 doc_id 建立的，强行传文件名反而找不到索引；
    3. 其它 PDF 用文件名（不含扩展名）作为 doc_id 尝试，若该 doc_id 没有索引，
       引擎句柄会自动回退到默认知识库并在界面给出中文提示。
    """
    override = str(st.session_state.get("doc_id_override") or "").strip()
    if override:
        return override
    try:
        if pdf_path.resolve() == _default_pdf().resolve():
            return None
    except Exception:
        logger.exception(UI_MODULE, "比较默认文档路径失败，改用文件名作为文档 ID")
    return pdf_path.stem


def _sync_conversation_messages() -> None:
    """把当前消息列表登记到“按会话索引”的历史字典（同一个列表对象，自动同步）。"""
    cid = st.session_state.get("conversation_id") or "__default__"
    st.session_state.messages_by_conv[cid] = st.session_state.messages


def _now_text() -> str:
    """当前时间（界面展示用）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================================
# 4. 引擎句柄：惰性导入 + 缓存 + 全链路容错
# ==========================================================================
@dataclass
class EngineHandle:
    """问答引擎句柄。

    ``engine`` 为 ``None`` 时表示引擎尚未就绪，``error`` 中保存中文错误原因。
    """

    engine: Any = None
    error: str = ""
    warning: str = ""
    doc_id: str | None = None

    @property
    def ready(self) -> bool:
        """引擎是否可用。"""
        return self.engine is not None


def _import_qa_engine() -> Any:
    """惰性导入 ``app.core.qa_engine.QAEngine``。

    单独抽出成函数，便于在 try/except 中捕获“模块不存在 / 依赖未就绪”等问题。
    """
    from app.core.qa_engine import QAEngine  # noqa: PLC0415 - 刻意惰性导入

    return QAEngine


def _engine_index_ready(engine: Any) -> bool | None:
    """通过 ``stats()`` 判断知识库索引是否就绪。

    返回 ``True``/``False``；接口不可用或未提供该字段时返回 ``None``（未知，
    此时调用方应保持原样，不要做任何回退）。
    """
    ok, stats, _ = _call_engine(engine, "stats")
    if not ok or not isinstance(stats, dict):
        return None
    value = stats.get("index_ready", None)
    return None if value is None else bool(value)


@trace
def _warm_up_engine(engine: Any) -> None:
    """预热引擎，把模型加载等一次性开销挪到界面启动阶段。

    背景：本地嵌入模型首次加载需要数秒。若等到用户问第一句话才加载，
    界面上显示的「首字响应时间」会是 5~10 秒，直接违反工单「首字 < 3 秒」
    的验收标准。因此引擎一旦构造成功就立刻预热。

    引擎若未提供 ``warmup``（例如旧版本），静默跳过，不影响功能。
    """
    warmup = getattr(engine, "warmup", None)
    if not callable(warmup):
        return
    try:
        detail = warmup()
        logger.info(UI_MODULE, "引擎预热完成", **({"detail": detail} if isinstance(detail, dict) else {}))
    except Exception:
        logger.exception(UI_MODULE, "引擎预热失败（不影响使用，但首字延迟可能偏高）")


def _create_engine_handle(doc_id: str | None = None) -> EngineHandle:
    """构造引擎句柄；任何失败都转成中文错误信息，不向上抛异常。"""
    try:
        qa_engine_cls = _import_qa_engine()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.exception(UI_MODULE, "导入 app.core.qa_engine 失败", doc_id=doc_id)
        return EngineHandle(
            error=f"问答引擎尚未就绪：无法导入 app.core.qa_engine（{reason}）",
            doc_id=doc_id,
        )

    # 场景一：指定了 doc_id，优先按 doc_id 构造
    if doc_id:
        try:
            engine = qa_engine_cls(doc_id=doc_id)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.exception(UI_MODULE, "按 doc_id 初始化引擎失败，回退默认知识库", doc_id=doc_id)
            try:
                engine = qa_engine_cls()
            except Exception as exc2:
                reason2 = f"{type(exc2).__name__}: {exc2}"
                logger.exception(UI_MODULE, "QAEngine() 初始化失败")
                return EngineHandle(
                    error=(
                        f"问答引擎尚未就绪：初始化失败（{reason2}）。"
                        f"指定的文档 ID「{doc_id}」也不可用（{reason}）。"
                    ),
                    doc_id=doc_id,
                )
            _warm_up_engine(engine)
            return EngineHandle(
                engine=engine,
                warning=f"文档 ID「{doc_id}」无法使用（{reason}），已回退到默认知识库。",
                doc_id=None,
            )

        # 该 doc_id 没有索引时（常见于索引是用自动生成的 doc_id 建立的），
        # 自动回退到引擎的默认知识库，避免整场问答都只能回答“不清楚”。
        if _engine_index_ready(engine) is False:
            logger.warning(UI_MODULE, "该 doc_id 没有索引，尝试回退默认知识库", doc_id=doc_id)
            try:
                fallback = qa_engine_cls()
            except Exception:
                logger.exception(UI_MODULE, "回退默认知识库失败，保留原引擎")
                fallback = None
            if fallback is not None and _engine_index_ready(fallback) is True:
                _warm_up_engine(fallback)
                return EngineHandle(
                    engine=fallback,
                    doc_id=None,
                    warning=f"文档 ID「{doc_id}」尚未建立索引，已自动回退到引擎的默认知识库。",
                )
            return EngineHandle(
                engine=engine,
                doc_id=doc_id,
                warning=f"文档 ID「{doc_id}」尚未建立索引，请先运行 scripts/build_index.py 建立索引。",
            )
        _warm_up_engine(engine)
        return EngineHandle(engine=engine, doc_id=doc_id)

    # 场景二：未指定 doc_id，使用引擎默认知识库
    try:
        engine = qa_engine_cls()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.exception(UI_MODULE, "QAEngine() 初始化失败")
        return EngineHandle(error=f"问答引擎尚未就绪：初始化失败（{reason}）", doc_id=doc_id)

    _warm_up_engine(engine)
    return EngineHandle(engine=engine, doc_id=doc_id)


@cache_resource(show_spinner=False)
def load_engine(doc_id: str | None = None) -> EngineHandle:
    """获取（带缓存的）问答引擎句柄。

    只有真正构造引擎时才写追踪日志；命中缓存时直接返回。
    """
    return _create_engine_handle(doc_id)


@trace
def _call_engine(engine: Any, method: str, *args: Any, **kwargs: Any) -> tuple[bool, Any, str]:
    """安全调用引擎方法。

    返回 ``(是否成功, 返回值, 中文错误信息)``，绝不抛异常到界面。
    """
    if engine is None:
        return False, None, "问答引擎尚未就绪。"
    func = getattr(engine, method, None)
    if func is None or not callable(func):
        return False, None, f"问答引擎未提供 {method}() 接口。"
    try:
        return True, func(*args, **kwargs), ""
    except Exception as exc:
        logger.exception(UI_MODULE, f"调用 QAEngine.{method} 失败", method=method)
        return False, None, f"调用问答引擎 {method}() 失败：{type(exc).__name__}: {exc}"


# ==========================================================================
# 5. 流式事件处理
# ==========================================================================
def _payload_field(payload: Any, name: str, default: Any = None) -> Any:
    """从事件负载中取值，兼容 dict 与对象两种形态。"""
    if payload is None:
        return default
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _normalize_event(item: Any) -> tuple[str, Any]:
    """把引擎产出的单个流式事件规整为 ``(事件名, 负载)``。"""
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]), item[1]
    if isinstance(item, str):  # 宽容处理：直接产出文本的实现
        return "delta", {"text": item}
    event = _payload_field(item, "event", "")
    payload = _payload_field(item, "payload", None)
    if event:
        return str(event), payload
    return "", item


def _iter_engine_events(engine: Any, question: str, conversation_id: str | None) -> Iterator[tuple[str, Any]]:
    """迭代引擎的流式输出，统一为 ``(事件名, 负载)``。

    契约事件：
    - ``first_token`` -> ``{"first_token_ms": float}``
    - ``delta``       -> ``{"text": str}``
    - ``done``        -> ``{"answer": Answer}``

    任何异常（包括生成器中途抛错）都会转成 ``error`` 事件，保证界面不崩溃。
    """
    stream_func = getattr(engine, "stream", None)
    if not callable(stream_func):
        yield "error", {"message": "问答引擎未提供 stream() 流式接口。"}
        return

    try:
        iterator = iter(stream_func(question, conversation_id))
    except Exception as exc:
        logger.exception(UI_MODULE, "启动流式问答失败", question=question)
        yield "error", {"message": f"启动流式问答失败：{type(exc).__name__}: {exc}"}
        return

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception as exc:
            logger.exception(UI_MODULE, "流式问答中断", question=question)
            yield "error", {"message": f"流式问答中断：{type(exc).__name__}: {exc}"}
            return
        event, payload = _normalize_event(item)
        if not event:
            logger.warning(UI_MODULE, "收到无法识别的流式事件，已忽略", item=repr(item)[:200])
            continue
        yield event, payload


# ==========================================================================
# 6. 数据转换：Answer / Citation -> 界面消息字典
# ==========================================================================
def _to_float(value: Any, default: float = 0.0) -> float:
    """宽松转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """宽松转 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_pages(raw: Any) -> list[int]:
    """页码列表：转 int、去重、升序。"""
    pages: set[int] = set()
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            page = _to_int(item, -1)
            if page > 0:
                pages.add(page)
    return sorted(pages)


def _citation_to_dict(citation: Any) -> dict:
    """把 ``Citation`` 对象（或字典）统一成界面用的普通字典。"""
    if isinstance(citation, dict):
        data = dict(citation)
        raw_label = ""
    else:
        data = {
            "page": _payload_field(citation, "page", 0),
            "chunk_id": _payload_field(citation, "chunk_id", ""),
            "snippet": _payload_field(citation, "snippet", ""),
            "section": _payload_field(citation, "section", ""),
            "score": _payload_field(citation, "score", 0.0),
        }
        raw_label = ""
        label_func = getattr(citation, "label", None)
        if callable(label_func):
            try:
                raw_label = str(label_func())
            except Exception:
                logger.exception(UI_MODULE, "调用 Citation.label() 失败")
                raw_label = ""

    page = _to_int(data.get("page", 0), 0)
    label = raw_label or f"[页码: {page}]"
    return {
        "page": page,
        "label": label,
        "chunk_id": str(data.get("chunk_id", "") or ""),
        "snippet": str(data.get("snippet", "") or ""),
        "section": str(data.get("section", "") or ""),
        "score": _to_float(data.get("score", 0.0), 0.0),
    }


def _collect_chunk_texts(answer: Any, citations: list[dict]) -> dict[str, str]:
    """收集被引用片段的完整原文，供“可展开查看原文”使用。

    注意：``Answer.retrieved`` 属于检索中间结果，默认不展示给用户；
    这里**只取出现在引用列表中的 chunk_id 对应的原文**，
    用于满足工单第 5.6 节“前端可展开查看原文”的要求。
    """
    wanted = {cit["chunk_id"] for cit in citations if cit.get("chunk_id")}
    if not wanted:
        return {}
    texts: dict[str, str] = {}
    raw_retrieved = _payload_field(answer, "retrieved", []) or []
    if not isinstance(raw_retrieved, (list, tuple)):
        return {}
    for item in raw_retrieved:
        chunk = _payload_field(item, "chunk", None)
        if chunk is None:
            continue
        chunk_id = str(_payload_field(chunk, "chunk_id", "") or "")
        if chunk_id and chunk_id in wanted:
            content = str(_payload_field(chunk, "content", "") or "")
            if content.strip():
                texts[chunk_id] = content
    return texts


def _build_answer_message(final_answer: Any, buffer: list[str], first_token_ms: float, started: float) -> dict:
    """把引擎返回的 ``Answer``（或流式增量）整理成界面消息字典。"""
    total_ms = (time.perf_counter() - started) * 1000.0
    text = ""
    is_unknown = False
    retrieved_count = 0
    pages: list[int] = []
    citations: list[dict] = []
    chunk_texts: dict[str, str] = {}

    if final_answer is not None:
        text = str(_payload_field(final_answer, "answer", "") or "")
        is_unknown = bool(_payload_field(final_answer, "is_unknown", False))
        total_ms = _to_float(_payload_field(final_answer, "total_ms", 0.0), 0.0) or total_ms
        first_token_ms = (
            _to_float(_payload_field(final_answer, "first_token_ms", 0.0), 0.0) or first_token_ms
        )
        retrieved_count = _to_int(_payload_field(final_answer, "retrieved_count", 0), 0)
        pages = _to_pages(_payload_field(final_answer, "pages", []) or [])
        raw_citations = _payload_field(final_answer, "citations", []) or []
        if isinstance(raw_citations, (list, tuple)):
            citations = [_citation_to_dict(item) for item in raw_citations]
        chunk_texts = _collect_chunk_texts(final_answer, citations)

    # 流式增量兜底：引擎未给出最终 Answer 时，用拼接结果作为答案
    if not text.strip() and buffer:
        text = "".join(buffer)
    # 完全没有内容时按工单要求统一兜底为“不清楚”
    if not text.strip():
        text = _unknown_answer()
        is_unknown = True
    if not pages and citations:
        pages = sorted({cit["page"] for cit in citations if cit["page"] > 0})
    if retrieved_count <= 0 and citations:
        # 引擎未上报片段数时用引用条数兜底，保证“片段数”指标可读
        retrieved_count = len(citations)

    return {
        "role": "assistant",
        "content": text,
        "citations": citations,
        "is_unknown": is_unknown,
        "first_token_ms": float(first_token_ms or 0.0),
        "total_ms": float(total_ms or 0.0),
        "retrieved_count": int(retrieved_count),
        "pages": pages,
        "chunk_texts": chunk_texts,
        "feedback": {},
        "ts": _now_text(),
    }


def answer_question(
    engine: Any,
    question: str,
    conversation_id: str | None,
    placeholder: Any = None,
) -> tuple[dict | None, str]:
    """执行一次问答（流式优先），把过程渲染到 ``placeholder``。

    返回 ``(助手消息字典, 中文错误信息)``；
    消息为 ``None`` 表示本次回答完全失败（错误信息已可直接展示）。
    """
    started = time.perf_counter()
    buffer: list[str] = []
    first_token_ms = 0.0
    final_answer: Any = None
    stream_error = ""
    has_delta = False

    if engine is None:
        return None, "问答引擎尚未就绪，无法回答。"

    logger.info(UI_MODULE, "开始问答", question=question, conversation_id=conversation_id)

    for event, payload in _iter_engine_events(engine, question, conversation_id):
        if event == "first_token":
            first_token_ms = _to_float(_payload_field(payload, "first_token_ms", 0.0), first_token_ms)
        elif event == "delta":
            text = _payload_field(payload, "text", "")
            if text:
                if not has_delta:
                    has_delta = True
                    if first_token_ms <= 0:  # 引擎未显式上报时自行测量首字时间
                        first_token_ms = (time.perf_counter() - started) * 1000.0
                buffer.append(str(text))
                if placeholder is not None:
                    placeholder.markdown("".join(buffer) + " ▌")
        elif event == "done":
            final_answer = _payload_field(payload, "answer", None)
        elif event == "error":
            stream_error = str(_payload_field(payload, "message", "") or "流式问答失败。")
        else:
            # 中间过程（意图识别 / Query 改写等）一律不展示，只记日志
            logger.debug(UI_MODULE, "忽略未知流式事件", event=event)

    if final_answer is None and not has_delta:
        # 流式接口不可用（未实现、报错、或没有任何产出）：退回一次性问答接口
        logger.warning(UI_MODULE, "流式问答不可用，改用 ask() 接口", error=stream_error)
        ok, result, call_error = _call_engine(engine, "ask", question, conversation_id)
        if not ok:
            return None, call_error or stream_error or "问答失败：引擎未返回结果。"
        final_answer = result

    message = _build_answer_message(final_answer, buffer, first_token_ms, started)
    if stream_error and has_delta:
        # 已有部分答案时只记日志，不打断用户阅读
        logger.warning(UI_MODULE, "流式问答提前结束，已展示部分内容", error=stream_error)

    logger.info(
        UI_MODULE,
        "问答完成",
        conversation_id=conversation_id,
        first_token_ms=round(message["first_token_ms"], 1),
        total_ms=round(message["total_ms"], 1),
        is_unknown=message["is_unknown"],
        retrieved_count=message["retrieved_count"],
        citation_count=len(message["citations"]),
        pages=message["pages"],
    )
    return message, ""


# ==========================================================================
# 7. 会话管理
# ==========================================================================
def _ensure_conversation_id(engine: Any) -> str | None:
    """确保存在会话 ID；引擎不支持时返回 ``None``（由引擎自行分配）。"""
    cid = st.session_state.get("conversation_id")
    if cid:
        return str(cid)
    ok, new_cid, err = _call_engine(engine, "new_conversation")
    if ok and isinstance(new_cid, str) and new_cid.strip():
        st.session_state.conversation_id = new_cid
        st.session_state.messages_by_conv.setdefault(new_cid, st.session_state.messages)
        logger.info(UI_MODULE, "已创建会话", conversation_id=new_cid)
        return new_cid
    logger.warning(UI_MODULE, "new_conversation 不可用，交由引擎自动分配会话", error=err)
    return None


def _start_new_conversation(engine: Any, reason: str) -> None:
    """新建会话并切换界面状态。"""
    new_cid: str | None = None
    if engine is not None:
        ok, result, err = _call_engine(engine, "new_conversation")
        if ok and isinstance(result, str) and result.strip():
            new_cid = result
        else:
            logger.warning(UI_MODULE, "新建会话失败，已退回本地空会话", reason=reason, error=err)
    st.session_state.conversation_id = new_cid
    st.session_state.messages = []
    if new_cid:
        st.session_state.messages_by_conv[new_cid] = st.session_state.messages
    logger.info(UI_MODULE, "已新建对话", conversation_id=new_cid or "(未分配)", reason=reason)


def _clear_conversation(engine: Any) -> str:
    """清空当前会话（引擎侧 + 界面侧），返回中文错误信息（空串表示成功）。"""
    cid = st.session_state.get("conversation_id")
    error = ""
    if engine is not None and cid:
        ok, _, err = _call_engine(engine, "clear_conversation", cid)
        if not ok:
            error = err
    st.session_state.messages = []
    if cid:
        st.session_state.messages_by_conv[cid] = st.session_state.messages
    logger.info(UI_MODULE, "已清空对话", conversation_id=cid or "(未分配)", error=error)
    return error


def _conversation_options(engine: Any) -> list[tuple[str, str, int]]:
    """读取会话列表，返回 ``[(显示标签, conversation_id, 消息数)]``。"""
    if engine is None:
        return []
    ok, conversations, err = _call_engine(engine, "list_conversations")
    if not ok or not isinstance(conversations, (list, tuple)):
        logger.warning(UI_MODULE, "会话列表获取失败", error=err)
        return []
    options: list[tuple[str, str, int]] = []
    meta: dict[str, int] = {}
    for position, conv in enumerate(conversations):
        cid = str(_payload_field(conv, "conversation_id", "") or "")
        if not cid:
            continue
        title = str(_payload_field(conv, "title", "") or "未命名对话")
        count = _to_int(_payload_field(conv, "message_count", 0), 0)
        # 标签必须**唯一**：前缀序号 + 会话 ID 尾部，避免同名会话互相覆盖
        # （会话 ID 形如 conv_xxxxxxxxxxxx，若只取前 6 位则全部是 "conv_x"，会冲突）
        suffix = f"（{count} 条消息）" if count else ""
        options.append((f"{len(options) + 1}. {title}{suffix} · {cid[-6:]}", cid, count))
        meta[cid] = count
    st.session_state.conv_meta = meta
    return options


def _switch_conversation(engine: Any, cid: str) -> str:
    """切换会话，返回中文错误信息（空串表示成功）。"""
    error = ""
    if engine is not None:
        ok, _, err = _call_engine(engine, "switch_conversation", cid)
        if not ok:
            error = err
    st.session_state.conversation_id = cid
    st.session_state.messages = st.session_state.messages_by_conv.setdefault(cid, [])
    logger.info(UI_MODULE, "已切换对话", conversation_id=cid, error=error)
    return error


# ==========================================================================
# 8. 界面渲染：侧边栏
# ==========================================================================
def _list_pdfs() -> list[Path]:
    """列出可选 PDF：默认文档优先，其次是 ``data/raw`` 下的全部 PDF。"""
    found: list[Path] = []
    default = _default_pdf()
    try:
        if default.exists():
            found.append(default)
        raw_dir = _raw_dir()
        if raw_dir.is_dir():
            for path in sorted(raw_dir.glob("*.pdf")):
                if path.is_file() and path.resolve() not in {item.resolve() for item in found}:
                    found.append(path)
    except Exception:
        logger.exception(UI_MODULE, "扫描 data/raw 下的 PDF 失败")
    current = st.session_state.get("pdf_path")
    if current:
        current_path = Path(str(current))
        if current_path.exists() and current_path.resolve() not in {item.resolve() for item in found}:
            found.append(current_path)
    return found


def _display_path(path: Path) -> str:
    """把绝对路径转成相对项目根目录的展示形式。"""
    try:
        return str(path.relative_to(_path_setting("project_root", _PROJECT_ROOT)))
    except Exception:
        return str(path)


def _select_document(engine: Any, pdf_path: str) -> None:
    """切换当前文档：更新状态并开启新会话。"""
    path = Path(pdf_path)
    st.session_state.pdf_path = str(path)
    st.session_state.doc_id = _doc_id_for(path)
    logger.info(UI_MODULE, "切换当前文档", pdf=str(path), doc_id=st.session_state.doc_id)
    _start_new_conversation(engine, reason="切换文档")


def _render_document_section(engine: Any) -> None:
    """侧边栏：PDF 文件选择 / 上传、当前文档信息。"""
    sidebar = st.sidebar
    sidebar.subheader("📁 文档")

    pdfs = _list_pdfs()
    options = [str(path) for path in pdfs]
    current = st.session_state.get("pdf_path")
    if options:
        index = options.index(current) if current in options else 0
        picked = sidebar.selectbox(
            "选择 PDF 文档",
            options,
            index=index,
            format_func=lambda value: Path(value).name,
            help="默认加载《招股说明书1.pdf》；其余文件来自 data/raw 目录。",
        )
        if picked != current:
            _select_document(engine, picked)
            st.rerun()
    else:
        sidebar.warning(f"未找到 PDF 文件。请上传，或把《招股说明书1.pdf》放到：{_display_path(_default_pdf())}")

    uploaded = sidebar.file_uploader("上传 PDF（可选）", type=["pdf"], key="pdf_uploader")
    if uploaded is not None:
        signature = (uploaded.name, getattr(uploaded, "size", 0))
        if st.session_state.get("_last_upload") != signature:
            try:
                raw_dir = _raw_dir()
                raw_dir.mkdir(parents=True, exist_ok=True)
                target = raw_dir / Path(uploaded.name).name
                target.write_bytes(uploaded.getbuffer())
                st.session_state["_last_upload"] = signature
                logger.info(UI_MODULE, "已保存上传的 PDF", path=str(target), size=signature[1])
                _select_document(engine, str(target))
                st.rerun()
            except Exception as exc:
                logger.exception(UI_MODULE, "保存上传的 PDF 失败")
                sidebar.error(f"保存上传文件失败：{type(exc).__name__}: {exc}")

    sidebar.text_input(
        "文档 ID（选填）",
        key="doc_id_override",
        placeholder="留空则使用 PDF 文件名",
        help="如果问答引擎要求特定 doc_id，可在此填写；留空时使用 PDF 文件名（不含扩展名）。",
    )
    if str(st.session_state.get("doc_id_override") or "").strip():
        if sidebar.button("应用文档 ID", key="apply_doc_id", use_container_width=True):
            pdf_path = st.session_state.get("pdf_path")
            st.session_state.doc_id = str(st.session_state["doc_id_override"]).strip()
            if pdf_path:
                _select_document(engine, str(pdf_path))
            st.rerun()

    # 当前文档信息
    pdf_path = st.session_state.get("pdf_path")
    doc_id = st.session_state.get("doc_id")
    if pdf_path:
        path = Path(str(pdf_path))
        size_text = "—"
        try:
            if path.exists():
                size_text = f"{path.stat().st_size / 1024 / 1024:.2f} MB"
        except Exception:
            logger.exception(UI_MODULE, "读取 PDF 文件信息失败")
        sidebar.markdown("**当前文档**")
        sidebar.caption(f"文件：{path.name}")
        sidebar.caption(f"路径：{_display_path(path)}")
        sidebar.caption(f"大小：{size_text}")
        sidebar.caption(f"文档 ID：{doc_id or '（引擎默认）'}")
    else:
        sidebar.markdown("**当前文档**：未选择（将使用问答引擎的默认知识库）")


def _render_stats_section(engine: Any, handle: "EngineHandle") -> None:
    """侧边栏：知识库统计（``engine.stats()``）与索引状态。"""
    sidebar = st.sidebar
    sidebar.subheader("📊 知识库统计")
    if not handle.ready:
        sidebar.caption("问答引擎尚未就绪，暂无统计信息。")
        return
    ok, stats, err = _call_engine(engine, "stats")
    if not ok:
        sidebar.warning(err)
        return
    if not isinstance(stats, dict) or not stats:
        sidebar.caption("引擎未返回统计数据。")
        return

    index_ready = stats.get("index_ready", None)
    if index_ready is False:
        sidebar.error("当前知识库尚未建立索引，问答只会回复“不清楚”。请先运行 scripts/build_index.py 建索引。")
    elif index_ready:
        sidebar.success("索引已就绪")

    numeric = [
        (str(key), value)
        for key, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if numeric:
        columns = sidebar.columns(2)
        for position, (key, value) in enumerate(numeric[:6]):
            columns[position % 2].metric(STAT_LABELS.get(key, key), value)
    with sidebar.expander("查看完整统计信息", expanded=False):
        try:
            st.json(stats)
        except Exception:
            logger.exception(UI_MODULE, "渲染 stats() 结果失败，改用文本展示")
            st.write(stats)


def _render_conversation_section(engine: Any, handle: "EngineHandle") -> None:
    """侧边栏：新建 / 切换 / 清空对话（工单 5.7）。"""
    sidebar = st.sidebar
    sidebar.subheader("💬 对话")
    disabled = not handle.ready

    if sidebar.button("➕ 新建对话", key="btn_new_conversation", use_container_width=True, disabled=disabled):
        _start_new_conversation(engine, reason="用户点击新建对话")
        st.rerun()

    if sidebar.button("🗑 清空当前对话", key="btn_clear_conversation", use_container_width=True, disabled=disabled):
        error = _clear_conversation(engine)
        if error:
            sidebar.warning(error)
        st.rerun()

    options = _conversation_options(engine)
    if not options:
        sidebar.caption("暂无历史会话（引擎未就绪或尚未创建会话）。")
        return

    # 组装下拉选项：必要时补一个“占位项”，避免首次加载就自动切换到某个历史会话
    current = st.session_state.get("conversation_id")
    known = {cid for _, cid, _ in options}
    labels = [label for label, _, _ in options]
    label_to_cid = {label: cid for label, cid, _ in options}
    if current and current not in known:
        placeholder = f"（当前会话 {current[:8]}）"
        labels.insert(0, placeholder)
        label_to_cid[placeholder] = current
    elif not current:
        placeholder = "（未选择历史会话）"
        labels.insert(0, placeholder)
        label_to_cid[placeholder] = ""

    index = next((position for position, label in enumerate(labels) if label_to_cid[label] == current), 0)
    # 不设置 key：切换会话后由 index 驱动显示，避免组件状态与当前会话不一致
    picked = sidebar.selectbox("切换历史对话", labels, index=index)
    picked_cid = label_to_cid.get(picked) or ""
    logger.debug(
        UI_MODULE,
        "会话下拉框",
        current=current,
        index=index,
        picked=picked,
        picked_cid=picked_cid,
        option_count=len(labels),
    )
    if picked_cid and picked_cid != current:
        error = _switch_conversation(engine, picked_cid)
        if error:
            sidebar.warning(error)
        st.rerun()
    sidebar.caption(f"当前会话：{(current or '未分配')[:8]}")


def _render_fallback_section() -> None:
    """侧边栏：“不清楚”兜底说明。"""
    unknown = _unknown_answer()
    budget = _first_token_budget_seconds()
    with st.sidebar.expander("ℹ️ 关于“不清楚”兜底", expanded=False):
        st.markdown(
            f"""
- 回答**只依据**《招股说明书1.pdf》的原文片段生成，并附引用页码。
- 检索不到相关内容、或原文中没有答案时，统一回复 **“{unknown}”**，绝不编造。
- 答案中形如 `[页码: 129]` 的标记可直接对应到下方“引用来源”。
- 首字响应时间预算为 **{budget:.1f} 秒**，超出时界面会给出醒目提示。
- 界面上只展示最终答案与引用；意图识别、Query 改写等中间过程仅写入日志。
"""
        )


def _render_demo_section(handle: "EngineHandle") -> None:
    """侧边栏：工单固定的 10 个演示问题，一键提问。

    中英文各一组（工单追加要求支持英文问答）：英文提问会走
    ``app/core/language.py`` 的语言桥接，给出英文回答与 ``[Page: N]`` 引用。
    """
    with st.sidebar.expander("🎯 演示问题（一键提问）", expanded=False):
        language = st.radio(
            "问题语言",
            options=["中文", "English"],
            horizontal=True,
            key="demo_language",
            help="英文问题会用英文作答，并给出 [Page: N] 形式的引用。",
        )
        questions = DEMO_QUESTIONS if language == "中文" else DEMO_QUESTIONS_EN
        for position, question in enumerate(questions):
            if st.button(
                question if len(question) <= 30 else question[:30] + "…",
                key=f"demo_question_{'zh' if language == '中文' else 'en'}_{position}",
                help=question,
                use_container_width=True,
                disabled=not handle.ready,
            ):
                st.session_state.pending_question = question
                st.rerun()


def _render_sidebar(engine: Any, handle: "EngineHandle") -> None:
    """渲染整个侧边栏。"""
    sidebar = st.sidebar
    sidebar.title("📄 " + _project_name())
    sidebar.caption(f"版本 {_project_version()} · Streamlit 网页界面")

    if handle.warning:
        sidebar.warning(handle.warning)
    if not handle.ready:
        sidebar.error("问答引擎尚未就绪")

    _render_document_section(engine)
    sidebar.divider()
    _render_stats_section(engine, handle)
    sidebar.divider()
    _render_conversation_section(engine, handle)
    sidebar.divider()
    _render_fallback_section()
    _render_demo_section(handle)
    sidebar.divider()
    # 引擎未就绪时，主区域已有同样的按钮（附带原因说明），此处不再重复
    if handle.ready and sidebar.button("🔄 重新初始化问答引擎", key="btn_reload_engine", use_container_width=True):
        _clear_engine_cache()
        st.rerun()
    sidebar.caption(f"日志目录：{_display_path(_path_setting('logs', _PROJECT_ROOT / 'logs'))}")


# ==========================================================================
# 9. 界面渲染：主区域
# ==========================================================================
def _render_header(engine: Any, handle: "EngineHandle") -> None:
    """主区域标题与当前文档摘要（引擎告警统一显示在侧边栏，避免重复）。"""
    st.title("📄 " + _project_name())
    pdf_path = st.session_state.get("pdf_path")
    doc_text = Path(str(pdf_path)).name if pdf_path else "引擎默认知识库"
    st.caption(f"基于 PDF 文档的检索增强问答 · 当前文档：{doc_text} · 答案均来自原文并附带引用页码")


def _render_engine_error(handle: "EngineHandle") -> None:
    """引擎不可用时的中文错误提示与处理建议（保证界面不白屏）。"""
    st.error(f"⚠️ {handle.error}")
    with st.expander("可能的原因与处理办法", expanded=False):
        st.markdown(
            """
1. `app/core/qa_engine.py` 尚未完成或导入失败 —— 确认该文件已实现 `QAEngine`。
2. 依赖未就绪（向量库 / 向量模型 / 向量索引文件缺失）—— 先运行索引构建脚本。
3. 大模型服务未启动（vLLM / SGLang 的 OpenAI 兼容接口）—— 启动后再点击下方按钮。
4. 若为环境依赖缺失，请按 `requirements.txt` 安装后重启 Streamlit。
"""
        )
        if _LOGGING_IMPORT_ERROR:
            st.caption(f"日志模块降级原因：{_LOGGING_IMPORT_ERROR}")
        if _SETTINGS_IMPORT_ERROR:
            st.caption(f"配置模块降级原因：{_SETTINGS_IMPORT_ERROR}")
    if st.button("🔄 重新初始化问答引擎", key="btn_retry_engine"):
        _clear_engine_cache()
        st.rerun()


def _render_metrics(message: dict) -> None:
    """答案指标：首字响应时间（毫秒 / 秒，对比预算）、总耗时、片段数、页码。"""
    budget_ms = _first_token_budget_seconds() * 1000.0
    first_token_ms = _to_float(message.get("first_token_ms"), 0.0)
    total_ms = _to_float(message.get("total_ms"), 0.0)
    retrieved_count = _to_int(message.get("retrieved_count"), 0)
    pages = message.get("pages") or []
    citations = message.get("citations") or []

    columns = st.columns(4)
    if first_token_ms > 0:
        columns[0].metric("首字响应", f"{first_token_ms:.0f} ms", f"{first_token_ms / 1000:.2f} s", delta_color="off")
    else:
        columns[0].metric("首字响应", "未记录")
    columns[1].metric("总耗时", f"{total_ms / 1000:.2f} s" if total_ms > 0 else "未记录")
    columns[2].metric("检索片段", f"{retrieved_count} 个")
    columns[3].metric("命中页码", f"{len(pages)} 页")

    if first_token_ms > 0:
        if first_token_ms > budget_ms:
            st.error(
                f"⏱ 首字响应 {first_token_ms:.0f} 毫秒（{first_token_ms / 1000:.2f} 秒），"
                f"已超出 {budget_ms / 1000:.1f} 秒预算！"
            )
        else:
            st.caption(
                f"✅ 首字响应 {first_token_ms:.0f} 毫秒（{first_token_ms / 1000:.2f} 秒），"
                f"满足 {budget_ms / 1000:.1f} 秒预算。"
            )
    if pages:
        st.caption("命中页码：" + "、".join(f"第 {page} 页" for page in pages))
    if citations:
        st.caption(f"共检索到 {retrieved_count} 个相关片段，其中 {len(citations)} 条作为引用来源。")


def _render_citations(message: dict, key_prefix: str) -> None:
    """引用来源展示区：页码 + 摘要 + chunk_id，可展开查看原文（工单 5.6）。"""
    citations = message.get("citations") or []
    if not citations:
        if message.get("is_unknown"):
            st.info(f"本次为“{_unknown_answer()}”兜底回复：文档中没有检索到可支撑该问题的内容。")
        else:
            st.caption("本次回答没有可展示的引用来源。")
        return

    chunk_texts = message.get("chunk_texts") or {}
    st.markdown(f"**📚 引用来源（{len(citations)} 条）**")
    for position, citation in enumerate(citations):
        section = str(citation.get("section") or "未标注章节")
        title = f"{citation.get('label', '')}　{section[:40]}　相关度 {_to_float(citation.get('score'), 0.0):.3f}"
        with st.expander(title, expanded=False):
            st.caption(f"片段 ID：{citation.get('chunk_id') or '—'}")
            st.markdown("**原文片段**")
            snippet = str(citation.get("snippet") or "")
            if snippet.strip():
                with st.container(border=True):
                    st.markdown(snippet)
            else:
                st.caption("（引擎未提供片段摘要）")
            full_text = str(chunk_texts.get(citation.get("chunk_id"), "") or "")
            if full_text.strip() and full_text.strip() != snippet.strip():
                if st.checkbox("查看完整片段原文", key=f"fulltext-{key_prefix}-{position}"):
                    with st.container(border=True):
                        st.markdown(full_text)


def _question_before(index: int) -> str:
    """取该条回答之前最近的一条用户提问（反馈记录用）。"""
    messages = st.session_state.get("messages") or []
    for position in range(min(index, len(messages)) - 1, -1, -1):
        if messages[position].get("role") == "user":
            return str(messages[position].get("content", ""))
    return ""


@trace
def _save_feedback(
    engine: Any,
    conversation_id: str,
    rating: str,
    comment: str,
    question: str,
) -> tuple[bool, str]:
    """提交用户反馈（点赞 / 点踩 / 评论），返回 ``(是否成功, 中文错误信息)``。"""
    if engine is None:
        return False, "问答引擎尚未就绪，无法提交反馈。"
    # 契约：submit_feedback(conversation_id, message_id, rating, comment="", question="")
    # 界面层拿不到引擎内部的消息自增 ID，故 message_id 传 None，并用 question 兜底关联。
    ok, feedback_id, err = _call_engine(
        engine, "submit_feedback", conversation_id, None, rating, comment, question
    )
    if not ok:
        return False, err
    logger.info(
        UI_MODULE,
        "用户反馈已提交",
        conversation_id=conversation_id,
        rating=rating,
        has_comment=bool(comment),
        feedback_id=feedback_id,
    )
    return True, ""


def _render_feedback(message: dict, index: int, engine: Any, is_last: bool) -> None:
    """点赞 / 点踩按钮 + 评论输入（工单 5.11 用户反馈）。"""
    conversation_id = str(st.session_state.get("conversation_id") or "")
    question = _question_before(index)
    feedback = message.get("feedback") or {}
    rating = feedback.get("rating")
    key_base = f"{conversation_id or 'default'}-{index}"
    columns = st.columns([1, 1, 1, 4])

    if not rating:
        clicked = ""
        if columns[0].button("👍 有帮助", key=f"up-{key_base}"):
            clicked = "up"
        if columns[1].button("👎 没帮助", key=f"down-{key_base}"):
            clicked = "down"
        if clicked:
            ok, error = _save_feedback(engine, conversation_id, clicked, "", question)
            if ok:
                message["feedback"] = {"rating": clicked, "comment": ""}
            else:
                st.warning(error)
            st.rerun()
    else:
        columns[0].caption("👍 已赞" if rating == "up" else "👎 已踩")

    if rating or is_last:
        comment_columns = st.columns([4, 1])
        comment = comment_columns[0].text_input(
            "补充评论（选填）",
            key=f"comment-{key_base}",
            placeholder="例如：答案与原文不符 / 缺少页码 / 检索不到内容",
            label_visibility="collapsed",
        )
        if comment_columns[1].button("提交评论", key=f"comment-submit-{key_base}"):
            if not rating:
                st.warning("请先点击 👍 或 👎，再提交评论。")
            elif not str(comment).strip():
                st.warning("评论内容为空，请输入后再提交。")
            else:
                ok, error = _save_feedback(engine, conversation_id, rating, str(comment).strip(), question)
                if ok:
                    message["feedback"] = {"rating": rating, "comment": str(comment).strip()}
                    st.success("评论已提交，感谢反馈！")
                else:
                    st.warning(error)
                st.rerun()
        if feedback.get("comment"):
            st.caption(f"已提交的评论：{feedback['comment']}")


def _render_assistant_message(message: dict, index: int, engine: Any, key_prefix: str, is_last: bool) -> None:
    """渲染一条助手回答：正文 + 指标 + 引用 + 反馈。"""
    st.markdown(str(message.get("content", "")))
    if message.get("is_unknown"):
        st.caption(f"（兜底回复：文档中未找到可支撑该问题的内容，按约定回复“{_unknown_answer()}”）")
    _render_metrics(message)
    _render_citations(message, key_prefix)
    _render_feedback(message, index, engine, is_last)


def _render_history(engine: Any) -> None:
    """渲染多轮对话历史（工单 5.7）。"""
    messages = st.session_state.get("messages") or []
    last_index = len(messages) - 1
    for index, message in enumerate(messages):
        role = str(message.get("role") or "assistant")
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(str(message.get("content", "")))
        else:
            with st.chat_message("assistant", avatar="🤖"):
                _render_assistant_message(
                    message, index, engine, key_prefix=f"hist-{index}", is_last=index == last_index
                )

    cid = st.session_state.get("conversation_id")
    if not messages and cid:
        engine_count = _to_int((st.session_state.get("conv_meta") or {}).get(cid, 0), 0)
        if engine_count > 0:
            st.info(
                f"该会话在问答引擎中已有 {engine_count} 条历史消息。"
                "引擎接口未提供消息回放能力，此处仅显示本次打开界面后产生的问答。"
            )


def _render_voice_input(engine_ready: bool) -> None:
    """渲染语音输入控件（工单追加要求）。

    兼容策略：
    - Streamlit ≥ 1.40 提供 ``st.audio_input``（浏览器内直接录音）；
    - 更低版本（本机是 1.37）退化为 ``st.file_uploader`` 上传音频文件，
      功能等价，只是需要用户先录好再上传；
    - 两条路径都会调用 ``app.core.asr`` 转写，转写文本进入同一个提问流程。

    转写失败一律给出可读的中文原因（例如未部署 Whisper 服务），
    绝不让用户面对“点了没反应”的界面。
    """
    with st.expander("🎤 语音输入", expanded=False):
        recognizer = _get_speech_recognizer()
        if recognizer is None:
            st.info("语音模块未就绪（app/core/asr.py 不可用），请使用文字提问。")
            return
        health = _safe_health(recognizer)
        if health.get("backend") == "none":
            st.warning(
                "当前没有可用的语音识别后端。请在算力云上启动 Whisper 服务"
                "（`bash scripts/run_whisper.sh`），或把 Whisper 模型放到 `models/` 目录。"
            )
        else:
            st.caption(
                f"识别后端：**{health.get('backend')}**"
                + (f"（本地模型：{Path(str(health.get('local_model'))).name}）" if health.get("local_model") else "")
            )

        audio = _capture_audio(engine_ready)
        if audio is None:
            st.caption("点击上方控件录音或上传音频；录音结束后会自动转写并填入提问框。")
            return

        with st.spinner("正在识别语音…"):
            result, error = _transcribe_audio(recognizer, audio)
        if error:
            st.error(f"⚠️ 语音识别失败：{error}")
            return
        text = str(result.get("text", "")).strip()
        if not text:
            st.error("⚠️ 没有识别出文字，请重新录制（靠近麦克风、减少背景噪声）。")
            return
        st.success(f"识别结果（{result.get('language') or '自动'}）：{text}")
        st.session_state["pending_question"] = text
        # 立即重跑，让识别结果直接进入正常提问流程
        st.rerun()


def _capture_audio(engine_ready: bool):
    """获取音频对象：优先录音控件，其次文件上传。"""
    if hasattr(st, "audio_input"):
        try:
            return st.audio_input("点击录音", disabled=not engine_ready)
        except Exception:
            logger.exception(UI_MODULE, "st.audio_input 不可用，改用文件上传")
    return st.file_uploader(
        "上传音频文件（wav / mp3 / m4a）",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        disabled=not engine_ready,
        key="voice_upload",
    )


def _get_speech_recognizer():
    """惰性导入语音识别器；不可用时返回 None（不抛异常）。"""
    if "speech_recognizer" in st.session_state:
        return st.session_state["speech_recognizer"]
    recognizer = None
    try:
        from app.core.asr import get_speech_recognizer  # noqa: PLC0415

        recognizer = get_speech_recognizer()
    except Exception:
        logger.exception(UI_MODULE, "导入 app.core.asr 失败，语音输入不可用")
    st.session_state["speech_recognizer"] = recognizer
    return recognizer


def _safe_health(recognizer: Any) -> dict:
    """安全读取健康状态（任何异常都退化为空字典）。"""
    try:
        return dict(recognizer.health())
    except Exception:
        logger.exception(UI_MODULE, "读取语音模块健康状态失败")
        return {}


def _transcribe_audio(recognizer: Any, audio: Any) -> tuple[dict, str]:
    """把音频转成文字，返回 ``(结果, 错误信息)``。"""
    try:
        payload = audio.getvalue() if hasattr(audio, "getvalue") else audio
        name = getattr(audio, "name", "audio.wav") or "audio.wav"
        result = recognizer.transcribe(payload)
        logger.info(UI_MODULE, "语音转写成功", chars=len(str(result.get("text", ""))), file=name)
        return dict(result), ""
    except Exception as exc:
        reason = str(exc) or f"{type(exc).__name__}"
        logger.exception(UI_MODULE, "语音转写失败")
        return {}, reason


def _read_prompt(engine_ready: bool) -> str:
    """读取用户提问：优先输入框，其次是侧边栏“演示问题”按钮。"""
    typed = st.chat_input(
        "请输入关于招股说明书的问题（支持中文/English），例如：注册资本是多少？",
        disabled=not engine_ready,
    )
    pending = st.session_state.pop("pending_question", None)
    if typed and str(typed).strip():
        return str(typed).strip()
    if pending and str(pending).strip():
        return str(pending).strip()
    return ""


def _handle_prompt(engine: Any, question: str) -> None:
    """处理一次用户提问：展示提问 -> 流式回答 -> 指标 / 引用 / 反馈。"""
    if engine is None:
        st.error("问答引擎尚未就绪，暂时无法回答，请先按上方提示修复后重试。")
        return

    conversation_id = _ensure_conversation_id(engine)
    logger.info(UI_MODULE, "收到用户提问", question=question, conversation_id=conversation_id)

    user_message = {"role": "user", "content": question, "ts": _now_text()}
    st.session_state.messages.append(user_message)
    _sync_conversation_messages()
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_正在检索文档并生成回答…_")
        message, error = answer_question(engine, question, conversation_id, placeholder)
        if message is None:
            placeholder.empty()
            st.error(f"⚠️ {error or '问答失败，请稍后重试。'}")
            return
        st.session_state.messages.append(message)
        _sync_conversation_messages()
        message_index = len(st.session_state.messages) - 1
        placeholder.markdown(str(message["content"]))
        _render_metrics(message)
        _render_citations(message, key_prefix=f"live-{message_index}")
        _render_feedback(message, message_index, engine, is_last=True)


# ==========================================================================
# 10. 入口
# ==========================================================================
DEMO_QUESTIONS: list[str] = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
]

# 英文演示问题（工单追加要求：支持英文问答）
DEMO_QUESTIONS_EN: list[str] = [
    "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
    "Who is the legal representative?",
    "Which technical standard did the company participate in formulating?",
    "What percentage of main business revenue came from the military sector during the reporting period?",
    "How much revenue did the company generate from the military sector during the reporting period?",
    "What are the upstream and downstream of the electronic information industry?",
    "In which field has the company become a major supplier?",
    "Which project won the First Prize of the National Science and Technology Progress Award?",
    "How much of the raised funds will be used to supplement working capital?",
    "What is the company's main business?",
]


@trace
def main() -> int:
    """界面主入口。返回进程退出码（0 表示正常）。"""
    if not HAS_STREAMLIT:
        message = (
            "无法启动网页界面：未安装 streamlit（"
            f"{STREAMLIT_IMPORT_ERROR}）。请先执行：pip install -r requirements.txt"
        )
        print(message, file=sys.stderr)
        return 1

    st.set_page_config(page_title=_project_name(), page_icon="📄", layout="wide")
    logger.info(UI_MODULE, "Streamlit 界面启动", project_root=str(_PROJECT_ROOT))
    _init_state()

    handle = load_engine(st.session_state.get("doc_id"))
    engine = handle.engine
    if handle.ready:
        logger.info(UI_MODULE, "问答引擎就绪", doc_id=handle.doc_id)

    _render_sidebar(engine, handle)
    _render_header(engine, handle)
    if not handle.ready:
        _render_engine_error(handle)
    _render_history(engine)

    # 语音输入（工单追加要求）。放在提问框之前，
    # 识别结果通过 session_state["pending_question"] 进入同一条提问流程。
    _render_voice_input(handle.ready)

    prompt = _read_prompt(handle.ready)
    if prompt:
        _handle_prompt(engine, prompt)
    return 0


# ``streamlit run`` 与 ``python -m app.ui.streamlit_app`` 都会以 __name__ == "__main__" 执行本文件
if __name__ == "__main__":
    _exit_code = main()
    # 在 Streamlit 运行时中只能直接返回；仅当以 ``python -m`` 直接运行且返回非 0 时，
    # 才用退出码告知调用方（例如缺少 streamlit 依赖）。
    if _exit_code and not _has_streamlit_runtime():
        raise SystemExit(_exit_code)
