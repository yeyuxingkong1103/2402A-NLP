# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-Query 理解优化
# 文件：部署/界面/gradio_app.py —— 首选多轮对话界面（Gradio Blocks，默认 127.0.0.1:7860）。
"""部署/界面/gradio_app.py —— 多轮对话「首选」界面（Gradio Blocks，默认端口 7860）。

工单编号：人工智能NLP-RAG-Query 理解优化

界面能力（对照任务书「部署契约」）：
    1. **多轮对话**：``gr.Chatbot`` + ``gr.State`` 持有 ``session_id``（uuid），
       服务端按会话维护焦点实体 / 槽位 / 历史；
    2. **改写结果可见**：独立 Markdown 面板展示 ``rewritten_query`` / ``strategy`` /
       ``focus_entity`` / 指代消解明细 —— 本工单的核心卖点；
    3. **引用页码 + 响应耗时（ms）**：逐轮展示 ``cited_pages`` 与
       ``latency_ms.{rewrite,retrieve,generate,total}``；
    4. **用户反馈**：每轮回答下「👍 有帮助 / 👎 没帮助」，追加写
       ``部署/日志/用户反馈.jsonl``（每行一个 JSON）；
    5. **重新生成**：对最后一轮用 ``use_cache=False`` 重跑；
    6. **清空对话**：调用 ``engine.reset(session_id)`` 并换新会话。

依赖与降级（**重要**）
----------------------
本机唯一可用解释器**没有装 gradio**，且红线禁止裸 ``pip install``。因此：

* gradio 缺失 → 捕获 ``ImportError``，打印中文安装指引并 ``sys.exit(3)``（不抛栈）；
* ``MultiTurnEngine`` 缺失（研发侧并行开发中）→ 界面**照常启动**，每次提问在页面上
  如实显示「引擎未就绪」与原因，不返回假答案；
* 两者都用**函数内延迟导入**，本文件在零依赖环境下也能被 ``ast.parse`` / ``import``；
* 真起不来时的兜底：``部署/界面/web_app.py``（纯标准库，端口 8123，必可用）。

启动（推荐走统一入口脚本）::

    pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860
    # 等价于：
    pwsh -NoProfile -File run_py.ps1 部署/界面/gradio_app.py --host 127.0.0.1 --port 7860

退出码：``0`` 正常退出 / ``2`` 端口占用或参数错误 / ``3`` 依赖缺失（gradio 不可用）。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"
VERSION = "1.0.0"
SERVICE = "部署/界面/gradio_app.py（Gradio 首选界面）"
TITLE = "招股说明书多轮对话问答（Query 理解优化 · Gradio 界面）"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
EXIT_OK, EXIT_PARAM, EXIT_DEPS = 0, 2, 3

#: 多轮演示问题：覆盖 pronoun / ellipsis_entity_switch / topic_carry 三条改写规则
DEMO_QUESTIONS: list[str] = [
    "武汉兴图新科电子股份有限公司的注册资本是多少？",
    "它的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "法定代表人是谁？",
]


# --------------------------------------------------------------------------- 路径
def _find_repo_root(start: Path) -> Path:
    """向上找仓库根（同时具备 run_py.ps1 与 研发/app 的目录），失败则退回 ``parents[2]``。"""
    for cand in (start, *start.parents):
        if (cand / "run_py.ps1").is_file() and (cand / "研发" / "app").is_dir():
            return cand
    return start.parents[2] if len(start.parents) >= 3 else start


REPO_ROOT: Path = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT / "研发") not in sys.path:                      # 与 研发/app/main.py 同一约定
    sys.path.insert(0, str(REPO_ROOT / "研发"))

#: 用户反馈落点（契约：每行一个 JSON）
FEEDBACK_PATH: Path = REPO_ROOT / "部署" / "日志" / "用户反馈.jsonl"


# --------------------------------------------------------------------------- 引擎（进程内单例）
_ENGINE_LOCK = threading.Lock()
_ENGINE: dict[str, Any] = {"done": False, "engine": None, "meta": {}}


def _multiturn_guidance(exc: BaseException) -> str:
    """缺模块时的中文指引（现场排障用）。"""
    return (
        f"多轮引擎不可用：无法 import 研发/app/multiturn_engine.py "
        f"（{type(exc).__name__}: {exc}）。\n"
        "  · 常见原因：研发/app/multiturn_engine.py 或它依赖的 "
        "研发/app/core/dialogue_state.py、研发/app/core/query_rewriter.py 尚未就绪。\n"
        "  · 自检：pwsh -NoProfile -File run_py.ps1 -c \"import app.multiturn_engine as m; print(m.__file__)\"\n"
        "  · 引擎就绪后重启本界面即可；界面代码无需改动。"
    )


def _load_multiturn_engine_class() -> Any:
    """**延迟导入**核心引擎（绝不在模块顶层裸 import，保证本文件永远可解析/可导入）。"""
    try:
        from app.multiturn_engine import MultiTurnEngine          # noqa: PLC0415 —— 刻意延迟
    except ImportError as exc:
        raise RuntimeError(_multiturn_guidance(exc)) from exc
    return MultiTurnEngine


def build_engine(*, warmup: bool = True) -> tuple[Any, dict[str, Any]]:
    """构造（并缓存）``MultiTurnEngine`` 单例；启动阶段打印 warmup 实测耗时。"""
    with _ENGINE_LOCK:
        if _ENGINE["done"]:
            return _ENGINE["engine"], _ENGINE["meta"]
        meta: dict[str, Any] = {"ready": False, "error": None, "warmup": None}
        try:
            engine_cls = _load_multiturn_engine_class()
            from app.core.config import describe_runtime, get_config   # noqa: PLC0415
            from app.core.logging_conf import setup_logging            # noqa: PLC0415
            from app.storage.sqlite_manager import SqliteManager       # noqa: PLC0415

            cfg = get_config()
            try:
                log: Any = setup_logging(cfg)
            except Exception as exc:  # noqa: BLE001 —— 日志失败不阻断界面
                log = None
                print(f"[警告] 日志初始化失败（界面继续，仅少落盘日志）：{type(exc).__name__}: {exc}",
                      flush=True)
            storage = None
            try:
                storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
            except Exception as exc:  # noqa: BLE001 —— 持久化失败不阻断问答（显式降级）
                print(f"[警告] 会话持久化不可用（问答继续，不落盘历史）：{type(exc).__name__}: {exc}",
                      flush=True)
            if log is not None:
                log.log_event("gradio_app.start", version=VERSION, runtime=describe_runtime(),
                              warmup=warmup, feedback_log=str(FEEDBACK_PATH))
            engine = engine_cls(cfg=cfg, logger=log, storage=storage)
            meta["runtime"] = describe_runtime()
            if warmup:
                started = time.perf_counter()
                try:
                    info = engine.warmup(with_generate=True)
                    ok, detail = True, info
                except Exception as exc:  # noqa: BLE001 —— 预热失败不阻断界面启动
                    ok, detail = False, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[警告] 预热失败（界面仍启动，提问时会如实报错）："
                          f"{type(exc).__name__}: {exc}", flush=True)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                meta["warmup"] = {"ok": ok, "elapsed_ms": elapsed_ms, "detail": detail}
                print(f"[预热] MultiTurnEngine.warmup(with_generate=True) 完成：ok={ok}，"
                      f"实测耗时 {elapsed_ms:.2f} ms", flush=True)
            _ENGINE["engine"] = engine
            meta["ready"] = True
        except Exception as exc:  # noqa: BLE001 —— 如实记录，页面上说清原因
            meta["error"] = {"code": "E_ENGINE_UNAVAILABLE", "type": type(exc).__name__,
                             "message": str(exc)}
            print(f"[错误] 多轮引擎构造失败：{type(exc).__name__}: {exc}", flush=True)
        _ENGINE["meta"] = meta
        _ENGINE["done"] = True
        return _ENGINE["engine"], _ENGINE["meta"]


def get_engine() -> Any:
    """取单例引擎；未就绪抛 ``RuntimeError``（页面侧捕获后显示中文原因）。"""
    engine, meta = build_engine()
    if engine is None:
        raise RuntimeError(((meta or {}).get("error") or {}).get("message", "多轮引擎未就绪"))
    return engine


# --------------------------------------------------------------------------- 反馈落盘
_FEEDBACK_LOCK = threading.Lock()


def _normalize_vote(raw: Any) -> str | None:
    text = str(raw if raw is not None else "").strip().lower()
    if text in {"up", "1", "+1", "👍", "有帮助", "like", "good", "true"}:
        return "up"
    if text in {"down", "-1", "👎", "没帮助", "dislike", "bad", "false"}:
        return "down"
    return None


def append_feedback(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    """把一条反馈追加写入 ``部署/日志/用户反馈.jsonl``（线程安全、UTF-8、每行一个 JSON）。"""
    vote = _normalize_vote(payload.get("vote"))
    if vote is None:
        raise ValueError("vote 只能是 up/down（或 👍/👎、有帮助/没帮助）")
    now = time.time()
    record = {
        "session_id": str(payload.get("session_id") or "default"),
        "turn_index": int(payload.get("turn_index") or 0),
        "question": str(payload.get("question") or ""),
        "rewritten_query": str(payload.get("rewritten_query") or ""),
        "answer": str(payload.get("answer") or ""),
        "vote": vote,
        "ts": round(now, 3),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "source": source,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _FEEDBACK_LOCK:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        count = 0
        with FEEDBACK_PATH.open("r", encoding="utf-8") as fh:
            for count, _ in enumerate(fh, start=1):
                pass
    return {"ok": True, "path": str(FEEDBACK_PATH), "lines": count, "record": record}


# --------------------------------------------------------------------------- Chatbot 兼容层
#: True = Chatbot 接受 ``type="messages"``（gradio ≥4）；False = 退回 tuples 格式（gradio 3.x）
_MESSAGES_MODE = True


def _append_user(history: Any, text: str) -> list[Any]:
    out = list(history or [])
    if _MESSAGES_MODE:
        out.append({"role": "user", "content": text})
    else:
        out.append([text, None])
    return out


def _append_assistant(history: Any, text: str) -> list[Any]:
    out = list(history or [])
    if _MESSAGES_MODE:
        out.append({"role": "assistant", "content": text})
    elif out and len(out[-1]) >= 2 and not out[-1][1]:
        out[-1] = [out[-1][0], text]
    else:
        out.append([None, text])
    return out


def _drop_last_assistant(history: Any) -> list[Any]:
    """「重新生成」用：抹掉最后一条助手回答（保留用户气泡）。"""
    out = list(history or [])
    if not out:
        return out
    last = out[-1]
    is_assistant = ((isinstance(last, dict) and last.get("role") == "assistant")
                    or (isinstance(last, (list, tuple)) and len(last) >= 2 and not last[1]))
    if is_assistant:
        out.pop()
    return out


# --------------------------------------------------------------------------- 面板渲染
def _fmt(value: Any, unit: str = "ms") -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f} {unit}".strip()
    except (TypeError, ValueError):
        return f"{value} {unit}".strip()


def rewrite_markdown(data: dict[str, Any] | None, session_id: str | None = None) -> str:
    """本轮「Query 理解」面板（改写后查询 / 策略 / 焦点实体 / 指代消解）——核心卖点。"""
    head = "### 🔎 本轮 Query 理解（改写可见）"
    if not data:
        return (f"{head}\n\n_尚无改写记录。_ 会话 `{session_id or '-'}`\n\n"
                "试试多轮指代：先问「武汉兴图新科电子股份有限公司的注册资本是多少？」，"
                "再问「它的法定代表人是谁？」")
    refs = (data.get("rewrite") or {}).get("resolved_refs") or []
    ref_lines = "\n".join(
        f"  - `{r.get('surface', '?')}` → `{r.get('replacement', '?')}`"
        f"（{r.get('kind', 'ref')}）" for r in refs) or "  - （无指代需要消解）"
    changed = data.get("rewritten_query") != data.get("question")
    return "\n".join([
        head,
        f"- **用户原句**：{data.get('question', '')}",
        f"- **改写后的查询（送检索）**：`{data.get('rewritten_query', '')}`"
        + ("　← 已改写 ✅" if changed else "　（与原句一致）"),
        f"- **改写策略**：`{data.get('strategy', 'none')}`",
        f"- **焦点实体**：{data.get('focus_entity') or '（无）'}",
        "- **指代/省略消解**：",
        ref_lines,
    ])


def meta_markdown(data: dict[str, Any] | None, state: dict[str, Any] | None = None) -> str:
    """引用页码 + 响应耗时（ms）+ 多轮状态（焦点实体 / 上一轮槽位）。"""
    if not data:
        st = state or {}
        return ("### 📌 引用与耗时\n\n_尚无问答记录。_\n\n"
                f"- 会话：`{st.get('session_id', '-')}`\n"
                f"- 焦点实体：{st.get('focus_entity') or '（无）'}\n"
                f"- 上一轮槽位：{st.get('last_slot') or '（无）'}")
    lat = data.get("latency_ms") or {}
    pages = data.get("cited_pages") or []
    page_text = "、".join(f"第{p}页" for p in pages) or "（无）"
    cites = data.get("citations") or []
    cite_lines = "\n".join(
        f"  - {c.get('corpus', '')}{'：' if c.get('corpus') else ''}第{c.get('page', '?')}页"
        f"{'（表格）' if c.get('modality') == 'table' else '（图像）' if c.get('modality') == 'image' else ''}"
        for c in cites) or "  - （无引用）"
    st = state or {}
    return "\n".join([
        "### 📌 引用与耗时",
        f"- **引用页码**：{page_text}",
        "- **引用明细**：", cite_lines,
        f"- **响应耗时**：总计 **{_fmt(lat.get('total'))}**"
        f"（改写 {_fmt(lat.get('rewrite'))} / 检索 {_fmt(lat.get('retrieve'))}"
        f" / 生成 {_fmt(lat.get('generate'))}）",
        f"- **是否拒答**：{'是（证据不足，属正确行为）' if data.get('refused') else '否'}",
        f"- **轮次**：#{data.get('turn_index')}　语言：{data.get('language', 'zh')}",
        f"- **焦点实体**：{data.get('focus_entity') or st.get('focus_entity') or '（无）'}"
        f"　上一轮槽位：{st.get('last_slot') or '（无）'}"
        f"　上一轮独立问句：`{st.get('last_query') or '（无）'}`",
    ])


def state_markdown(session_id: str | None, meta: dict[str, Any] | None = None) -> str:
    """会话与运行时状态（session_id / 预热 / 引擎就绪）。"""
    meta = meta or {}
    warm = meta.get("warmup") or {}
    err = (meta.get("error") or {}).get("message")
    # 注意：Python 3.11 的 f-string 表达式内不能换行（PEP 701 是 3.12+），先算好再插值
    warm_text = ("ok=" + str(warm.get("ok")) + "，耗时 " + _fmt(warm.get("elapsed_ms"))
                 if warm else "（未预热）")
    lines = [
        "### ⚙️ 会话与运行时",
        f"- **session_id**：`{session_id or '-'}`（清空对话 + 重新生成都基于它）",
        f"- **引擎就绪**：{'是 ✅' if meta.get('ready') else '否 ❌'}",
        f"- **预热**：{warm_text}",
        f"- **反馈留痕**：`{FEEDBACK_PATH}`",
    ]
    if err:
        lines.append(f"- **引擎错误**：{err}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 事件回调
def _result_panels(result: Any, session_id: str, engine: Any) -> tuple[str, str, str]:
    data = result.to_dict()
    try:
        state = engine.state(session_id)
    except Exception:  # noqa: BLE001 —— 状态只是附加展示，失败不阻断回答
        state = {}
    return (rewrite_markdown(data, session_id), meta_markdown(data, state),
            state_markdown(session_id, _ENGINE.get("meta")))


def on_send(question: str, history: Any, session_id: str | None, last_turn: Any) -> tuple[Any, ...]:
    """发送一轮问题：改写 → 检索 → 生成 → 更新面板（返回 7 个输出）。"""
    history = list(history or [])
    text = (question or "").strip()
    sid = session_id or ("gradio-" + uuid.uuid4().hex[:12])
    if not text:
        return (history, sid, last_turn,
                "⚠️ **空输入未提交**：多轮界面里请直接输入问句（空输入不会被改写/检索）。",
                meta_markdown(None, {}), state_markdown(sid, _ENGINE.get("meta")), "")
    history = _append_user(history, text)
    try:
        engine = get_engine()
        result = engine.ask(text, session_id=sid, lang="auto", use_cache=True)
    except Exception as exc:  # noqa: BLE001 —— 页面如实显示原因，不让事件栈抛到前端
        msg = f"⚠️ 多轮问答不可用：{type(exc).__name__}: {exc}"
        history = _append_assistant(history, msg)
        return (history, sid, last_turn, msg, meta_markdown(None, {}),
                state_markdown(sid, _ENGINE.get("meta")), "")
    data = result.to_dict()
    history = _append_assistant(history, data.get("answer") or "")
    turn = {"session_id": sid, "turn_index": data.get("turn_index"), "question": data.get("question"),
            "rewritten_query": data.get("rewritten_query"), "answer": data.get("answer")}
    rw, meta_md, state_md = _result_panels(result, sid, engine)
    return (history, sid, turn, rw, meta_md, state_md, "")


def on_regenerate(history: Any, session_id: str | None, last_turn: Any) -> tuple[Any, ...]:
    """「重新生成」：对最后一轮忽略缓存（``use_cache=False``）重跑。"""
    history = list(history or [])
    sid = session_id or ("gradio-" + uuid.uuid4().hex[:12])
    turn = last_turn or {}
    question = str(turn.get("question") or "").strip()
    if not question:
        return (history, sid, last_turn, "⚠️ **没有可重新生成的轮次**：请先提一个问题。",
                meta_markdown(None, {}), state_markdown(sid, _ENGINE.get("meta")), "")
    history = _drop_last_assistant(history)
    try:
        engine = get_engine()
        result = engine.ask(question, session_id=sid, lang="auto", use_cache=False)
    except Exception as exc:  # noqa: BLE001
        msg = f"⚠️ 重新生成失败：{type(exc).__name__}: {exc}"
        history = _append_assistant(history, msg)
        return (history, sid, turn, msg, meta_markdown(None, {}),
                state_markdown(sid, _ENGINE.get("meta")), "")
    data = result.to_dict()
    answer = (data.get("answer") or "") + "\n\n_（本轮为「重新生成」：已忽略缓存 use_cache=False）_"
    history = _append_assistant(history, answer)
    new_turn = {"session_id": sid, "turn_index": data.get("turn_index"),
                "question": data.get("question"), "rewritten_query": data.get("rewritten_query"),
                "answer": data.get("answer")}
    rw, meta_md, state_md = _result_panels(result, sid, engine)
    return (history, sid, new_turn, rw, meta_md, state_md, "")


def on_clear(session_id: str | None) -> tuple[Any, ...]:
    """「清空对话」：``engine.reset(session_id)`` + 换新 session_id。"""
    old = session_id
    new_sid = "gradio-" + uuid.uuid4().hex[:12]
    note: list[str] = []
    if old:
        try:
            get_engine().reset(old)
            note.append(f"已在服务端重置旧会话 `{old}`")
        except Exception as exc:  # noqa: BLE001 —— 引擎不可用时也要能换新会话
            note.append(f"服务端重置失败（不影响换新会话）：{type(exc).__name__}: {exc}")
    note.append(f"新会话 `{new_sid}`")
    return ([], new_sid, None, rewrite_markdown(None, new_sid), meta_markdown(None, {}),
            state_markdown(new_sid, _ENGINE.get("meta")), "；".join(note))


def on_vote(vote: str, session_id: str | None, last_turn: Any) -> str:
    """👍/👎：把本轮反馈追加写入 ``部署/日志/用户反馈.jsonl``。"""
    turn = last_turn or {}
    if not turn.get("question"):
        return "⚠️ 还没有可反馈的回答（请先提问）。"
    payload = {
        "session_id": turn.get("session_id") or session_id or "default",
        "turn_index": turn.get("turn_index") or 0,
        "question": turn.get("question") or "",
        "rewritten_query": turn.get("rewritten_query") or "",
        "answer": turn.get("answer") or "",
        "vote": vote,
    }
    try:
        out = append_feedback(payload, source="gradio_app")
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ 反馈写入失败：{type(exc).__name__}: {exc}"
    mark = "👍 有帮助" if out["record"]["vote"] == "up" else "👎 没帮助"
    return (f"✅ 已记录反馈 **{mark}**（第 {out['record']['turn_index']} 轮）→ "
            f"`{out['path']}`，当前累计 {out['lines']} 条。")


# --------------------------------------------------------------------------- 界面
def _make_chatbot(gr: Any) -> tuple[Any, bool]:
    """构造 ``gr.Chatbot`` 并返回 ``(组件, 是否 messages 格式)``。

    本机装不了 gradio，无法实测版本差异，故按 gradio 3.x / 4.x / 5.x 的差异逐个降级尝试：
    ``type="messages"``（≥4）→ 去掉 ``show_copy_button`` → 退回 tuples（3.x）→ 全默认。
    """
    label, height = "多轮对话", 460
    candidates: list[dict[str, Any]] = [
        {"type": "messages", "label": label, "height": height, "show_copy_button": True},
        {"type": "messages", "label": label, "height": height},
        {"label": label, "height": height, "show_copy_button": True},
        {"label": label, "height": height},
        {},
    ]
    for kwargs in candidates:
        try:
            chatbot = gr.Chatbot(**kwargs)
        except TypeError:                  # 该版本不认识某个参数 → 换下一个候选
            continue
        return chatbot, kwargs.get("type") == "messages"
    raise RuntimeError("无法构造 gr.Chatbot（gradio 版本过旧或接口异常）")


def build_app(gr: Any) -> Any:
    """用 ``gr.Blocks`` 组装界面（``gr`` 由调用方传入，便于无依赖环境下静态检查）。"""
    global _MESSAGES_MODE
    chatbot, _MESSAGES_MODE = _make_chatbot(gr)

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(
            f"# {TITLE}\n"
            f"工单5 · **{WORK_ORDER}** ｜ 多轮对话 + 查询改写可视化 + 引用页码/耗时 + 用户反馈\n\n"
            "> 试试多轮指代：先问「武汉兴图新科电子股份有限公司的注册资本是多少？」，"
            "再问「它的法定代表人是谁？」——右侧会显示改写后的独立问句与策略。"
        )
        session_state = gr.State(None)
        last_turn_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot.render()
                msg = gr.Textbox(label="你的问题（Enter 发送）", lines=2,
                                 placeholder="例如：它的法定代表人是谁？")
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    regen_btn = gr.Button("重新生成（忽略缓存）")
                    clear_btn = gr.Button("清空对话")
                with gr.Row():
                    up_btn = gr.Button("👍 有帮助")
                    down_btn = gr.Button("👎 没帮助")
                    fb_status = gr.Markdown("_反馈会写入 `部署/日志/用户反馈.jsonl`（每行一个 JSON）。_")
                gr.Examples(examples=[[q] for q in DEMO_QUESTIONS], inputs=[msg],
                            label="示例问题（点击填入）")
            with gr.Column(scale=2):
                rewrite_panel = gr.Markdown(rewrite_markdown(None, None))
                meta_panel = gr.Markdown(meta_markdown(None, {}))
                state_panel = gr.Markdown(state_markdown(None, _ENGINE.get("meta")))

        outputs = [chatbot, session_state, last_turn_state, rewrite_panel, meta_panel,
                   state_panel, msg]
        send_btn.click(on_send, inputs=[msg, chatbot, session_state, last_turn_state],
                       outputs=outputs)
        msg.submit(on_send, inputs=[msg, chatbot, session_state, last_turn_state],
                   outputs=outputs)
        regen_btn.click(on_regenerate, inputs=[chatbot, session_state, last_turn_state],
                        outputs=outputs)
        clear_btn.click(on_clear, inputs=[session_state], outputs=outputs)
        up_btn.click(lambda sid, turn: on_vote("up", sid, turn),
                     inputs=[session_state, last_turn_state], outputs=[fb_status])
        down_btn.click(lambda sid, turn: on_vote("down", sid, turn),
                       inputs=[session_state, last_turn_state], outputs=[fb_status])
        gr.Markdown(
            "---\n启动命令：`pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860`　"
            "｜ 兜底界面（零第三方依赖）：`-Mode web`（8123）　"
            f"｜ 反馈留痕：`{FEEDBACK_PATH}`"
        )
    return demo


# --------------------------------------------------------------------------- 启动
def _import_gradio() -> Any:
    """延迟导入 gradio；缺失时打印中文指引并 ``sys.exit(3)``。"""
    try:
        import gradio as gr                             # noqa: PLC0415 —— 刻意延迟
        return gr
    except ImportError as exc:
        print(f"[缺依赖] 无法 import gradio：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  本机禁止裸 pip install（落点是只读的 工单1 venv，会写坏共享环境）。", file=sys.stderr)
        print("  需要安装时请装进仓库内的私有目录：", file=sys.stderr)
        print("      pwsh -NoProfile -File run_py.ps1 -m pip install --target 研发/第三方库 gradio",
              file=sys.stderr)
        print("  装好后用启动脚本（会自动把 研发/第三方库 加进 PYTHONPATH）：", file=sys.stderr)
        print("      pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("  ★ 不想装依赖：直接用零第三方依赖的兜底界面（必可用）：", file=sys.stderr)
        print("      pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web -Port 8123",
              file=sys.stderr)
        sys.exit(EXIT_DEPS)
    except Exception as exc:  # noqa: BLE001 —— gradio 自身导入期报错也要给出中文说明
        print(f"[缺依赖] 导入 gradio 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("  建议改用兜底界面：pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web",
              file=sys.stderr)
        sys.exit(EXIT_DEPS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gradio_app.py",
                                     description=f"工单5 多轮对话 Gradio 界面：{WORK_ORDER}")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"监听端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网分享链接（默认关闭）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动预热（仅调试）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gr = _import_gradio()                       # 缺 gradio → 中文指引 + 退出码 3

    print(f"[启动] {SERVICE} —— {WORK_ORDER}", flush=True)
    print(f"[启动] 仓库根 {REPO_ROOT}", flush=True)
    print("[启动] 引擎预热中（首次加载索引 + 嵌入 + 生成模型，请稍候）…", flush=True)
    _, meta = build_engine(warmup=not args.no_warmup)
    if not meta.get("ready"):
        print("[警告] 引擎未就绪：界面仍会启动，提问时页面会如实显示原因（503 口径）。", flush=True)
        print(f"[警告] 原因：{(meta.get('error') or {}).get('message', '未知')}", flush=True)

    demo = build_app(gr)
    try:
        demo.queue().launch(server_name=str(args.host), server_port=int(args.port),
                            share=bool(args.share), show_error=True)
    except OSError as exc:
        print(f"[失败] 无法监听 {args.host}:{args.port} —— {type(exc).__name__}: {exc}", flush=True)
        print(f"       端口可能被占用：netstat -ano | findstr :{int(args.port)} → 换端口用 --port <n>",
              flush=True)
        return EXIT_PARAM
    except KeyboardInterrupt:
        print("\n[停止] 收到 Ctrl+C", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
