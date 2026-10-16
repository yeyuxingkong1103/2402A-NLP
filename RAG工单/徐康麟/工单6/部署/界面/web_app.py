# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-Query 理解优化
# 文件：部署/界面/web_app.py —— 零第三方依赖（纯标准库）的多轮对话兜底 Web 界面，默认 127.0.0.1:8123。
"""部署/界面/web_app.py —— 多轮对话「兜底」Web 界面（**只用 Python 标准库**，默认端口 8123）。

工单编号：人工智能NLP-RAG-Query 理解优化

为什么要有这个界面：本机唯一可用解释器（工单1 venv）**没有装 gradio / streamlit / pandas**，
且红线禁止裸 ``pip install``。所以 ``部署/界面/gradio_app.py``（首选界面）可能起不来，
本文件保证「任何解释器、任何依赖状态下都有一页能用的多轮对话界面」。

设计纪律
--------
* 依赖：``http.server`` / ``socketserver`` / ``json`` / ``urllib.parse`` / ``threading`` —— 零第三方包；
* 引擎：进程内**单例**（启动构造一次 + ``warmup()`` 预热一次），每个请求直接调
  ``MultiTurnEngine.ask()``，**不经过** 8600 端口的 FastAPI 服务；
* 契约：请求/响应严格照 ``设计/接口契约_冻结.md`` §1.3 / §2（``MultiTurnEngine`` /
  ``POST /api/chat``），不自行发明字段；
* 延迟导入：``app.multiturn_engine`` 在**函数内**导入（不在模块顶层裸 import），
  核心引擎未就绪时本文件仍能被 ``ast.parse`` / ``import`` 解析；
* 诚实降级：引擎不可用时服务**照常启动**，``GET /health`` 返回 503 + 中文原因，
  ``POST /api/chat`` 返回 503 + 中文指引（绝不返回假答案、不谎报 ok）。

HTTP 路由
---------
    GET  /                 单页中文界面（内嵌 HTML，UTF-8，无外部 CDN）
    GET  /health           健康检查：{"status":"ok", ...}；引擎未就绪 → 503 且 status=degraded
    POST /api/chat         {"question","session_id","language","use_cache"} → 契约 JSON
    POST /api/feedback     写 ``部署/日志/用户反馈.jsonl``（每行一个 JSON）
    POST /api/reset        （附加路由，服务于「清空对话」按钮）→ engine.reset(session_id)

启动（推荐走统一入口脚本，保证解释器 / UTF-8 / PYTHONPATH 一致）::

    pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode web -Port 8123
    # 等价于：
    pwsh -NoProfile -File run_py.ps1 部署/界面/web_app.py --host 127.0.0.1 --port 8123

退出码：``0`` 正常退出 / ``2`` 端口被占用或参数错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"
VERSION = "2.0.0"
SERVICE = "部署/界面/web_app.py（纯标准库界面 · 含检索策略配置）"
TITLE = "招股说明书问答 · 混合检索（向量 / 全文 / 混合 可切换）"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8123
MAX_QUESTION_CHARS = 500
EXIT_OK, EXIT_PARAM = 0, 2

ENDPOINTS = ["GET /", "GET /health", "GET /api/strategies", "GET /reports/…",
             "POST /api/chat", "POST /api/feedback", "POST /api/reset"]

#: 多轮演示问题：前两轮看指代消解（pronoun），后两轮看省略/话题继承（ellipsis_entity_switch / topic_carry）
DEMO_QUESTIONS: list[dict[str, str]] = [
    {"tag": "首轮·显式实体", "question": "武汉兴图新科电子股份有限公司的注册资本是多少？"},
    {"tag": "指代·他/它", "question": "它的法定代表人是谁？"},
    {"tag": "省略·换实体", "question": "那武汉力源信息技术股份有限公司呢？"},
    {"tag": "话题继承·无实体", "question": "法定代表人是谁？"},
    {"tag": "英文提问", "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"},
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

#: 工单6：测试报告静态页根目录（只读暴露给 ``GET /reports/...``）。
#: 为什么需要：浏览器禁止 ``file://``，而评测截图页是本地 HTML；
#: 走 HTTP 暴露只读目录，截图与人工复核才能在同一地址栏里完成。
REPORTS_ROOT: Path = REPO_ROOT / "测试" / "测试数据"
REPORTS_PREFIX = "/reports/"


# --------------------------------------------------------------------------- 引擎（进程内单例）
class EngineUnavailable(RuntimeError):
    """多轮引擎不可用（缺模块 / 构造失败 / 索引缺失）——对外如实转成 503。"""


_ENGINE_LOCK = threading.Lock()
_ENGINE: dict[str, Any] = {"done": False, "engine": None, "meta": {}}
_FEEDBACK_LOCK = threading.Lock()
_SESSION_LOCKS: dict[str, threading.Lock] = {}
_SESSION_LOCKS_LOCK = threading.Lock()


def _multiturn_guidance(exc: BaseException) -> str:
    """缺模块时的中文指引（带自检与安装命令，便于现场排障）。"""
    return (
        f"多轮引擎不可用：无法 import 研发/app/multiturn_engine.py "
        f"（{type(exc).__name__}: {exc}）。\n"
        "  · 常见原因：研发/app/multiturn_engine.py 或它依赖的 "
        "研发/app/core/dialogue_state.py、研发/app/core/query_rewriter.py 尚未就绪。\n"
        "  · 自检：pwsh -NoProfile -File run_py.ps1 -c \"import app.multiturn_engine as m; print(m.__file__)\"\n"
        "  · 本界面本身零第三方依赖；引擎就绪后重启本服务即可，无需改任何代码。"
    )


def _load_multiturn_engine_class() -> Any:
    """**延迟导入**核心引擎（绝不在模块顶层裸 import，保证本文件永远可解析/可导入）。"""
    try:
        from app.multiturn_engine import MultiTurnEngine          # noqa: PLC0415 —— 刻意延迟
    except ImportError as exc:
        raise EngineUnavailable(_multiturn_guidance(exc)) from exc
    return MultiTurnEngine


def build_engine(*, warmup: bool = True) -> tuple[Any, dict[str, Any]]:
    """构造（并缓存）``MultiTurnEngine`` 单例；预热耗时实测打印，失败则记录原因不抛出。"""
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
            except Exception as exc:  # noqa: BLE001 —— 日志初始化失败不阻断问答（显式降级）
                log = None
                print(f"[警告] 日志初始化失败（问答继续，仅少落盘日志）：{type(exc).__name__}: {exc}",
                      flush=True)
            storage = None
            try:
                storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
            except Exception as exc:  # noqa: BLE001 —— 持久化失败不阻断问答（显式降级）
                print(f"[警告] 会话持久化不可用（问答继续，不落盘历史）：{type(exc).__name__}: {exc}",
                      flush=True)
            if log is not None:
                log.log_event("web_app.start", version=VERSION, runtime=describe_runtime(),
                              endpoints=ENDPOINTS, warmup=warmup, feedback_log=str(FEEDBACK_PATH))
            engine = engine_cls(cfg=cfg, logger=log, storage=storage)
            meta["runtime"] = describe_runtime()
            if warmup:
                started = time.perf_counter()
                try:
                    info = engine.warmup(with_generate=True)
                    ok = True
                    detail = info
                except Exception as exc:  # noqa: BLE001 —— 预热失败（如索引缺失）不阻断服务启动
                    ok = False
                    detail = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[警告] 预热失败（服务仍启动；/health 与 /api/chat 会如实报错）："
                          f"{type(exc).__name__}: {exc}", flush=True)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                meta["warmup"] = {"ok": ok, "elapsed_ms": elapsed_ms, "detail": detail}
                print(f"[预热] MultiTurnEngine.warmup(with_generate=True) 完成：ok={ok}，"
                      f"实测耗时 {elapsed_ms:.2f} ms", flush=True)
            _ENGINE["engine"] = engine
            meta["ready"] = True
        except Exception as exc:  # noqa: BLE001 —— 必须如实记录，让 /health 说出原因
            meta["error"] = {"code": "E_ENGINE_UNAVAILABLE", "type": type(exc).__name__,
                             "message": str(exc)}
            print(f"[错误] 多轮引擎构造失败：{type(exc).__name__}: {exc}", flush=True)
        _ENGINE["meta"] = meta
        _ENGINE["done"] = True
        return _ENGINE["engine"], _ENGINE["meta"]


def get_engine_or_none() -> tuple[Any, dict[str, Any]]:
    """取单例引擎；未就绪返回 ``(None, meta)``（调用方据此回 503，而不是 500）。"""
    return build_engine()


def session_lock(session_id: str) -> threading.Lock:
    """同一 session_id 串行化（避免并发请求把同一会话的历史写乱）。"""
    with _SESSION_LOCKS_LOCK:
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[session_id] = lock
        return lock


# --------------------------------------------------------------------------- 反馈落盘
def _normalize_vote(raw: Any) -> str | None:
    """把前端传的赞/踩归一化成 ``up`` / ``down``；无法识别返回 ``None``。"""
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


# --------------------------------------------------------------------------- 单页界面
_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --line:#e2e6f0; --ink:#1b2233; --muted:#6b7390;
          --accent:#2f6fed; --warn:#b7791f; --bad:#c0392b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { background:linear-gradient(90deg,#2f6fed,#5b8def); color:#fff; padding:14px 22px; }
  header h1 { margin:0; font-size:18px; }
  header p { margin:5px 0 0; font-size:12.5px; opacity:.93; }
  .layout { max-width:1180px; margin:16px auto 40px; padding:0 14px;
            display:grid; grid-template-columns:minmax(0,1fr) 336px; gap:14px; align-items:start; }
  @media (max-width:900px) { .layout { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .chat { min-height:340px; max-height:58vh; overflow-y:auto; display:flex;
          flex-direction:column; gap:12px; }
  .msg { display:flex; flex-direction:column; }
  .msg.user { align-items:flex-end; }
  .msg.bot { align-items:flex-start; }
  .bubble { max-width:84%; padding:10px 13px; border-radius:12px; font-size:15px;
            line-height:1.72; white-space:pre-wrap; word-break:break-word; }
  .msg.user .bubble { background:var(--accent); color:#fff; border-bottom-right-radius:3px; }
  .msg.bot .bubble { background:#f1f5fd; border:1px solid var(--line); border-bottom-left-radius:3px; }
  .msg.bot.refused .bubble { background:#fff8e6; color:var(--warn); font-weight:600; }
  .msg.sys .bubble { background:#eef2fb; color:var(--muted); font-size:13px; }
  .msg.err .bubble { background:#fdecea; border:1px solid #f5c6c0; color:var(--bad); font-size:13.5px; }
  .meta { font-size:12px; color:var(--muted); margin:6px 0 0 3px; max-width:84%; }
  .meta b { color:var(--ink); font-weight:600; }
  .rw { font-size:12.5px; color:#1f4fd8; background:#eef3ff; border:1px dashed #bcd0ff;
        border-radius:8px; padding:6px 9px; margin:6px 0 0 3px; max-width:84%; }
  .fb { margin:6px 0 0 3px; display:flex; gap:8px; align-items:center; }
  .fb button { font-size:12px; padding:4px 10px; border-radius:6px; border:1px solid var(--line);
               background:#fff; color:var(--muted); cursor:pointer; }
  .fb button.on { background:#eaf1ff; border-color:#bcd0ff; color:var(--accent); font-weight:600; }
  .fb span { font-size:12px; color:var(--muted); }
  .composer { margin-top:12px; display:flex; gap:8px; align-items:flex-end; }
  textarea { flex:1; min-height:62px; max-height:180px; border:1px solid var(--line);
             border-radius:8px; padding:9px 10px; font-size:15px; font-family:inherit; resize:vertical; }
  button.primary { border:0; border-radius:8px; padding:11px 20px; font-size:15px; cursor:pointer;
                   background:var(--accent); color:#fff; }
  button.ghost { border:1px solid var(--line); border-radius:8px; padding:10px 14px; font-size:13.5px;
                 cursor:pointer; background:#fff; color:var(--ink); }
  button:disabled { opacity:.55; cursor:progress; }
  .demos { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .demos button { border:1px solid var(--line); background:#fff; border-radius:14px; padding:5px 11px;
                  font-size:12.5px; color:var(--muted); cursor:pointer; }
  .demos button:hover { border-color:#bcd0ff; color:var(--accent); }
  .status { margin-top:9px; font-size:12.5px; color:var(--muted); min-height:18px; }
  .status.bad { color:var(--bad); }
  aside h3 { margin:0 0 8px; font-size:13.5px; color:var(--muted); font-weight:600; }
  aside .card + .card { margin-top:12px; }
  dl { margin:0; font-size:12.5px; }
  dt { color:var(--muted); margin-top:6px; }
  dd { margin:2px 0 0; word-break:break-all; }
  code { background:#f1f3f9; border-radius:4px; padding:1px 4px; font-size:12px; }
  .strat { display:flex; flex-direction:column; gap:9px; font-size:12.5px; }
  .strat label { display:flex; flex-direction:column; gap:4px; color:var(--muted); }
  .strat label[data-when] { display:none; }
  .strat label[data-when].on { display:flex; }
  .strat select, .strat input[type=number] { font-size:12.5px; padding:4px 6px; border:1px solid var(--line);
    border-radius:6px; background:#fff; color:var(--fg); width:100%; box-sizing:border-box; }
  .strat input[type=range] { width:100%; }
  .strat b { color:var(--fg); }
  .strat-actions { display:flex; gap:6px; flex-wrap:wrap; }
  .strat-actions button { font-size:12px; padding:5px 9px; }
  .strat-summary { font-size:12px; line-height:1.6; color:var(--fg); background:#f7f8fc;
    border:1px solid var(--line); border-radius:6px; padding:7px 8px; word-break:break-all; }
  .strat-help { font-size:11.5px; color:var(--muted); line-height:1.6; }
  .cmp { width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }
  .cmp th, .cmp td { border:1px solid var(--line); padding:5px 6px; text-align:left; vertical-align:top; }
  .cmp th { background:#f1f3f9; }
  .cmp .m { font-weight:600; white-space:nowrap; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>工单6 · 人工智能NLP-RAG-混合检索任务 ｜ 向量检索 / 全文检索 / 混合检索 三模式可切换 ｜ 多轮对话 + 查询改写 ｜ 零第三方依赖（Python 标准库）· 端口 __PORT__</p>
</header>
<div class="layout">
  <main>
    <section class="card">
      <div id="chat" class="chat"></div>
      <div class="composer">
        <textarea id="q" placeholder="输入问题后按 Enter 发送（Shift+Enter 换行）；例如：它的法定代表人是谁？"></textarea>
        <button id="send" class="primary">发送</button>
        <button id="clear" class="ghost" title="重置服务端该会话状态并换一个新 session_id">清空对话</button>
      </div>
      <div id="demos" class="demos"></div>
      <div id="status" class="status"></div>
    </section>
  </main>
  <aside>
    <div class="card" id="strat-card">
      <h3>检索策略（工单6 核心）</h3>
      <div class="strat">
        <label>检索模式
          <select id="st-mode">
            <option value="vector">向量检索（嵌入召回 + 重排）</option>
            <option value="fulltext">全文检索（倒排索引 + 布尔/短语/模糊）</option>
            <option value="hybrid" selected>混合检索（向量 + 全文 融合）</option>
          </select>
        </label>
        <label data-when="vector hybrid">嵌入模型
          <select id="st-embed"></select>
        </label>
        <label data-when="hybrid">融合算法
          <select id="st-fusion"></select>
        </label>
        <label data-when="hybrid">权重：向量 <b id="st-vw-label">0.50</b> ／ 全文 <b id="st-fw-label">0.50</b>
          <input type="range" id="st-vw" min="0" max="1" step="0.05" value="0.5" />
        </label>
        <label data-when="vector fulltext hybrid">重排模型
          <select id="st-rerank"></select>
        </label>
        <label data-when="vector fulltext hybrid">返回 Top-K
          <input type="number" id="st-topk" min="1" max="20" value="5" />
        </label>
        <label data-when="vector hybrid">向量召回 K
          <input type="number" id="st-vk" min="1" max="100" value="20" />
        </label>
        <label data-when="fulltext hybrid">全文召回 K
          <input type="number" id="st-fk" min="1" max="100" value="20" />
        </label>
        <label data-when="vector fulltext hybrid">归一化分数阈值 <b id="st-th-label">0.00</b>
          <input type="range" id="st-th" min="0" max="0.9" step="0.05" value="0" />
        </label>
        <label data-when="vector hybrid">向量相似度下限 <b id="st-ms-label">0.00</b>
          <input type="range" id="st-ms" min="0" max="0.9" step="0.05" value="0" />
        </label>
        <label data-when="fulltext hybrid">全文检索字段（多选）
          <select id="st-fields" multiple size="4"></select>
        </label>
        <label data-when="fulltext hybrid">默认布尔运算符
          <select id="st-op"><option value="OR">OR（任一命中）</option><option value="AND">AND（全部命中）</option></select>
        </label>
        <label data-when="fulltext hybrid">
          <span><input type="checkbox" id="st-fuzzy" checked style="width:auto" /> 启用模糊匹配（查询串写 <code>词~</code> 触发）</span>
        </label>
        <div class="strat-actions">
          <button id="st-preset" class="ghost">套用预设</button>
          <button id="st-compare" class="ghost" title="同一问题分别用向量 / 全文 / 混合检索各跑一次并对比">三模式对比</button>
        </div>
        <div id="st-summary" class="strat-summary">策略未加载</div>
        <div id="st-help" class="strat-help">全文语法：甲 AND 乙 ／ 甲 OR 乙 ／ NOT 甲 ／ "短语" ／ 词~ ／ 标题:词</div>
      </div>
    </div>
    <div class="card">
      <h3>会话 / 槽位（多轮状态）</h3>
      <dl>
        <dt>当前 session_id</dt><dd id="s-session">-</dd>
        <dt>焦点实体</dt><dd id="s-focus">-</dd>
        <dt>上一轮槽位</dt><dd id="s-slot">-</dd>
        <dt>上一轮独立问句</dt><dd id="s-lastq">-</dd>
        <dt>会话轮数</dt><dd id="s-turns">-</dd>
      </dl>
    </div>
    <div class="card">
      <h3>本轮 Query 理解（核心卖点）</h3>
      <dl>
        <dt>改写后的查询</dt><dd id="s-rewritten">-</dd>
        <dt>改写策略</dt><dd id="s-strategy">-</dd>
        <dt>指代消解</dt><dd id="s-refs">-</dd>
        <dt>耗时（总 / 改写 / 检索 / 生成）</dt><dd id="s-latency">-</dd>
        <dt>引用页码</dt><dd id="s-pages">-</dd>
      </dl>
    </div>
    <div class="card">
      <h3>本轮检索策略明细（工单6）</h3>
      <dl>
        <dt>生效策略</dt><dd id="s-st-label">-</dd>
        <dt>各通道召回</dt><dd id="s-st-channels">-</dd>
        <dt>融合 / 重排</dt><dd id="s-st-fusion">-</dd>
        <dt>阶段耗时</dt><dd id="s-st-timing">-</dd>
        <dt>说明 / 降级</dt><dd id="s-st-notes">-</dd>
      </dl>
    </div>
    <div class="card">
      <h3>说明</h3>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.7">
        每条回答下方的 👍/👎 会追加写入 <code>部署/日志/用户反馈.jsonl</code>（每行一个 JSON）。<br />
        检索策略三模式：<code>vector</code> / <code>fulltext</code> / <code>hybrid</code>，
        权重与融合算法实时可调；「三模式对比」用同一问题各跑一次并列表对照。<br />
        首选界面：<code>gradio_app.py</code>（7860，缺 gradio 时用本页兜底）。<br />
        接口：<code>GET /health</code>、<code>GET /api/strategies</code>、<code>POST /api/chat</code>、
        <code>POST /api/feedback</code>、<code>POST /api/reset</code>。
      </div>
    </div>
  </aside>
</div>
<script>
const DEMOS = __DEMOS__;
const TURNS = [];
let SESSION = newSessionId();
let BUSY = false;

function $(id) { return document.getElementById(id); }
function esc(s) {
  return (s === null || s === undefined) ? '' : String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function newSessionId() {
  try {
    if (window.crypto && crypto.randomUUID) {
      return 'web-' + crypto.randomUUID().replace(/-/g, '').slice(0, 12);
    }
  } catch (e) { /* 退化到随机串 */ }
  return 'web-' + Math.random().toString(36).slice(2, 14);
}
function setStatus(text, bad) {
  const el = $('status');
  el.textContent = text || '';
  el.className = bad ? 'status bad' : 'status';
}
function scrollBottom() { const c = $('chat'); c.scrollTop = c.scrollHeight; }

function bubble(role, html, extraClass) {
  const msg = document.createElement('div');
  msg.className = 'msg ' + role + (extraClass ? ' ' + extraClass : '');
  const b = document.createElement('div');
  b.className = 'bubble';
  b.innerHTML = html;
  msg.appendChild(b);
  $('chat').appendChild(msg);
  scrollBottom();
  return msg;
}
function addUser(text) { return bubble('user', esc(text)); }
function addSystem(text) { return bubble('sys', esc(text)); }
function addError(text) {
  setStatus(text, true);
  return bubble('err', '⚠️ ' + esc(text));
}

function latencyLine(lat) {
  const l = lat || {};
  const f = (v) => (v === null || v === undefined) ? '-' : Math.round(v * 10) / 10;
  return '耗时 总 ' + f(l.total) + ' ms（改写 ' + f(l.rewrite) + ' / 检索 ' + f(l.retrieve) +
         ' / 生成 ' + f(l.generate) + '）';
}
function metaHtml(data) {
  const pages = (data.cited_pages || []);
  const pageText = pages.length ? pages.map(p => '第' + p + '页').join('、') : '无';
  const cites = (data.citations || []).map(c =>
    (c.corpus ? c.corpus + '：' : '') + '第' + c.page + '页' + (c.modality === 'table' ? '（表格）'
      : (c.modality === 'image' ? '（图像）' : ''))).join('；');
  return '引用页码：<b>' + esc(pageText) + '</b>' + (cites ? '（' + esc(cites) + '）' : '') +
         '<br />' + esc(latencyLine(data.latency_ms)) +
         ' ｜ 策略 ' + esc(data.strategy || '-') +
         ' ｜ 焦点实体 ' + esc(data.focus_entity || '-') +
         ' ｜ 轮次 #' + esc(data.turn_index);
}
function rewriteHtml(data) {
  if (!data.rewritten_query || data.rewritten_query === data.question) { return ''; }
  return '🔎 改写成：' + esc(data.rewritten_query) + '（策略 ' + esc(data.strategy || '-') + '）';
}

function renderTurn(data, wallMs) {
  const refused = !!data.refused;
  const msg = bubble('bot', esc(data.answer || ''), refused ? 'refused' : '');
  const idx = TURNS.length;
  TURNS.push({
    turn_index: data.turn_index, question: data.question,
    rewritten_query: data.rewritten_query, answer: data.answer, voted: null
  });
  const rw = rewriteHtml(data);
  if (rw) {
    const d = document.createElement('div');
    d.className = 'rw';
    d.innerHTML = rw;
    msg.appendChild(d);
  }
  const m = document.createElement('div');
  m.className = 'meta';
  m.innerHTML = metaHtml(data) + ' ｜ 浏览器往返 ' + wallMs + ' ms';
  msg.appendChild(m);

  const fb = document.createElement('div');
  fb.className = 'fb';
  fb.id = 'fb-' + idx;
  const up = document.createElement('button');
  up.textContent = '👍 有帮助';
  up.onclick = () => vote(idx, 'up');
  const down = document.createElement('button');
  down.textContent = '👎 没帮助';
  down.onclick = () => vote(idx, 'down');
  const tip = document.createElement('span');
  tip.textContent = '';
  fb.appendChild(up); fb.appendChild(down); fb.appendChild(tip);
  msg.appendChild(fb);
  setSidebar(data);
  return msg;
}

async function vote(idx, v) {
  const t = TURNS[idx];
  if (!t) { return; }
  if (t.voted) { setStatus('该轮已投过票：' + (t.voted === 'up' ? '👍 有帮助' : '👎 没帮助')); return; }
  t.voted = v;
  const box = $('fb-' + idx);
  if (box) {
    box.children[0].className = (v === 'up') ? 'on' : '';
    box.children[1].className = (v === 'down') ? 'on' : '';
    box.children[2].textContent = '已记录';
  }
  try {
    const resp = await fetch('/api/feedback', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION, turn_index: t.turn_index, question: t.question,
                             rewritten_query: t.rewritten_query, answer: t.answer, vote: v })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      setStatus('反馈写入失败：' + ((data.error && data.error.message) || ('HTTP ' + resp.status)), true);
      return;
    }
    setStatus('已写入反馈（' + t.turn_index + ' 轮 ' + v + '）→ ' + (data.path || ''));
  } catch (err) {
    setStatus('反馈写入失败：' + err, true);
  }
}

function setText(id, value) { $(id).textContent = (value === null || value === undefined || value === '') ? '-' : value; }

function setSidebar(data) {
  const st = (data && data.state) || {};
  setText('s-session', (data && data.session_id) || SESSION);
  setText('s-focus', (data && data.focus_entity) || st.focus_entity);
  setText('s-slot', st.last_slot);
  setText('s-lastq', st.last_query);
  setText('s-turns', (st.turns && st.turns.length) ? String(st.turns.length) : (data ? data.turn_index : ''));
  setText('s-rewritten', data && data.rewritten_query);
  setText('s-strategy', data && data.strategy);
  const refs = ((data && data.rewrite && data.rewrite.resolved_refs) || [])
    .map(r => (r.surface || '?') + ' → ' + (r.replacement || '?')).join('；');
  setText('s-refs', refs);
  const l = (data && data.latency_ms) || {};
  const f = (v) => (v === null || v === undefined) ? '-' : (Math.round(v * 10) / 10) + ' ms';
  setText('s-latency', data ? (f(l.total) + ' / ' + f(l.rewrite) + ' / ' + f(l.retrieve) + ' / ' + f(l.generate)) : '-');
  setText('s-pages', ((data && data.cited_pages) || []).join('、'));
}

// ---------------------------------------------------------------- 工单6 检索策略
let STRAT_META = null;

function stratMode() { return $('st-mode').value; }

function syncStratVisibility() {
  const m = stratMode();
  document.querySelectorAll('#strat-card label[data-when]').forEach(function (el) {
    const modes = (el.getAttribute('data-when') || '').split(/\s+/);
    el.classList.toggle('on', modes.indexOf(m) >= 0);
  });
}

function syncWeightLabels() {
  const vw = parseFloat($('st-vw').value) || 0;
  $('st-vw-label').textContent = vw.toFixed(2);
  $('st-fw-label').textContent = (1 - vw).toFixed(2);
  $('st-th-label').textContent = (parseFloat($('st-th').value) || 0).toFixed(2);
  $('st-ms-label').textContent = (parseFloat($('st-ms').value) || 0).toFixed(2);
}

function fillSelect(sel, items, selected) {
  sel.innerHTML = '';
  (items || []).forEach(function (it) {
    const o = document.createElement('option');
    o.value = it.value;
    o.textContent = it.label;
    if (it.disabled) { o.disabled = true; }
    if (selected && selected.indexOf(it.value) >= 0) { o.selected = true; }
    sel.appendChild(o);
  });
}

function currentStrategy() {
  const vw = parseFloat($('st-vw').value);
  const fields = Array.from($('st-fields').selectedOptions).map(function (o) { return o.value; });
  return {
    mode: stratMode(),
    embedding_model: $('st-embed').value || 'bge-m3',
    fusion: $('st-fusion').value || 'weighted_average',
    vector_weight: isNaN(vw) ? 0.5 : vw,
    fulltext_weight: isNaN(vw) ? 0.5 : Math.round((1 - vw) * 100) / 100,
    rerank_model: $('st-rerank').value || 'rule',
    top_k: parseInt($('st-topk').value, 10) || 5,
    vector_k: parseInt($('st-vk').value, 10) || 20,
    fulltext_k: parseInt($('st-fk').value, 10) || 20,
    score_threshold: parseFloat($('st-th').value) || 0,
    vector_min_similarity: parseFloat($('st-ms').value) || 0,
    fulltext_fields: fields.length ? fields : ['content'],
    fulltext_op: $('st-op').value || 'OR',
    enable_fuzzy: $('st-fuzzy').checked
  };
}

function strategyLine(s) {
  if (s.mode === 'vector') {
    return '向量检索｜' + s.embedding_model + '｜召回 K=' + s.vector_k + '｜重排=' + s.rerank_model;
  }
  if (s.mode === 'fulltext') {
    return '全文检索｜字段=' + s.fulltext_fields.join('+') + '｜默认 ' + s.fulltext_op +
      '｜模糊=' + (s.enable_fuzzy ? '开' : '关') + '｜重排=' + s.rerank_model;
  }
  return '混合检索｜融合=' + s.fusion + '｜向量 ' + s.vector_weight.toFixed(2) +
    ' / 全文 ' + s.fulltext_weight.toFixed(2) + '｜' + s.embedding_model + '｜重排=' + s.rerank_model;
}

function refreshStratSummary() {
  const s = currentStrategy();
  $('st-summary').textContent = strategyLine(s) + '｜Top-K=' + s.top_k +
    '｜阈值=' + s.score_threshold.toFixed(2);
}

async function loadStrategies() {
  try {
    const resp = await fetch('/api/strategies');
    const data = await resp.json().catch(function () { return {}; });
    if (!resp.ok || data.ok === false) {
      throw new Error((data.error && data.error.message) || ('HTTP ' + resp.status));
    }
    STRAT_META = data;
    fillSelect($('st-embed'), (data.embedding_models || []).map(function (m) {
      return { value: m.id, label: m.label + (m.index_built ? '' : '（索引未建）'), disabled: !m.index_built };
    }), ['bge-m3']);
    fillSelect($('st-fusion'), (data.fusion_algorithms || []).map(function (a) {
      return { value: a.id, label: a.label };
    }), ['weighted_average']);
    fillSelect($('st-rerank'), (data.rerank_models || []).map(function (a) {
      return { value: a.id, label: a.label };
    }), ['rule']);
    fillSelect($('st-fields'), (data.fulltext_fields || []).map(function (f) {
      return { value: f.id, label: f.label + '（加成 ×' + f.default_boost + '）' };
    }), ['content']);
  } catch (err) {
    $('st-summary').textContent = '策略清单加载失败（界面仍可用，将使用内置默认值）：' + err;
  }
  syncStratVisibility();
  syncWeightLabels();
  refreshStratSummary();
}

function applyPreset() {
  const name = stratMode();
  const s = (STRAT_META && STRAT_META.presets && STRAT_META.presets[name]) || null;
  if (s) {
    $('st-embed').value = s.embedding_model || 'bge-m3';
    $('st-fusion').value = s.fusion || 'weighted_average';
    $('st-rerank').value = s.rerank_model || 'rule';
    $('st-vw').value = (s.vector_weight === undefined ? 0.5 : s.vector_weight);
    $('st-topk').value = s.top_k || 5;
    $('st-vk').value = s.vector_k || 20;
    $('st-fk').value = s.fulltext_k || 20;
  }
  syncWeightLabels();
  refreshStratSummary();
  setStatus('已套用「' + name + '」预设。');
}

function renderCompare(question, rows) {
  let html = '<div style="font-weight:600;margin-bottom:4px">检索策略对比：' + esc(question) + '</div>';
  html += '<table class="cmp"><tr><th>模式</th><th>答案</th><th>Top-1 证据</th><th>召回(向量/全文)</th><th>耗时(ms)</th></tr>';
  rows.forEach(function (r) {
    if (!r.ok) {
      html += '<tr><td class="m">' + esc(r.mode) + '</td><td colspan="4">失败：' + esc(r.error || '见状态栏') + '</td></tr>';
      return;
    }
    const d = r.data || {};
    const base = d.base || {};
    const dg = (base.diagnostics || {}).strategy || {};
    const t = dg.timings || {};
    const v = dg.vector || {}, f = dg.fulltext || {};
    const hits = ((base.diagnostics || {}).hits || []);
    const top1 = hits.length ? (hits[0].chunk_id + ' · p' + hits[0].page) : '—';
    const ans = (d.answer || '').split('\n')[0];
    const lat = (d.latency_ms || {}).total;
    html += '<tr><td class="m">' + esc(r.mode) + '</td><td>' + esc(ans.slice(0, 160)) +
      (ans.length > 160 ? '…' : '') + '</td><td>' + esc(top1) + '</td><td>' +
      ((v.recalled === undefined ? '—' : v.recalled) + ' / ' + (f.recalled === undefined ? '—' : f.recalled)) +
      '</td><td>' + (lat === undefined ? '—' : lat) + '（检索 ' + (t.total_ms === undefined ? '—' : t.total_ms) + '）</td></tr>';
  });
  html += '</table>';
  bubble('sys', html);
  scrollBottom();
}

async function compareModes() {
  const box = $('q');
  let question = box.value.trim();
  if (!question && TURNS.length) { question = TURNS[TURNS.length - 1].question; }
  if (!question) { setStatus('请先在输入框写下要对比的问题，再点「三模式对比」。', true); return; }
  if (BUSY) { return; }
  BUSY = true;
  $('st-compare').disabled = true;
  setStatus('正在用「向量 / 全文 / 混合」三种模式分别检索并生成，请稍候…');
  const base = currentStrategy();
  const rows = [];
  const modes = ['vector', 'fulltext', 'hybrid'];
  for (let i = 0; i < modes.length; i++) {
    const mode = modes[i];
    const st = Object.assign({}, base, { mode: mode });
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question, session_id: SESSION + '-cmp-' + mode,
          language: 'auto', use_cache: false, strategy: st })
      });
      const data = await resp.json().catch(function () { return {}; });
      rows.push({ mode: mode, ok: (resp.ok && data.ok !== false), data: data,
        error: (data.error && data.error.message) || ('HTTP ' + resp.status) });
    } catch (err) {
      rows.push({ mode: mode, ok: false, error: String(err) });
    }
  }
  renderCompare(question, rows);
  BUSY = false;
  $('st-compare').disabled = false;
  setStatus('');
}

function setStrategySidebar(base) {
  const dg = base && base.diagnostics ? (base.diagnostics.strategy || null) : null;
  if (!dg) {
    ['s-st-label', 's-st-channels', 's-st-fusion', 's-st-timing', 's-st-notes'].forEach(function (id) { setText(id, '-'); });
    return;
  }
  const v = dg.vector || {}, f = dg.fulltext || {};
  setText('s-st-label', dg.strategy_label || '-');
  setText('s-st-channels', '向量 ' + (v.recalled === undefined ? '—' : v.recalled) +
    ' 条（模型 ' + (v.model || '—') + '，最高相似度 ' + (v.top_similarity === undefined ? '—' : v.top_similarity) + '） ／ ' +
    '全文 ' + (f.recalled === undefined ? '—' : f.recalled) + ' 条（布尔命中 ' +
    (f.boolean_hits === undefined ? '—' : f.boolean_hits) + '，解析 ' + (f.query_ast || '—') + '）');
  const fu = dg.fusion || {}, rr = dg.rerank || {};
  setText('s-st-fusion', (fu.algorithm_label || fu.algorithm || '—') +
    '（向量 ' + (fu.vector_weight === undefined ? '—' : fu.vector_weight) +
    ' / 全文 ' + (fu.fulltext_weight === undefined ? '—' : fu.fulltext_weight) + '） ／ ' +
    (rr.applied || '—') + (rr.degraded ? '（已降级：' + (rr.reason || '') + '）' : ''));
  const t = dg.timings || {};
  setText('s-st-timing', '检索合计 ' + (t.total_ms === undefined ? '—' : t.total_ms) + ' ms ｜ 向量 ' +
    (t.vector_ms === undefined ? '—' : t.vector_ms) + ' ／ 全文 ' + (t.fulltext_ms === undefined ? '—' : t.fulltext_ms) +
    ' ／ 融合 ' + (t.fusion_ms === undefined ? '—' : t.fusion_ms) + ' ／ 重排 ' +
    (t.rerank_ms === undefined ? '—' : t.rerank_ms));
  const notes = (dg.notes || []).slice(0, 4);
  const oov = (f.oov_terms || []);
  setText('s-st-notes', (notes.length ? notes.join('；') : '无') +
    (oov.length ? '｜词表外词项：' + oov.slice(0, 6).join('、') : ''));
}

async function send() {
  const box = $('q');
  const question = box.value.trim();
  if (!question) { setStatus('请输入问题后再发送：空输入不会被检索（避免无意义的改写与检索开销）。', true); return; }
  if (question.length > 500) { setStatus('问题过长（>500 字），请精简后再发送。', true); return; }
  if (BUSY) { return; }
  BUSY = true; $('send').disabled = true;
  setStatus('检索与生成中…');
  addUser(question);
  box.value = '';
  const t0 = performance.now();
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, session_id: SESSION, language: 'auto', use_cache: true,
        strategy: currentStrategy() })
    });
    const data = await resp.json().catch(() => ({ ok: false, error: { message: '响应不是合法 JSON' } }));
    if (!resp.ok || data.ok === false) {
      const e = data && data.error ? data.error : {};
      addError((e.code ? '[' + e.code + '] ' : '') + (e.message || ('HTTP ' + resp.status)));
      setSidebar(null);
      setStrategySidebar(null);
      return;
    }
    renderTurn(data, Math.round(performance.now() - t0));
    setStrategySidebar(data.base || {});
    setStatus('');
  } catch (err) {
    addError('请求失败：' + err);
  } finally {
    BUSY = false; $('send').disabled = false;
  }
}

async function clearAll() {
  const old = SESSION;
  SESSION = newSessionId();
  TURNS.length = 0;
  $('chat').innerHTML = '';
  setSidebar(null);
  addSystem('已开启新会话 ' + SESSION + '（旧会话 ' + old + ' 已在服务端 reset）。');
  try {
    const resp = await fetch('/api/reset', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: old })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      setStatus('服务端重置未成功（不影响本地换新会话）：' +
        ((data.error && data.error.message) || ('HTTP ' + resp.status)), true);
    }
  } catch (err) {
    setStatus('服务端重置未成功（不影响本地换新会话）：' + err, true);
  }
}

function renderDemos() {
  const box = $('demos');
  DEMOS.forEach(d => {
    const b = document.createElement('button');
    b.textContent = '[' + d.tag + '] ' + d.question.slice(0, 26) + (d.question.length > 26 ? '…' : '');
    b.title = d.question + '（点击填入输入框）';
    b.onclick = () => { $('q').value = d.question; $('q').focus(); };
    box.appendChild(b);
  });
}

$('send').onclick = send;
$('clear').onclick = clearAll;
$('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
// 工单6：检索策略控件
$('st-mode').onchange = function () { syncStratVisibility(); refreshStratSummary(); };
$('st-vw').oninput = function () { syncWeightLabels(); refreshStratSummary(); };
$('st-th').oninput = syncWeightLabels;
$('st-ms').oninput = syncWeightLabels;
['st-embed', 'st-fusion', 'st-rerank', 'st-fields', 'st-op', 'st-fuzzy', 'st-topk', 'st-vk', 'st-fk']
  .forEach(function (id) { $(id).onchange = refreshStratSummary; });
$('st-preset').onclick = applyPreset;
$('st-compare').onclick = compareModes;
loadStrategies();
renderDemos();
setSidebar(null);
setStrategySidebar(null);
addSystem('新会话 ' + SESSION + '。右侧「检索策略」可切换 向量检索 / 全文检索 / 混合检索，并调整权重与融合算法；'
  + '点「三模式对比」可用同一问题一次跑完三种模式并列表对照。');
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- HTTP
class _Handler(BaseHTTPRequestHandler):
    """单请求处理：全部路由都在这里（无框架）。"""

    server_version = "Gongdan6HybridRetrievalWeb/2.0"
    protocol_version = "HTTP/1.1"

    #: 本次请求是否已经回过包（避免异常路径重复回包，破坏 HTTP/1.1 连接）
    _responded = False

    # -- 响应工具 --------------------------------------------------------
    def _send(self, body: bytes, status: int, content_type: str) -> None:
        self._responded = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):    # 客户端提前断开：不是服务错误
            pass

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), status,
                   "application/json; charset=utf-8")

    def _html(self, html: str, status: int = 200) -> None:
        self._send(html.encode("utf-8"), status, "text/html; charset=utf-8")

    def _error(self, code: str, message: str, status: int) -> None:
        self._json({"ok": False, "error": {"code": code, "message": message}}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 —— 覆盖默认 stderr 输出
        print(f"[web] {self.client_address[0]} {fmt % args}", flush=True)

    # -- 路由 ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 约定
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (_PAGE_HTML.replace("__TITLE__", TITLE)
                    .replace("__PORT__", str(self.server.server_port))          # type: ignore[attr-defined]
                    .replace("__DEMOS__", json.dumps(DEMO_QUESTIONS, ensure_ascii=False)))
            self._html(html)
            return
        if path in ("/health", "/api/health"):
            payload = self._health_payload()
            self._json(payload, int(payload.pop("_status", 200)))
            return
        if path == "/favicon.ico":
            self._send(b"", 204, "image/x-icon")
            return
        if path in ("/api/strategies", "/strategies"):
            self._json(self._strategies_payload())
            return
        if path.startswith(REPORTS_PREFIX):
            self._serve_report(path)
            return
        self._error("E_NOT_FOUND", f"未知路径 {path}", 404)

    def _serve_report(self, path: str) -> None:
        """只读暴露 ``测试/测试数据/**``（评测报告 / 截图页）；带路径穿越防护。"""
        rel = unquote(path[len(REPORTS_PREFIX):]).replace("\\", "/").lstrip("/")
        root = REPORTS_ROOT.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)               # 越界（../）直接拒
        except ValueError:
            self._error("E_FORBIDDEN", "路径越界：只允许访问 测试/测试数据 目录下的文件", 403)
            return
        if target.is_dir():
            files = sorted(p.name for p in target.iterdir() if p.is_file())
            links = "".join(
                f'<li><a href="{REPORTS_PREFIX}{rel.rstrip("/")}/{n}">{n}</a></li>' for n in files)
            self._html(f"<!DOCTYPE html><meta charset='utf-8'><h3>{rel}</h3><ul>{links}</ul>")
            return
        if not target.is_file():
            self._error("E_NOT_FOUND", f"报告文件不存在：{rel}", 404)
            return
        ctype = ("text/html; charset=utf-8" if target.suffix.lower() in (".html", ".htm")
                 else "application/json; charset=utf-8" if target.suffix.lower() == ".json"
                 else "text/plain; charset=utf-8" if target.suffix.lower() in (".md", ".csv", ".jsonl")
                 else "application/octet-stream")
        self._send(target.read_bytes(), 200, ctype)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/api/chat", "/chat"):
                self._on_chat()
                return
            if path in ("/api/feedback", "/feedback"):
                self._on_feedback()
                return
            if path in ("/api/reset", "/reset"):
                self._on_reset()
                return
        except Exception as exc:  # noqa: BLE001 —— 未预期异常必须带中文说明返回，不裸 500
            if not self._responded:
                self._error("E_INTERNAL", f"服务内部错误：{type(exc).__name__}: {exc}", 500)
            return
        self._error("E_NOT_FOUND", f"未知路径 {path}", 404)

    # -- 各端点 ----------------------------------------------------------
    def _strategies_payload(self) -> dict[str, Any]:
        """工单6：检索策略清单（供界面下拉框/滑块初始化）。

        策略层不可用时**不报 5xx** —— 界面是有内置默认值的，返回 ``ok=False`` + 中文原因即可，
        保证「引擎没起来也能打开页面看到原因」。
        """
        try:
            from app.core import retrieval_strategy as rs            # noqa: PLC0415 —— 延迟导入
        except ImportError as exc:
            return {"ok": False, "work_order": WORK_ORDER,
                    "error": {"code": "E_STRATEGY_UNAVAILABLE",
                              "message": f"检索策略模块不可用：{type(exc).__name__}: {exc}"}}
        try:
            index_root = REPO_ROOT / "研发" / "data" / "index"
            models = []
            for spec in rs.list_models_for_ui(index_root=index_root):
                models.append(spec)
        except Exception as exc:  # noqa: BLE001 —— 模型探测失败不影响其余选项
            models = []
            probe_error = f"{type(exc).__name__}: {exc}"
        else:
            probe_error = None
        return {
            "ok": True,
            "work_order": WORK_ORDER,
            "modes": rs.describe_modes(),
            "fusion_algorithms": rs.fusion_mod.describe_algorithms(),
            "rerank_models": rs.describe_rerank_models(),
            "embedding_models": models,
            "embedding_probe_error": probe_error,
            "fulltext_fields": rs.describe_fulltext_fields(),
            "fulltext_syntax_help": rs.fulltext_syntax_help(),
            "defaults": rs.StrategyConfig().to_dict(),
            "presets": {name: cfg.to_dict() for name, cfg in rs.PRESETS.items()},
        }

    def _health_payload(self) -> dict[str, Any]:
        engine, meta = get_engine_or_none()
        warmup = (meta or {}).get("warmup") or {}
        payload: dict[str, Any] = {
            "status": "degraded", "version": VERSION, "service": SERVICE,
            "work_order": WORK_ORDER, "repo_root": str(REPO_ROOT),
            "endpoints": ENDPOINTS, "feedback_log": str(FEEDBACK_PATH),
            "engine_ready": bool((meta or {}).get("ready")),
            "warmed": bool(warmup),
            "warmup": ({"ok": warmup.get("ok"), "elapsed_ms": warmup.get("elapsed_ms")}
                       if warmup else None),
            "active_sessions": len(_SESSION_LOCKS),
        }
        if engine is None:
            payload["checks"] = {"engine": {"ok": False,
                                            "reason": ((meta or {}).get("error") or {}).get("message",
                                                                                             "引擎未就绪")}}
            payload["error"] = (meta or {}).get("error") or {"code": "E_ENGINE_UNAVAILABLE"}
            payload["_status"] = 503
            return payload
        try:
            health = engine.health()
        except Exception as exc:  # noqa: BLE001 —— 如实报告
            payload["checks"] = {"engine": {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}}
            payload["error"] = {"code": "E_INTERNAL", "message": str(exc)}
            payload["_status"] = 503
            return payload
        payload["engine"] = health
        if isinstance(health, dict):
            payload["checks"] = health.get("checks")
            payload["stats"] = health.get("stats")
            payload["status"] = health.get("status", "degraded")
        else:
            payload["status"] = "ok"
        payload["_status"] = 200 if payload["status"] == "ok" else 503
        return payload

    def _engine_or_503(self) -> Any:
        engine, meta = get_engine_or_none()
        if engine is None:
            err = (meta or {}).get("error") or {}
            self._error(err.get("code", "E_ENGINE_UNAVAILABLE"),
                        err.get("message", "多轮引擎未就绪"), 503)
            return None
        return engine

    def _on_chat(self) -> None:
        payload = self._body()
        question = str(payload.get("question") or "").strip()
        if not question:
            self._error("E_PARAM_QUESTION_EMPTY",
                        "请输入问题后再发送：空输入不会被检索（多轮界面里请直接输入问句）。", 400)
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._error("E_PARAM_QUESTION_TOO_LONG",
                        f"问题过长（{len(question)} 字 > {MAX_QUESTION_CHARS} 字），请精简后再发送。", 400)
            return
        session_id = str(payload.get("session_id") or "default").strip() or "default"
        language = str(payload.get("language") or payload.get("lang") or "auto")
        use_cache = bool(payload.get("use_cache", True))
        # 工单6：解析并**校验**检索策略；不合法直接 400 报错，不静默改用默认策略
        strategy = None
        raw_strategy = payload.get("strategy")
        if raw_strategy:
            try:
                from app.core import retrieval_strategy as rs         # noqa: PLC0415 —— 延迟导入
                strategy = rs.StrategyConfig.from_dict(raw_strategy)
                problems = strategy.validate()
            except Exception as exc:  # noqa: BLE001 —— 配置解析失败按参数错误返回
                self._error("E_PARAM_STRATEGY_INVALID",
                            f"检索策略参数不合法：{type(exc).__name__}: {exc}", 400)
                return
            if problems:
                self._error("E_PARAM_STRATEGY_INVALID",
                            "检索策略参数不合法：" + "；".join(problems), 400)
                return
        engine = self._engine_or_503()
        if engine is None:
            return
        started = time.perf_counter()
        with session_lock(session_id):
            try:
                result = engine.ask(question, session_id=session_id, lang=language,
                                    use_cache=use_cache, strategy=strategy)
            except Exception as exc:  # noqa: BLE001 —— 引擎异常按 503/500 如实返回
                code = "E_INDEX_MISSING" if "IndexMissing" in type(exc).__name__ else "E_INTERNAL"
                status = 503 if code == "E_INDEX_MISSING" else 500
                self._error(code, f"多轮问答失败：{type(exc).__name__}: {exc}", status)
                return
            body = result.to_dict()
            try:
                body["history"] = engine.history(session_id)
            except Exception:  # noqa: BLE001 —— history 只是附加信息，失败不致命
                body["history"] = []
            try:
                body["state"] = engine.state(session_id)
            except Exception:  # noqa: BLE001
                body["state"] = {}
        body["ok"] = True
        body["session_id"] = body.get("session_id") or session_id
        body["latency_ms"] = body.get("latency_ms") or {}
        body.setdefault("wall_ms", round((time.perf_counter() - started) * 1000, 2))
        self._json(body)

    def _on_feedback(self) -> None:
        payload = self._body()
        try:
            out = append_feedback(payload, source="web_app")
        except ValueError as exc:
            self._error("E_PARAM_VOTE_INVALID", f"反馈参数不合法：{exc}", 400)
            return
        except OSError as exc:
            self._error("E_FEEDBACK_WRITE_FAILED",
                        f"反馈写入失败（{FEEDBACK_PATH}）：{type(exc).__name__}: {exc}", 500)
            return
        self._json(out)

    def _on_reset(self) -> None:
        payload = self._body()
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            self._error("E_PARAM_SESSION_ID_EMPTY", "session_id 不得为空（清空对话需要指定要重置的会话）。", 400)
            return
        engine = self._engine_or_503()
        if engine is None:
            return
        with session_lock(session_id):
            try:
                engine.reset(session_id)
            except Exception as exc:  # noqa: BLE001
                self._error("E_INTERNAL", f"会话重置失败：{type(exc).__name__}: {exc}", 500)
                return
        self._json({"ok": True, "session_id": session_id, "reset": True})


# --------------------------------------------------------------------------- 启动
def build_server(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """构造 ``ThreadingHTTPServer``（并发；每个连接一个线程）。"""
    server = ThreadingHTTPServer((host, int(port)), _Handler)
    server.daemon_threads = True
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="web_app.py",
        description=f"工单5 多轮对话兜底 Web 界面（纯标准库）：{WORK_ORDER}")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"监听端口（默认 {DEFAULT_PORT}；与 8600/7860 错开）")
    parser.add_argument("--no-warmup", action="store_true",
                        help="跳过启动预热（仅调试；首个问题会背负冷启动延迟）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    host, port = str(args.host), int(args.port)

    print(f"[启动] {SERVICE} —— {WORK_ORDER}", flush=True)
    print(f"[启动] 仓库根 {REPO_ROOT}", flush=True)
    print(f"[启动] 引擎预热中（首次加载索引 + 嵌入 + 生成模型，请稍候）…", flush=True)
    _, meta = build_engine(warmup=not args.no_warmup)
    if not meta.get("ready"):
        err = (meta.get("error") or {}).get("message", "未知原因")
        print("[警告] 引擎未就绪：服务仍会启动，/health 与 /api/chat 将如实返回 503。", flush=True)
        print(f"[警告] 原因：{err}", flush=True)

    try:
        server = build_server(host=host, port=port)
    except OSError as exc:
        print(f"[失败] 无法监听 {host}:{port} —— {type(exc).__name__}: {exc}", flush=True)
        print("       端口可能被占用：netstat -ano | findstr :%d   → 换端口用 --port <n>" % port,
              flush=True)
        return EXIT_PARAM

    print(f"[就绪] 界面地址 http://{host}:{port}/   （Ctrl+C 停止）", flush=True)
    print(f"[就绪] 健康检查 http://{host}:{port}/health ；反馈留痕 {FEEDBACK_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 收到 Ctrl+C，正在关闭…", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
