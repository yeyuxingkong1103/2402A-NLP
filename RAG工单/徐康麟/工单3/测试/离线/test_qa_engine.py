# -*- coding: utf-8 -*-
"""T7 端到端验收：问答引擎 + 存储层 + 两个界面（共用同一套 app/core 逻辑）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定项（全部实测，退出码 0 = 全过）：
    A. 存储层：``SQLiteManager`` 建表/计数/回查/反馈/轨迹；
    B. 引擎：``warmup()`` 真调用分词器+嵌入+探测，``files()`` 来自自动发现，``ask()`` 带引用、
       ``stream()`` 首字、会话与消息落库、``feedback``/``clear`` 生效；
    C. 备用界面：真起 ``ThreadingHTTPServer``，用 ``http.client`` 打 ``/``、``/api/health``、
       ``/api/files``、``/api/ask``、``/api/feedback``，断言答案带引用且首字 ≤3000 ms；
    D. 界面共用：AST 断言 ``serve_fallback.py`` / ``streamlit_app.py`` 只从 ``qa_engine`` 取数据，
       且**不**各自实现检索/生成（禁止 import retriever/generator/embedder/bm25 等）；
    E. Streamlit 应用：本机无 streamlit → 用**桩模块**注入 ``sys.modules`` 后 import 并调用 ``main()``，
       断言渲染流程真的走了 ``QAEngine``（files/multistream/chat_input/expander 等被调用）。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/test_qa_engine.py
"""

from __future__ import annotations

import ast
import http.client
import io
import json
import sys
import threading
import time
import types
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import build_engine  # noqa: E402
from app.storage.sqlite_manager import SQLiteManager  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录并打印一条断言。"""
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'✅' if ok else '❌'} {name}{('　' + detail) if detail else ''}")


def _module_imports(path: Path) -> set[str]:
    """AST 抽取一个模块里所有 import 的名字（用于「不得各写一套业务逻辑」的机器判定）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            names |= {f"{module}.{a.name}" for a in node.names}
    return names


def test_storage(cfg) -> None:
    """A. 存储层。"""
    print("\n[A] SQLiteManager")
    mgr = SQLiteManager(cfg=cfg)
    check("建表含 feedback", any(r[0] == "feedback" for r in
                               mgr._conn().execute("SELECT name FROM sqlite_master WHERE type='table'")))
    count_all = mgr.count_chunks()
    check("chunks 计数 > 0", count_all > 0, f"{count_all} 块")
    check("按文件计数一致", mgr.count_chunks(file_name="招股说明书1.pdf") > 0,
          f"{mgr.count_chunks(file_name='招股说明书1.pdf')} 块")
    chunk_id = mgr._conn().execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()[0]
    chunk = mgr.fetch_chunk(chunk_id)
    check("fetch_chunk 回查到块", chunk is not None and chunk.chunk_id == chunk_id,
          f"{chunk_id} → p{getattr(chunk, 'page', None)}")
    batch = mgr.fetch_chunks([chunk_id])
    check("fetch_chunks 批量回查", len(batch) == 1)
    check("page_count 可查", mgr.page_count("招股说明书1.pdf") == 548,
          f"实际 {mgr.page_count('招股说明书1.pdf')}")
    mgr.record_retrieval({"trace_id": "t7test", "session_id": "s7", "question": "测试",
                          "file_names": ["招股说明书1.pdf"], "top_k": 5, "chunk_ids": [chunk_id],
                          "scores": [0.5], "stages": {"total_ms": 1.0}})
    row = mgr._conn().execute("SELECT COUNT(*) FROM retrieval_traces WHERE trace_id='t7test'").fetchone()[0]
    check("retrieval_traces 落行", row == 1)
    mgr.record_run(run_id="r7test", kind="unit", stats={"k": 1}, ok=True)
    row = mgr._conn().execute("SELECT ok FROM runs WHERE run_id='r7test'").fetchone()[0]
    check("runs 落行", row == 1)
    fid = mgr.record_feedback(rating="up", session_id="s7", answer_id="a7", trace_id="t7test")
    check("feedback 落行", fid > 0 and mgr.list_feedback(session_id="s7")[0]["rating"] == "up",
          f"id={fid}")
    bad = False
    try:
        mgr.record_feedback(rating="maybe")
    except Exception as exc:  # noqa: BLE001 —— 这里就是断言必须抛
        bad = "StorageError" in type(exc).__name__ or "rating" in str(exc)
    check("非法 rating 显式报错（不静默）", bad)


def test_fallback_http(engine, cfg) -> None:
    """C. 备用界面真起 HTTP 服务并实测。"""
    print("\n[C] serve_fallback（纯标准库界面）")
    sys.path.insert(0, str(REPO_ROOT / "研发"))
    from app.ui import serve_fallback

    httpd = serve_fallback.serve(host="127.0.0.1", port=0, engine=engine)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        conn.request("GET", "/")
        resp = conn.getresponse()
        html = resp.read().decode("utf-8")
        check("GET / 返回内联 HTML", resp.status == 200 and "<title>工单3" in html and "http://" not in html.split("</head>")[0],
              f"{len(html)} B，无外部静态依赖")
        conn.request("GET", "/api/health")
        health = json.loads(conn.getresponse().read().decode("utf-8"))
        check("GET /api/health 正常", health.get("ok") is True)
        conn.request("GET", "/api/files")
        files = json.loads(conn.getresponse().read().decode("utf-8"))["files"]
        check("GET /api/files 返回自动发现列表", len(files) == 2, json.dumps([f["file_name"] for f in files], ensure_ascii=False))
        body = json.dumps({"question": "武汉兴图新科电子股份有限公司法定代表人是谁？", "top_k": 5})
        conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
        payload = json.loads(conn.getresponse().read().decode("utf-8"))
        check("POST /api/ask 返回答案（§7 契约字段名 text）", bool(payload.get("text")), str(payload.get("text"))[:50])
        check("POST /api/ask 含 chunks（content_digest 摘要）",
              bool(payload.get("chunks")) and "content_digest" in payload["chunks"][0],
              f"{len(payload.get('chunks') or [])} 块")
        check("HTTP 答案带引用", bool(payload.get("citations")),
              " ".join(f"[{c['file_name']}: {c['page']}]" for c in payload.get("citations") or []))
        check("HTTP 首字 ≤3000 ms", float(payload.get("first_token_ms") or 0) <= 3000,
              f"{payload.get('first_token_ms')} ms")
        check("HTTP 返回检索片段与页码（契约字段 chunks）", bool(payload.get("chunks")))
        sid = payload.get("session_id")
        body = json.dumps({"rating": "up", "session_id": sid, "answer_id": payload.get("answer_id")})
        conn.request("POST", "/api/feedback", body=body, headers={"Content-Type": "application/json"})
        fb = json.loads(conn.getresponse().read().decode("utf-8"))
        check("POST /api/feedback 落库", int(fb.get("feedback_id") or 0) > 0)
        body = json.dumps({"session_id": sid})
        conn.request("POST", "/api/clear", body=body, headers={"Content-Type": "application/json"})
        cleared = json.loads(conn.getresponse().read().decode("utf-8"))
        check("POST /api/clear 生效", int(cleared.get("removed") or 0) >= 2, f"删除 {cleared.get('removed')} 条")
        conn.request("POST", "/api/ask", body=json.dumps({"question": ""}), headers={"Content-Type": "application/json"})
        empty = conn.getresponse()
        check("空问题返回 400（显式拒绝）", empty.status == 400, f"status={empty.status}")
        conn.request("POST", "/api/ask", body=json.dumps({"question": "注册资本是多少？",
                                                          "file_names": ["不存在的文件.pdf"]}),
                     headers={"Content-Type": "application/json"})
        missing = conn.getresponse()
        check("不存在的 file_names 返回 400（§7 契约）", missing.status == 400, f"status={missing.status}")
    finally:
        conn.close()
        httpd.shutdown()
        httpd.server_close()


def test_ui_sharing() -> None:
    """D. 两个界面共用 app/core（AST 机器判定）。"""
    print("\n[D] 界面共用同一套业务逻辑")
    banned = {"app.core.retriever", "app.core.generator", "app.core.embedder", "app.core.bm25_index",
              "app.core.vector_store", "app.core.chunker", "app.core.citation"}
    for name in ("serve_fallback.py", "streamlit_app.py"):
        path = REPO_ROOT / "研发" / "app" / "ui" / name
        imports = _module_imports(path)
        hits = sorted(i for i in imports if any(i.startswith(b) for b in banned))
        check(f"{name} 不直接依赖检索/生成模块", not hits, f"命中 {hits}" if hits else "只用 qa_engine")
        check(f"{name} 使用 qa_engine", any("qa_engine" in i for i in imports))
    st_src = (REPO_ROOT / "研发" / "app" / "ui" / "streamlit_app.py").read_text(encoding="utf-8")
    check("streamlit_app 模块顶层 import streamlit", "\nimport streamlit as st" in st_src)


def test_streamlit_with_stub(engine) -> None:
    """E. 用桩 streamlit 真跑一遍渲染流程（本机没有 streamlit 也能验证界面逻辑）。"""
    print("\n[E] Streamlit 应用（桩模块实测渲染流程）")
    calls: dict[str, list] = {}

    def _record(name):
        def _fn(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            if name == "columns":
                return [types.SimpleNamespace(button=_record("fb_button"), caption=_record("caption"))
                        for _ in args[0]]
            if name == "cache_resource":
                def _decorator(fn):
                    return fn
                return _decorator
            # 与真实 Streamlit 的返回类型保持一致，避免界面代码对 None 做迭代
            if name == "multiselect":
                return []
            if name == "slider":
                return int(kwargs.get("value") or 5)
            if name == "button":
                return False
            return None
        return _fn

    class _SessionState(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:  # noqa: TRY003
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    class _Ctx:
        """通用上下文管理器桩（sidebar / expander / spinner / chat_message 共用）。"""

        def __init__(self, *args, **kwargs):
            self._name = args[0] if args else "ctx"

        def __enter__(self):
            calls.setdefault("ctx_enter", []).append((self._name, {}))
            return types.SimpleNamespace(markdown=_record("ctx.markdown"), text=_record("ctx.text"),
                                         code=_record("ctx.code"), caption=_record("ctx.caption"),
                                         header=_record("ctx.header"), multiselect=_record("ctx.multiselect"),
                                         slider=_record("ctx.slider"), button=_record("ctx.button"),
                                         divider=_record("ctx.divider"), write=_record("ctx.write"),
                                         success=_record("ctx.success"))

        def __exit__(self, *exc):
            return False

    stub = types.ModuleType("streamlit")
    for fn in ("set_page_config", "title", "caption", "header", "multiselect", "slider",
               "button", "divider", "success", "write", "markdown", "text", "code",
               "columns", "chat_input", "toast", "rerun", "cache_resource"):
        setattr(stub, fn, _record(fn))
    stub.session_state = _SessionState()
    stub.sidebar = _Ctx("sidebar")

    def _chat_input(*args, **kwargs):
        calls.setdefault("chat_input", []).append((args, kwargs))
        return "武汉兴图新科电子股份有限公司注册资本是多少？"

    stub.chat_input = _chat_input
    stub.chat_message = _Ctx
    stub.expander = _Ctx
    stub.spinner = _Ctx
    sys.modules["streamlit"] = stub

    # 让界面复用已预热的引擎（避免二次加载索引）
    import app.ui.streamlit_app as st_app

    st_app._engine = lambda: engine  # 覆写缓存装饰器包装前的取引擎函数
    st_app.main()
    check("渲染调用了 set_page_config/title", bool(calls.get("set_page_config")) and bool(calls.get("title")))
    multiselect_calls = (calls.get("multiselect") or calls.get("ctx.multiselect")
                         or calls.get("sidebar.multiselect") or [])
    detail = json.dumps(multiselect_calls[0][0][0], ensure_ascii=False)[:120] if multiselect_calls else "未记录"
    check("侧栏用 engine.files() 列出 PDF", bool(multiselect_calls), detail)
    check("进入了 sidebar 上下文", any(name == "sidebar" for name, _ in calls.get("ctx_enter", [])))
    check("读取了 chat_input 问题", bool(calls.get("chat_input")))
    check("渲染了答案/引用展开块", bool(calls.get("ctx.markdown")) or bool(calls.get("markdown")),
          f"markdown 调用 {len(calls.get('ctx.markdown', [])) + len(calls.get('markdown', []))} 次")
    check("历史写入 session_state", "history" in stub.session_state,
          f"history={len(stub.session_state.get('history', []))} 条")


def test_contract_and_shape(cfg, engine) -> None:
    """F. §7 契约形状与块形态鲁棒性（t14 修复项）。"""
    print("\n[F] §7 契约形状与块形态（t14）")
    import sqlite3
    import types as _types

    from app.core import retrieval_utils

    health = engine.health()
    check("health.index.count > 0", int((health.get("index") or {}).get("count", 0)) > 0,
          f"count={(health.get('index') or {}).get('count')} dim={(health.get('index') or {}).get('dim')}")
    check("health.files 是数量(int) 且 ≥2", isinstance(health.get("files"), int) and health["files"] >= 2,
          f"files={health.get('files')!r}")
    check("health.llm.available 存在", "available" in (health.get("llm") or {}))

    raised = False
    try:
        engine.validate_files(["不存在的文件.pdf"])
    except Exception as exc:  # noqa: BLE001 —— 必须显式报错（HTTP 层转 400）
        raised = "RAG-6001" in str(exc) or "file_names" in str(exc)
    check("validate_files 对不存在的文件显式报错", raised)
    check("validate_files 对合法文件返回原名", engine.validate_files(["招股说明书1.pdf"]) == ["招股说明书1.pdf"])

    conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))
    row = conn.execute("SELECT chunk_id,file_name,page,type,content FROM chunks WHERE chunk_id=?",
                       ("招股说明书1_p0490_x1131",)).fetchone()
    conn.close()
    chunk_dict = {"chunk_id": row[0], "file_name": row[1], "page": row[2], "type": row[3], "content": row[4]}
    obj = _types.SimpleNamespace(chunk_id=row[0], file_name=row[1], page=row[2], type=row[3], content=row[4])
    evidence = "拟使用本次发行募集资金15,000 万元用于补充流动资金"
    check("is_evidence_hit：dict/对象/纯文本一致为 True",
          retrieval_utils.is_evidence_hit([chunk_dict], evidence)
          and retrieval_utils.is_evidence_hit([obj], evidence)
          and retrieval_utils.is_evidence_hit([row[4]], evidence))
    check("换行变体也命中（squash 归一化）",
          retrieval_utils.is_evidence_hit([chunk_dict], "拟使用本次发行募集资\n金15,000 万元用于补充流动资金"))
    strict_raised = False
    try:
        retrieval_utils.is_evidence_hit([{"chunk_id": "x"}], evidence, strict=True)
    except Exception:  # noqa: BLE001 —— strict 必须抛
        strict_raised = True
    check("形状不匹配 strict=True 抛错（非静默）", strict_raised)
    check("形状不匹配默认仍返回 False（已留 WARN）",
          retrieval_utils.is_evidence_hit([{"chunk_id": "x"}], evidence) is False)

    try:
        from app.core import evaluator_bridge

        ok, reason = evaluator_bridge.check_answer("注册资本为5,520万元。", "注册资本为 5,520 万元。")
        check("evaluator_bridge 可用且阈值 0.62", bool(ok) and evaluator_bridge.FUZZY_THRESHOLD() == 0.62,
              f"判分={reason[:44]}")
    except Exception as exc:  # noqa: BLE001 —— 降级如实记录
        check("evaluator_bridge 可用且阈值 0.62", False, f"{type(exc).__name__}: {exc}")

    answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", top_k=cfg.retrieval.top_k)
    payload = engine.answer_payload(answer, session_id="t14-shape", wall_ms=1.0)
    required = ("answer_id", "text", "citations", "is_unknown", "unknown_reason", "first_token_ms",
                "total_ms", "backend", "model", "language", "session_id", "trace_id", "chunks")
    missing = [k for k in required if k not in payload]
    check("answer_payload 含 §7 全部字段", not missing, f"缺 {missing}" if missing else f"{len(required)} 个字段齐备")
    check("answer_payload.chunks 用 content_digest 摘要",
          bool(payload["chunks"]) and set(payload["chunks"][0]["content_digest"]) == {"chars", "head"},
          f"{len(payload['chunks'])} 块")


def main() -> int:
    """跑完 A~E 五组验收。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    print(f"环境：{cfg.llm.ollama_base_url} / {cfg.llm.ollama_gen_model} / 索引 {cfg.paths.index_dir}")
    test_storage(cfg)
    engine = build_engine(cfg=cfg, warmup=True)
    test_engine_with_engine(cfg, engine)
    test_fallback_http(engine, cfg)
    test_contract_and_shape(cfg, engine)
    test_ui_sharing()
    test_streamlit_with_stub(engine)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("\n" + "─" * 74)
    print(f"T7 端到端验收：✅{passed} / {len(CHECKS)}")
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  ❌ {name}　{detail}")
    shutdown_logging()
    return 0 if passed == len(CHECKS) else 1


def test_engine_with_engine(cfg, engine) -> None:
    """B 组：复用已构造引擎（避免重复预热）。"""
    print("\n[B] QAEngine")
    w = engine.warmup_info
    check("warmup 真调用分词器", isinstance(w.get("tokenizer"), dict) and w["tokenizer"].get("backend") == "jieba",
          f"cold_ms={w.get('tokenizer', {}).get('cold_ms')}")
    check("warmup 真调用嵌入", bool((w.get("embedding") or {}).get("dim")),
          f"dim={(w.get('embedding') or {}).get('dim')}")
    check("warmup 真做后端探测", w.get("backend") in {"ollama", "openai", "extractive"},
          f"backend={w.get('backend')}，probe={w.get('llm_probe_ms')} ms")
    files = engine.files()
    check("files() 来自自动发现（2 份 PDF）", len(files) == 2 and all(f["chunk_count"] > 0 for f in files),
          json.dumps([f["file_name"] for f in files], ensure_ascii=False))
    health = engine.health()
    check("health 含检索/生成/存储", health["ok"] and health["retriever"].get("chunks") == 2599
          and health["llm"].get("backend") == "ollama", f"chunks={health['retriever'].get('chunks')}")

    question = "武汉兴图新科电子股份有限公司的注册资本是多少？"
    answer = engine.ask(question, top_k=cfg.retrieval.top_k)
    check("ask 返回非空答案", bool(answer.text), answer.text.replace("\n", " ")[:60])
    check("ask 带真实引用（可回溯）", bool(answer.citations),
          " ".join(c.render() for c in answer.citations))
    check("ask 首字 ≤3000 ms", answer.first_token_ms <= 3000, f"{answer.first_token_ms} ms")
    sessions = engine.sessions(limit=5)
    check("ask 已建会话", bool(sessions), f"{len(sessions)} 个会话")
    sid = sessions[0]["session_id"]
    msgs = engine.messages(sid)
    check("会话消息成对（user+assistant）", len(msgs) >= 2 and {m["role"] for m in msgs} == {"user", "assistant"})
    src = engine.citation_source(answer.citations[0].file_name, answer.citations[0].page, limit=200)
    check("citation_source 能取到引用页原文", len(src) > 20, f"{len(src)} 字")
    t0 = time.perf_counter()
    first_ms, final = None, None
    for delta in engine.ask(question, stream=True, session_id=sid):
        if delta.is_first:
            first_ms = delta.first_token_ms
        if delta.done:
            final = delta.answer
    check("stream 首字 ≤3000 ms", first_ms is not None and first_ms <= 3000,
          f"{first_ms} ms（迭代 {round((time.perf_counter() - t0) * 1000)} ms）")
    check("stream 末块携带完整答案与引用", final is not None and bool(final.citations),
          f"{(final.text or '')[:40]!r}")
    fid = engine.feedback(rating="down", session_id=sid, answer_id=final.answer_id, trace_id=final.trace_id)
    check("engine.feedback 落库", fid > 0)
    removed = engine.clear_conversation(sid)
    check("clear_conversation 生效", removed >= 2, f"删除 {removed} 条消息")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底
        import traceback

        traceback.print_exc()
        shutdown_logging()
        raise SystemExit(1)
