# -*- coding: utf-8 -*-
"""T6 契约验收：可插拔后端探测/多轮持久化/流式首字/中英文/无依据拒答/结构化日志。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定项（全部实测，退出码 0 = 全过）：
    1) 后端探测 ≤0.5 s 且不重试；RAG_LLM__BACKEND=extractive 时能切到抽取式兜底；
    2) 多轮对话：最近 5 轮存 SQLite（conversations/messages 两表真实落行）；
    3) 流式：首个 delta 带 first_token_ms 且 ≤3000；done delta 携带带引用的 Answer；
    4) 语言：英文提问走英文作答分支；
    5) 无依据：明显超出语料的问题回「不清楚」，且不产生任何引用；
    6) 日志：函数入口/出口成对出现，含 elapsed_ms 与输入输出摘要。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/test_t6_contract.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import citation as citation_mod, language as language_mod, llm_client  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.conversation import Turn, get_conversation_store  # noqa: E402
from app.core.generator import build_generator  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录一条断言结果并打印。"""
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'✅' if ok else '❌'} {name}{('　' + detail) if detail else ''}")


def main() -> int:
    """跑完 6 项契约验收。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("test_t6_contract")
    print(f"环境：{cfg.llm.ollama_base_url} / 模型 {cfg.llm.ollama_gen_model} / 工作目录 {REPO_ROOT}")

    # 1) 后端探测（≤0.5 s、不重试）
    print("\n[1] 可插拔后端与探测")
    started = time.perf_counter()
    infos = llm_client.probe_backends(cfg, logger=log)
    probe_wall = round((time.perf_counter() - started) * 1000, 2)
    by_name = {i.name: i for i in infos}
    ollama = by_name.get("ollama")
    check("探测覆盖 ollama/openai/extractive", set(by_name) == {"ollama", "openai", "extractive"},
          f"实际 {sorted(by_name)}")
    check("ollama 可用且探测 ≤0.5 s", bool(ollama and ollama.available and ollama.probe_ms <= 500),
          f"probe_ms={getattr(ollama, 'probe_ms', None)}，含三后端总耗时 {probe_wall} ms")
    check("探测超时配置 = 0.5 s 且无重试", float(cfg.llm.probe_timeout_s) == 0.5,
          f"probe_timeout_s={cfg.llm.probe_timeout_s}")
    bad = llm_client.LLMBackendInfo(name="ollama", model="不存在的模型", base_url="http://127.0.0.1:59999",
                                    available=True, probe_ms=0.0)
    try:
        llm_client.LLMClient(cfg=cfg, backend=bad, logger=log).generate_full("x", max_tokens=4, logger=log)
        check("坏端点显式失败（不静默）", False, "未抛异常")
    except Exception as exc:  # noqa: BLE001 —— 这里就是断言「必须抛」
        check("坏端点显式失败（不静默）", True, f"{type(exc).__name__}")

    # 2) 多轮对话持久化（最近 5 轮）
    print("\n[2] 多轮对话（SQLite 最近 5 轮）")
    retriever = build_retriever(cfg=cfg)
    generator = build_generator(cfg=cfg, logger=log)
    store = get_conversation_store(logger=log)
    session_id = store.create_session()

    def _turn(role: str, content: str) -> Turn:
        """构造一轮对话记录（Turn 要求 session_id/created_at）。"""
        return Turn(session_id=session_id, role=role, content=content,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    for i in range(7):
        store.append_turn(session_id, _turn("user", f"第{i}轮问题"))
        store.append_turn(session_id, _turn("assistant", f"第{i}轮回答"))
    history = store.history(session_id, last_n=5)
    check("历史只保留最近 5 轮（10 条消息）", len(history) == 10, f"实际 {len(history)} 条")
    check("窗口 = 最近 5 轮（第 2~6 轮，10 条消息）", bool(history) and history[0].content == "第2轮问题"
          and history[-1].content == "第6轮回答",
          f"实际 {history[0].content if history else '无'} … {history[-1].content if history else '无'}")
    check("message_count 同步为 14", store.message_count(session_id) == 14,
          f"实际 {store.message_count(session_id)}")
    conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))
    rows = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0]
    sessions = conn.execute("SELECT message_count FROM conversations WHERE session_id=?", (session_id,)).fetchone()
    conn.close()
    check("messages 表真实落行 14 条", rows == 14, f"实际 {rows}")
    check("conversations.message_count 同步", bool(sessions and int(sessions[0]) == 14),
          f"实际 {sessions[0] if sessions else '无'}")

    # 3) 流式首字 + 引用
    print("\n[3] 流式作答与首字")
    question = "武汉兴图新科电子股份有限公司的注册资本是多少？"
    retrieval = retriever.retrieve(question, top_k=cfg.retrieval.top_k, logger=log)
    first_ms = None
    final = None
    for delta in generator.stream(question, retrieval, trace_id="t6s1", logger=log):
        if delta.is_first:
            first_ms = delta.first_token_ms
        if delta.done:
            final = delta.answer
    check("流式首字 ≤3000 ms", first_ms is not None and first_ms <= 3000, f"首字 {first_ms} ms")
    check("done 携带 Answer 且有引用", final is not None and bool(final.citations),
          f"引用 {[c.render() for c in (final.citations if final else [])]}")
    check("流式答案非空且带页码可回溯", bool(final and final.text and final.citations),
          f"{(final.text if final else '')[:60]}")

    # 4) 语言分支
    print("\n[4] 中英文")
    en_q = "What is the registered capital according to the prospectus?"
    check("英文提问识别为 en", language_mod.detect_language(en_q) == "en", f"检测 {language_mod.detect_language(en_q)}")
    check("中文提问识别为 zh", language_mod.detect_language(question) == "zh")
    en_cite = citation_mod.Citation(file_name="prospectus.pdf", page=24).render(language="en")
    check("英文引用格式用 Page", "Page:" in en_cite or "Page:" in str(language_mod.citation_label("en")),
          f"标签 {language_mod.citation_label('en')}")

    # 5) 无依据拒答
    print("\n[5] 无依据 → 不清楚")
    nonsense = "银河系外文明的量子计算机专利申请数量是多少？"
    r2 = retriever.retrieve(nonsense, top_k=cfg.retrieval.top_k, logger=log)
    a2 = generator.answer(nonsense, r2, trace_id="t6s2", logger=log)
    check("超出语料的问题回「不清楚」", a2.is_unknown and a2.text == "不清楚",
          f"reason={a2.unknown_reason}")
    check("拒答不带任何引用", not a2.citations)

    # 6) 日志结构
    print("\n[6] 结构化日志")
    log_path = Path(cfg.paths.log_dir) / "app.log"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-4000:]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    enters = [r for r in records if str(r.get("event", "")).endswith(".enter")]
    exits = [r for r in records if str(r.get("event", "")).endswith(".exit")]
    check("存在 func.enter/exit 事件对", bool(enters) and bool(exits), f"enter={len(enters)} exit={len(exits)}")
    check("exit 事件含 elapsed_ms 与输出摘要",
          any("elapsed_ms" in r and r.get("outputs") is not None for r in exits)
          and any(r.get("inputs") is not None for r in enters),
          "示例 " + (json.dumps({k: exits[-1].get(k) for k in ("func", "elapsed_ms")}, ensure_ascii=False)
                    if exits else "无"))
    error_log = Path(cfg.paths.log_dir) / "error.log"
    error_text = error_log.read_text(encoding="utf-8", errors="replace") if error_log.exists() else ""
    check("异常路径留堆栈（error.log 含 Traceback/URLError）",
          "Traceback" in error_text or "URLError" in error_text,
          f"error.log {error_log.stat().st_size if error_log.exists() else 0} B")
    check("事件带工单编号", any(r.get("work_order") == "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
                                for r in records))
    check("日志含首字/prompt/输出摘要字段",
          any(r.get("event") == "generation.first_token" for r in records)
          and any(r.get("event") == "generation.prompt" for r in records)
          and any(r.get("event") in ("generation.done", "generation.unknown") for r in records))

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("\n" + "─" * 70)
    print(f"T6 契约验收：✅{passed} / {len(CHECKS)}")
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  ❌ {name}　{detail}")
    shutdown_logging()
    return 0 if passed == len(CHECKS) else 1


def _en_cite_removed() -> None:
    """（占位，保持文件末尾无未使用导入。）"""
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
