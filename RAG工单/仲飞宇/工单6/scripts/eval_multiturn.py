#!/usr/bin/env python
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
"""
多轮对话评测（工单05 核心产出）。

逐段会话**按顺序**真实跑：改写 → 检索 → 生成，再与 `eval/multiturn.json` 里的
人工 gold 比对。三种口径，全部**离线可复核、不依赖 LLM 打分**：

  rewrite_exact（主口径）  归一化后与 gold_rewritten **字符串等值** —— 完全客观
  rewrite_doc_ok          改写后路由到的文档 == gold_doc_key（宽容口径，兼作安全网）
  rule_resolve_rate       method ∈ {rule-coref, rule-switch} 的比例 —— 量化「规则优先」
  rule_hit                复用 evaluator.rule_hit（与单轮 16 题**同一套判法**）

用法：
    PY=~/rag-data/venv/bin/python
    $PY scripts/eval_multiturn.py                       # 全部脚本，optimized 剖面
    $PY scripts/eval_multiturn.py --scripts WO05-main   # 只跑验收剧本
    $PY scripts/eval_multiturn.py --via-api http://localhost:8080   # 走一次 HTTP 冒烟
    $PY scripts/eval_multiturn.py --judge               # 叠加 LLM judge（默认关）

【为什么默认直连 core 而不是走 HTTP】与 `scripts/eval.py` 一致：评测直接调
Retriever/Generator，不经过 API 层的响应缓存 —— TTFT 数字才是真的。
另配 `--via-api` 专门验 FastAPI 接线（否则"脚本绿了、接口没接上"会漏网）。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings                                    # noqa: E402
from app.core.doc_profiles import get_doc_profile                  # noqa: E402
from app.core.evaluator import LLMJudge, rule_hit                  # noqa: E402
from app.core.generator import Generator, warmup                   # noqa: E402
from app.core.multiturn import commit_turn, prepare_turn, refresh_history  # noqa: E402
from app.core.profiles import get_profile                          # noqa: E402
from app.core.query_understanding import QueryUnderstanding        # noqa: E402
from app.core.retriever import Retriever, build_context            # noqa: E402
from app.core.router import route                                  # noqa: E402
from app.core.session import SessionStore                          # noqa: E402

SCRIPTS_FILE = ROOT / "eval" / "multiturn.json"
RULE_METHODS = {"rule-coref", "rule-switch", "rule-ellipsis"}


# ----------------------------------------------------------------------
def norm(text: str) -> str:
    """归一化：去空白 + 全角标点转半角 + 去尾部问号。用于 rewrite_exact 比较。"""
    s = (text or "").strip()
    table = str.maketrans("，。？！；：（）", ",.?!;:()")
    s = s.translate(table)
    s = "".join(s.split())
    return s.rstrip("?.!。")


@dataclass
class TurnItem:
    script_id: str
    turn: int
    session_id: str
    question: str
    gold_rewritten: str
    rewritten: str
    method: str
    method_expected: str
    rewrite_exact: bool
    rewrite_doc_ok: bool | None
    resolved_entity: str
    carried_intent: str
    focus_entity: str
    session_expired: bool
    expired_expected: bool
    answer: str
    rule_hit: bool | None
    rule_detail: str
    doc_key: str
    routed_doc: str
    route_fallback: bool
    citations: list = field(default_factory=list)
    retrieved_pages: list = field(default_factory=list)
    context_chars: int = 0
    history_chars: int = 0
    history_dropped: bool = False
    ttft_ms: int = 0
    rewrite_ms: int = 0
    total_ms: int = 0
    answer_correctness: float | None = None
    error: str = ""


# ----------------------------------------------------------------------
async def run_script(script: dict, *, profile_name: str, qu: QueryUnderstanding,
                     with_judge: bool, quiet: bool) -> list[TurnItem]:
    """跑一段会话。返回逐轮结果。"""
    p = get_profile(profile_name)
    retriever = Retriever(profile=p)
    generator = Generator()
    judge = LLMJudge(generator) if with_judge else None

    # TTL 脚本用**独立**的 store（注入很小的 ttl），不污染全局单例
    store = (SessionStore(ttl=script["ttl_seconds"])
             if script.get("ttl_seconds") else SessionStore())

    # 会话名 → 真实 session id（支持一段脚本里并行多个会话，用于隔离演示）
    # 【为什么预先发 id 而不是让首轮传 None】`prepare_turn` 对 `session_id=None`
    # 的语义是「单轮、不建会话」（API 就是这么用的：不传 = 单轮）。
    # 评测脚本是**主动要走多轮**的，所以自己发 id。
    sessions: dict[str, str] = {}
    for t in script["turns"]:
        sname = t.get("session", script.get("session", script["id"]))
        sessions.setdefault(sname, f"{script['id']}-{sname}-{int(time.time() * 1000)}")
    items: list[TurnItem] = []
    prev_turn_number = 0

    for t in script["turns"]:
        sname = t.get("session", script.get("session", script["id"]))
        sid = sessions[sname]

        # TTL 脚本：轮次之间等待，跨过 ttl
        if script.get("ttl_seconds") and prev_turn_number:
            time.sleep(float(script["ttl_seconds"]) + 0.3)
        prev_turn_number = t["turn"]

        t0 = time.perf_counter()
        ctx = await prepare_turn(store, qu, t["question"], sid)
        sessions[sname] = ctx.session_id

        rres = await retriever.retrieve(ctx.rewritten)
        context = build_context(rres.hits, query=ctx.rewritten, profile=p)
        refresh_history(store, ctx, budget_chars=len(context))

        doc = get_doc_profile(rres.routed_doc)
        # 【必须用流式】工单的响应口径是 TTFT（首个 token），非流式测不出它 ——
        # 用非流式只能得到"整段耗时"，拿它当 TTFT 会虚高、也不符合验收定义。
        # 这里与 /api/chat/stream 完全同口径：从 t0 到第一个增量。
        ttft: int | None = None
        parts: list[str] = []
        async for piece in generator.generate_stream(
                ctx.rewritten, rres.hits, context=context, history=ctx.history,
                profile=p, doc=doc, lang_question=t["question"]):
            if ttft is None:
                ttft = int((time.perf_counter() - t0) * 1000)
            parts.append(piece)
        answer = "".join(parts)
        commit_turn(store, ctx, answer, doc.key if doc else "")
        total_ms = int((time.perf_counter() - t0) * 1000)

        # ---- 判分 ----
        kw = t.get("rule_keywords") or []
        hit, detail = (rule_hit(answer, kw, t.get("rule_type", "any"))
                       if kw else (None, "（未设判据）"))
        correct = None
        if judge is not None and t.get("judge_answer", True) and t.get("reference_answer"):
            jr = await judge.judge(t["question"], context, answer, t["reference_answer"])
            correct = jr.get("answer_correctness")

        gold_key = t.get("gold_doc_key", "")
        rt = route(ctx.rewritten)
        item = TurnItem(
            script_id=script["id"], turn=t["turn"], session_id=ctx.session_id,
            question=t["question"], gold_rewritten=t.get("gold_rewritten", ""),
            rewritten=ctx.rewritten, method=ctx.method,
            method_expected=t.get("expect_method", "none"),
            rewrite_exact=(norm(ctx.rewritten) == norm(t.get("gold_rewritten", ""))),
            rewrite_doc_ok=(rt.doc_key == gold_key) if gold_key else None,
            resolved_entity=ctx.resolved_entity, carried_intent=ctx.carried_intent,
            focus_entity=store.focus_entity(ctx.session_id),
            session_expired=ctx.expired,
            expired_expected=bool(t.get("expect_session_expired", False)),
            answer=answer, rule_hit=hit, rule_detail=detail,
            doc_key=doc.key if doc else "", routed_doc=rres.routed_doc,
            route_fallback=rres.route_fallback,
            citations=[c.page_label for c in rres.evidence_hits],
            retrieved_pages=[h.page_label for h in rres.hits],
            context_chars=len(context), history_chars=ctx.history_chars,
            history_dropped=ctx.history_dropped,
            rewrite_ms=ctx.rewrite_ms, total_ms=total_ms,
            # TTFT 含改写 + 检索 + prefill + 首 token —— 与 API 同口径
            ttft_ms=ttft if ttft is not None else total_ms,
            answer_correctness=correct,
        )
        items.append(item)
        if not quiet:
            mark = "OK " if (hit is not False) else "MISS"
            star = "" if item.rewrite_exact else "  ← 改写与 gold 不一致"
            print(f"  [{script['id']} t{t['turn']}] {mark} "
                  f"method={ctx.method:11s} {detail}{star}")
            print(f"      → {ctx.rewritten}")
            print(f"      A: {answer.strip()[:120]}")
    return items


# ----------------------------------------------------------------------
def build_report(items: list[TurnItem], *, profile_name: str) -> dict:
    def rate(nums):
        ok = [x for x in nums if x is not None]
        return (sum(1 for x in ok if x) / len(ok)) if ok else 0.0

    judged = [i for i in items if i.rule_hit is not None]
    exact = [i.rewrite_exact for i in items]
    docok = [i.rewrite_doc_ok for i in items if i.rewrite_doc_ok is not None]
    # 规则解析率：只统计「期望由规则解决」的轮次
    rule_expected = [i for i in items if i.method_expected in RULE_METHODS]
    ttft = sorted(i.ttft_ms for i in items)
    rw = sorted(i.rewrite_ms for i in items)

    def pctl(xs, p):
        if not xs:
            return 0
        return xs[round((p / 100) * (len(xs) - 1))]

    summary = {
        "n_scripts": len({i.script_id for i in items}),
        "n_turns": len(items),
        "profile": profile_name,
        "warmup": True,
        # ---- 改写质量（主口径，离线可复核）----
        "rewrite_exact_rate": round(sum(exact) / len(exact), 4) if exact else 0.0,
        "rewrite_exact_hits": sum(exact),
        "rewrite_doc_rate": round(sum(docok) / len(docok), 4) if docok else 0.0,
        "rule_resolve_rate": round(
            sum(1 for i in rule_expected if i.method in RULE_METHODS) / len(rule_expected), 4
        ) if rule_expected else 0.0,
        "rule_resolve_n": len(rule_expected),
        "rewrite_llm_calls": sum(1 for i in items if i.method == "llm"),
        "rewrite_llm_failed": sum(1 for i in items if i.method == "llm-failed"),
        # ---- 答案质量 ----
        "rule_hit": sum(1 for i in judged if i.rule_hit),
        "rule_hit_n": len(judged),
        "rule_hit_rate": round(rate([i.rule_hit for i in judged]), 4),
        # ---- 性能 ----
        "ttft_p50_ms": pctl(ttft, 50),
        "ttft_p95_ms": pctl(ttft, 95),
        "ttft_max_ms": max(ttft) if ttft else 0,
        "ttft_under_3s": sum(1 for x in ttft if x <= 3000),
        "ttft_under_3s_rate": round(sum(1 for x in ttft if x <= 3000) / len(ttft), 4) if ttft else 0.0,
        "rewrite_ms_p50": pctl(rw, 50),
        "rewrite_ms_max": max(rw) if rw else 0,
        # ---- 会话机制 ----
        "session_expired_reported": sum(1 for i in items if i.session_expired),
        "session_expired_expected": sum(1 for i in items if i.expired_expected),
        "isolation_ok": _isolation_ok(items),
        "avg_context_chars": round(sum(i.context_chars for i in items) / len(items)) if items else 0,
        "max_context_chars": max((i.context_chars for i in items), default=0),
        "avg_history_chars": round(sum(i.history_chars for i in items) / len(items)) if items else 0,
        "history_dropped_total": sum(1 for i in items if i.history_dropped),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if any(i.answer_correctness is not None for i in items):
        vals = [i.answer_correctness for i in items if i.answer_correctness is not None]
        summary["avg_answer_correctness"] = round(sum(vals) / len(vals), 4)
    return {"summary": summary, "items": [asdict(i) for i in items]}


def _isolation_ok(items: list[TurnItem]) -> bool | None:
    """隔离用例：A 会话被 B 插入一轮后，焦点仍应是兴图（若串了会变力源）。"""
    iso = [i for i in items if i.script_id == "WO05-isolation"]
    if not iso:
        return None
    last = [i for i in iso if i.turn == 3]
    if not last:
        return None
    return last[0].rule_hit is True and "程家明" in last[0].answer


# ----------------------------------------------------------------------
def save_report(report: dict, *, tag: str | None = None) -> dict:
    out = settings.data_path / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mid = f"-{tag}" if tag else ""
    jp = out / f"mt-{stamp}{mid}.json"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not tag:
        (out / "mt-latest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cp = out / f"mt-{stamp}{mid}.csv"
    cols = [f.name for f in fields(TurnItem)]
    with cp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for it in report["items"]:
            row = dict(it)
            for k in ("citations", "retrieved_pages"):
                row[k] = " / ".join(map(str, row.get(k) or []))
            w.writerow(row)
    return {"json": str(jp), "csv": str(cp)}


def print_summary(s: dict) -> None:
    print("\n" + "=" * 78)
    print(f"工单05 多轮对话评测 · 剖面 {s['profile']} · 共 {s['n_scripts']} 段会话 / {s['n_turns']} 轮")
    print("-" * 78)
    print(f"  改写逐字正确（主口径）  {s['rewrite_exact_hits']}/{s['n_turns']}"
          f" = {s['rewrite_exact_rate'] * 100:.1f}%")
    print(f"  改写后路由正确          {s['rewrite_doc_rate'] * 100:.1f}%")
    print(f"  规则解析率（期望规则解）{s['rule_resolve_rate'] * 100:.1f}%"
          f"（{s['rule_resolve_n']} 轮）")
    print(f"  LLM 兜底调用 / 失败     {s['rewrite_llm_calls']} / {s['rewrite_llm_failed']}")
    print(f"  规则化命中（准确率）    {s['rule_hit']}/{s['rule_hit_n']}"
          f" = {s['rule_hit_rate'] * 100:.1f}%")
    print(f"  TTFT ≤3s 的轮数         {s['ttft_under_3s']}/{s['n_turns']}"
          f"（P50 {s['ttft_p50_ms']} ms / P95 {s['ttft_p95_ms']} ms / 最大 {s['ttft_max_ms']} ms）")
    print(f"  改写耗时                P50 {s['rewrite_ms_p50']} ms / 最大 {s['rewrite_ms_max']} ms")
    print(f"  上下文 / history 字数   均 {s['avg_context_chars']} / 均 {s['avg_history_chars']}"
          f"（丢轮 {s['history_dropped_total']}）")
    print(f"  会话过期上报 / 期望     {s['session_expired_reported']} / {s['session_expired_expected']}")
    print(f"  会话隔离（A 焦点不串）  {s['isolation_ok']}")
    if "avg_answer_correctness" in s:
        print(f"  答案正确性（LLM judge） {s['avg_answer_correctness']}")


# ----------------------------------------------------------------------
async def main_async(args) -> int:
    data = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
    scripts = data["scripts"]
    if args.scripts:
        want = {x.strip() for x in args.scripts.split(",")}
        scripts = [s for s in scripts if s["id"] in want]
        if not scripts:
            print(f"没有匹配的脚本：{args.scripts}")
            return 2

    if args.via_api:
        return await run_via_api(scripts, args.via_api)

    await warmup()
    qu = QueryUnderstanding(llm_enabled=not args.no_llm)
    all_items: list[TurnItem] = []
    for sc in scripts:
        print(f"\n▶ {sc['id']} · {sc.get('title', '')}")
        # script 级 llm 开关（TTL 脚本要关掉兜底，才能断言"规则没解出来"）
        used_qu = QueryUnderstanding(llm_enabled=False) if sc.get("llm") is False else qu
        all_items += await run_script(sc, profile_name=args.profile, qu=used_qu,
                                      with_judge=args.judge, quiet=args.quiet)

    report = build_report(all_items, profile_name=args.profile)
    if not args.quiet:
        print_summary(report["summary"])
    paths = save_report(report, tag=args.tag)
    print(f"\n报告：{paths['json']}\n      {paths['csv']}")
    s = report["summary"]
    return 0 if (s["rewrite_exact_rate"] >= 0.9 and s["rule_hit_rate"] >= 0.9) else 1


async def run_via_api(scripts: list[dict], base: str) -> int:
    """走 HTTP 冒烟：验 FastAPI 接线（SSE 事件里有 rewrite/会话字段）。"""
    import httpx
    bad = 0
    for sc in scripts:
        sid = f"{sc['id']}-api-{int(time.time())}"
        for t in sc["turns"]:
            q = httpx.QueryParams({"question": t["question"], "top_k": 3,
                                   "profile": "optimized",
                                   "session_id": t.get("session", sid)})
            got = None
            with httpx.stream("GET", f"{base}/api/chat/stream?{q}", timeout=120.0) as r:
                ev = None
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        ev = line[7:].strip()
                    elif line.startswith("data: ") and ev == "meta":
                        got = json.loads(line[6:])
            ok = bool(got and got.get("rewrite", {}).get("method") is not None
                      and "session_id" in got)
            print(f"  [{sc['id']} t{t['turn']}] {'OK ' if ok else 'FAIL'} "
                  f"method={got.get('rewrite', {}).get('method') if got else None}")
            bad += 0 if ok else 1
    print(f"\nHTTP 冒烟：{len(scripts)} 段会话，失败 {bad} 轮")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="工单05 多轮对话评测")
    ap.add_argument("--scripts", default=None, help="只跑指定脚本 id（逗号分隔）")
    ap.add_argument("--profile", default="optimized", help="检索剖面")
    ap.add_argument("--judge", action="store_true", help="叠加 LLM judge（默认关）")
    ap.add_argument("--no-llm", action="store_true", help="禁用改写 LLM 兜底（只走规则）")
    ap.add_argument("--via-api", default=None, help="走 HTTP 冒烟，如 http://localhost:8080")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default=None, help="报告文件名后缀")
    args = ap.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
