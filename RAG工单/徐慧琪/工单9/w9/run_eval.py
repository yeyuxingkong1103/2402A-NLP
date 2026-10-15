# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：优化前后对比评估的执行入口（硬性要求2、6）

流程：
  ① 同一评测集（默认主集 = sample_questions 的 4 题）
  ② 分别跑 **08 基线** 与 **09 优化链路**，记录答案、上下文、逐题耗时
  ③ 用 RAGAS 对两边各评一次（context precision / context recall）
  ④ 汇总成对比报告并落盘

为什么两边必须跑同一批题、同一份语料、同一份向量索引：
  只有这样，指标差异才能**归因于三个层面的优化本身**，而不是数据差异。
  这也是 09 采用「只读引用 08」而非「复制改造」的根本原因。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

from w9 import config9
from w9 import questions as Q


def _run_one_08(q: dict) -> dict:
    """跑 08 基线（算法原样调用）。"""
    from w9 import baseline08
    t0 = time.perf_counter()
    try:
        r = baseline08.ask(q["question_resolved"])
        elapsed = time.perf_counter() - t0
        return {
            "id": q["id"], "question": q["question_resolved"],
            "answer": r.answer,
            "contexts": _contexts_08(r),
            "reference": q["reference"],
            "elapsed_s": round(elapsed, 3),
            "mode": getattr(r, "mode", ""),
            "n_graph_facts": len(getattr(r, "graph_facts", []) or []),
            "n_text_hits": len(getattr(r, "text_hits", []) or []),
            "degraded": bool(getattr(r, "degraded", False)),
            "degraded_reason": getattr(r, "degraded_reason", ""),
        }
    except Exception as exc:                          # noqa: BLE001
        return {"id": q["id"], "question": q["question_resolved"],
                "answer": "", "contexts": [], "reference": q["reference"],
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "error": f"{type(exc).__name__}: {exc}"}


def _contexts_08(r) -> list[str]:
    """08 实际送入 LLM 的上下文（与其 graph_fusion.build_context_text 同口径）。"""
    from src.graph_fusion import FusedContext, build_context_text
    facts = list(getattr(r, "graph_facts", []) or [])
    hits = list(getattr(r, "text_hits", []) or [])
    ctx = build_context_text(FusedContext(graph_facts=facts, text_hits=hits))
    # 按证据块切分，供 RAGAS 逐块判相关性
    out: list[str] = [f"【图谱事实】{f}" for f in facts]
    for h in hits:
        doc = h.get("source") or h.get("source_doc") or ""
        page = h.get("page_idx", h.get("page", -1))
        out.append(f"【文本证据】来源：{doc} 第 {page} 页\n{h.get('text') or ''}")
    return out or ([ctx] if ctx else [])


# 09 的运行档位（full = 完整作答；fast = 压缩输出守住 ≤3 秒）
# 由 main() 在启动时设置，供 _run_one_09 读取。
_MODE9 = "full"


def _run_one_09(q: dict) -> dict:
    """跑 09 优化链路。"""
    from w9.qa_v2 import ask_v2
    t0 = time.perf_counter()
    try:
        r = ask_v2(q["question_resolved"], mode=_MODE9)
        elapsed = time.perf_counter() - t0
        return {
            "id": q["id"], "question": q["question_resolved"],
            "answer": r.answer, "contexts": r.contexts(),
            "reference": q["reference"],
            "elapsed_s": round(elapsed, 3),
            "qtype": r.qtype, "queries": r.queries,
            "n_graph_facts": len(r.graph_facts),
            "n_text_hits": len(r.text_hits),
            "timings": {k: round(v, 3) for k, v in r.timings.items()},
            "info": {k: v for k, v in r.info.items()
                     if isinstance(v, (int, float, str))},
            "degraded": r.degraded, "degraded_reason": r.degraded_reason,
        }
    except Exception as exc:                          # noqa: BLE001
        return {"id": q["id"], "question": q["question_resolved"],
                "answer": "", "contexts": [], "reference": q["reference"],
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "error": f"{type(exc).__name__}: {exc}"}


def run(system: str, qs: list[dict], verbose: bool = True) -> list[dict]:
    """跑一个系统的全部题目。"""
    fn = _run_one_08 if system == "wo08" else _run_one_09
    out: list[dict] = []
    for i, q in enumerate(qs, 1):
        rec = fn(q)
        out.append(rec)
        if verbose:
            flag = "❌" if rec.get("error") else "✅"
            print(f"[{system}] {i}/{len(qs)} {rec['id']} {flag} "
                  f"{rec.get('elapsed_s')}s  块={rec.get('n_text_hits')}", flush=True)
    return out


def save_records(records: list[dict], name: str) -> str:
    os.makedirs(config9.EVAL9_DIR, exist_ok=True)
    path = os.path.join(config9.EVAL9_DIR, f"{name}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="工单09 优化前后对比评估")
    ap.add_argument("--set", default="main", choices=["main", "aux"],
                    help="main=sample_questions 4 题；aux=08 的 14 题")
    ap.add_argument("--systems", default="wo08,wo09")
    ap.add_argument("--ragas", default="core", choices=["core", "full", "none"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default=config9.GEN9_MODE_DEFAULT,
                    choices=list(config9.GEN9_MODES),
                    help="09 生成档位：full=完整作答；fast=压缩输出守住 ≤3 秒")
    args = ap.parse_args()

    global _MODE9
    _MODE9 = args.mode

    qs = Q.main_set() if args.set == "main" else Q.aux_set()
    if args.limit:
        qs = qs[:args.limit]
    print(f"评测集：{args.set}，{len(qs)} 题")

    # 先预热再计时。否则第一题会把「加载 bge-m3 + reranker + BM25 倒排索引 +
    # Ollama 模型入显存」的冷启动时间算进「响应时间」——实测首题 51~53s、
    # 其余题 2~8s，那 50s 全是启动开销，不是真实响应时间。
    print("预热模型（不计入响应时间）…", flush=True)
    try:
        from src import rag
        t_w = time.perf_counter()
        w = rag.warmup()
        print(f"  预热完成 {time.perf_counter() - t_w:.1f}s  {w}", flush=True)
    except Exception as exc:                           # noqa: BLE001
        print(f"  预热失败（继续，但首题耗时会偏高）：{exc}", flush=True)

    # 产物名带上档位：否则「--mode fast」会把 full 档的结果覆盖掉，
    # 而两者指标差异很大（实测 precision 0.81 vs 0.86），覆盖后就分不清了。
    tag = f"{args.set}_{{sys}}" + (f"_{args.mode}" if args.mode != "full" else "")
    summary: dict = {"set": args.set, "n": len(qs), "mode": args.mode,
                     "systems": {}}
    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        print(f"\n=== 运行 {system}（mode={args.mode}）===")
        recs = run(system, qs)
        p = save_records(recs, f"run_{tag.format(sys=system)}")
        ok = [r for r in recs if not r.get("error")]
        times = [r["elapsed_s"] for r in ok]
        entry = {
            "records_file": p, "n_ok": len(ok), "n_error": len(recs) - len(ok),
            "elapsed_mean_s": round(sum(times) / len(times), 3) if times else None,
            "elapsed_max_s": round(max(times), 3) if times else None,
            "elapsed_min_s": round(min(times), 3) if times else None,
            "n_over_3s": sum(1 for t in times if t > config9.RESPONSE_BUDGET_S),
        }
        print(f"[{system}] 均值 {entry['elapsed_mean_s']}s  "
              f"最大 {entry['elapsed_max_s']}s  超 3s {entry['n_over_3s']} 题")

        if args.ragas != "none" and ok:
            print(f"[{system}] 跑 RAGAS（{args.ragas}）…")
            from w9 import ragas_eval
            res = ragas_eval.evaluate(ok, metrics=args.ragas)
            rp = ragas_eval.save(res, f"ragas_{tag.format(sys=system)}")
            entry["ragas"] = res.get("scores", {})
            entry["ragas_file"] = rp
            entry["ragas_nan"] = res.get("nan_counts", {})
        summary["systems"][system] = entry

    sp = os.path.join(config9.EVAL9_DIR, f"summary_{args.set}.json")
    with io.open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已写出：{sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
