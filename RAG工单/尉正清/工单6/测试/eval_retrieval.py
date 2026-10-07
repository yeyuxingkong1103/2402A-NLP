# 工单编号：人工智能NLP-RAG-混合检索任务
"""检索策略评测：对比向量检索 / 全文检索 / 混合检索

用法：
    python eval_retrieval.py                 # 三种策略全跑，12 题（工单4 的 16 题）
    python eval_retrieval.py --mode hybrid --alpha 0.7

指标（工单要求）：
    准确率  答案命中该题全部事实点的题目占比      目标 >= 90%
    召回率  正确答案的事实点出现在召回上下文中的比例  目标 >= 95%
    响应时间 从提问到返回答案的耗时               目标 <= 3 秒
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ground_truth import GROUND_TRUTH, QUESTION_TEXT, _normalize       # noqa: E402

DOCS = {"招股说明书1": r"D:\BW\RAG 工单\附件\招股说明书1.pdf",
        "招股说明书2": r"D:\BW\RAG 工单\附件\招股说明书2.pdf"}
COMPANIES = {"招股说明书1": "武汉兴图新科电子股份有限公司",
             "招股说明书2": "武汉力源信息技术股份有限公司"}


def fact_hits(facts, text):
    """事实点在文本里出现了几个（写法带 & 表示要同时出现）。"""
    norm = _normalize(text or "")
    return sum(1 for group in facts
               if any(all(_normalize(x) in norm for x in pat.split("&"))
                      for pat in group))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent / "研发"))
    ap.add_argument("--mode", default=None, help="只测某一种：vector/fulltext/hybrid")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--fusion", default="weighted")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src).resolve()))
    import config, rag_engine, vector_store                          # noqa: PLC0415

    print("=== 加载知识库 ===")
    engines = {}
    for stem in DOCS:
        store = vector_store.BGEM3VectorStore()
        if not (Path(config.CACHE_ROOT) / stem).exists() or \
                not store.load(Path(config.CACHE_ROOT) / stem):
            print(f"  [缺少索引] {stem}"); return
        engines[stem] = store
        print(f"  [缓存] {stem} ({len(store.chunks)} 块)")

    # 按公司名路由（沿用多文档引擎的分流规则）
    def route(q):
        for stem, comp in COMPANIES.items():
            if comp in q or comp[:4] in q:
                return stem
        return "招股说明书1"

    modes = [args.mode] if args.mode else ["vector", "fulltext", "hybrid"]
    summary_all = {}
    for mode in modes:
        rags = {stem: rag_engine.RAGEngine(st, top_k=config.TOP_K, mode=mode,
                                           alpha=args.alpha) for stem, st in engines.items()}
        print(f"\n=== 策略：{mode}" + (f" (alpha={args.alpha}, {args.fusion})"
                                       if mode == "hybrid" else "") + " ===")
        rags["招股说明书1"].answer("预热", use_query_understanding=False)   # 预热 GPU

        rows, times = [], []
        for qid in sorted(GROUND_TRUTH):
            q = QUESTION_TEXT[qid]
            gt = GROUND_TRUTH[qid]
            started = time.time()
            res = rags[route(q)].answer(q)
            elapsed = time.time() - started
            times.append(elapsed)

            ctx = "\n".join(c["text"] for c, _ in res["contexts"])
            ans_ok, hit, total = _facts_ok(gt["facts"], res["answer"])
            recalled = fact_hits(gt["facts"], ctx)
            rows.append({"id": qid, "correct": ans_ok, "facts": f"{hit}/{total}",
                         "recall": recalled / len(gt["facts"]),
                         "elapsed": elapsed,
                         "pages": [c["page"] for c, _ in res["contexts"]]})
            print(f"  ID {qid:<4} 答案{'OK' if ans_ok else 'NG'} "
                  f"事实{hit}/{total} 召回{recalled}/{len(gt['facts'])} {elapsed:.2f}s",
                  flush=True)

        n = len(rows)
        summary_all[mode] = {
            "accuracy": sum(r["correct"] for r in rows) / n,
            "recall": statistics.mean(r["recall"] for r in rows),
            "avg_time": statistics.mean(times), "max_time": max(times),
            "over_3s": sum(1 for t in times if t > 3.0), "rows": rows,
        }
        s = summary_all[mode]
        print(f"  --- 准确率 {s['accuracy']:.0%} | 召回率 {s['recall']:.0%} | "
              f"平均 {s['avg_time']:.2f}s | 超3秒 {s['over_3s']}/{n}")

    print("\n=== 三种策略对比 ===")
    print(f"{'策略':<10}{'准确率':<10}{'召回率':<10}{'平均耗时':<12}{'超3秒'}")
    for mode, s in summary_all.items():
        print(f"{mode:<10}{s['accuracy']:<10.0%}{s['recall']:<10.0%}"
              f"{s['avg_time']:<12.2f}{s['over_3s']}/{len(s['rows'])}")

    out = HERE / f"result_检索策略{('_' + args.mode) if args.mode else ''}.json"
    out.write_text(json.dumps(summary_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out}")


def _facts_ok(facts, answer):
    hit = fact_hits(facts, answer)
    return hit == len(facts), hit, len(facts)


if __name__ == "__main__":
    main()
