# 工单编号：人工智能NLP-RAG-Query 理解优化任务
"""多轮对话评测：按工单演示的 5 轮对话跑一遍，检查改写与回答

用法：
    python eval_dialogue.py --src ../研发 --label 多轮对话

指标：
    改写正确率  指代消解/省略补全是否得到自足的独立问题
    回答准确率  答案是否命中该轮全部事实点
    响应时间    从提问到返回答案的耗时（工单要求 <= 3 秒）
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
from dialogue_truth import DIALOGUE, check_rewrite, _normalize       # noqa: E402

DOCS = {
    "招股说明书1": r"D:\BW\RAG 工单\附件\招股说明书1.pdf",
    "招股说明书2": r"D:\BW\RAG 工单\附件\招股说明书2.pdf",
}
COMPANIES = {"招股说明书1": "武汉兴图新科电子股份有限公司",
             "招股说明书2": "武汉力源信息技术股份有限公司"}


def check_facts(facts, answer):
    """答案是否命中全部事实点（写法里带 & 表示要同时出现）。"""
    norm = _normalize(answer or "")
    hit = sum(1 for group in facts
              if any(all(_normalize(x) in norm for x in pat.split("&"))
                     for pat in group))
    return hit == len(facts), hit, len(facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent / "研发"))
    ap.add_argument("--label", default="多轮对话")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src).resolve()))
    import config, dialogue, rag_engine, vector_store                 # noqa: PLC0415

    print(f"=== 多轮对话评测 [{args.label}] ===")
    multi = rag_engine.MultiDocEngine()
    for stem, pdf in DOCS.items():
        store = vector_store.BGEM3VectorStore()
        cache = Path(config.CACHE_ROOT) / stem
        if not (cache.exists() and store.load(cache)):
            print(f"  [缺少索引] {stem}，请先用研发/app.py 初始化"); return
        print(f"  [缓存] {stem} ({len(store.chunks)} 块)")
        multi.add(COMPANIES[stem], rag_engine.RAGEngine(store, top_k=config.TOP_K))

    conv = dialogue.Conversation(list(COMPANIES.values()))
    print("  [预热] ...", flush=True)
    multi.answer("公司名称是什么？")

    rows, times = [], []
    for item in DIALOGUE:
        turn = conv.resolve(item["question"])
        started = time.time()
        res = multi.answer(turn["resolved"])
        elapsed = time.time() - started
        times.append(elapsed)

        rw_ok, rw_note = check_rewrite(item["expect"], turn["resolved"])
        ans_ok, hit, total = check_facts(item["facts"], res["answer"])
        conv.add_turn(item["question"], turn["resolved"], res["answer"])
        rows.append({"turn": item["turn"], "question": item["question"],
                     "resolved": turn["resolved"], "strategy": turn["strategy"],
                     "rewrite_ok": rw_ok, "answer": res["answer"],
                     "correct": ans_ok, "facts": f"{hit}/{total}", "elapsed": elapsed,
                     "pages": [c["page"] for c, _ in res["contexts"]]})

        print(f"\n  第 {item['turn']} 轮  {'改写OK' if rw_ok else '改写NG'} / "
              f"{'答案OK' if ans_ok else '答案NG'}  事实 {hit}/{total}  {elapsed:.2f}s")
        print(f"    问：{item['question']}")
        print(f"    改：{turn['resolved']}")
        print(f"    策略：{turn['strategy']}  {rw_note}")
        print(f"    答：{res['answer'][:110].replace(chr(10), ' ')}")

    n = len(rows)
    summary = {
        "label": args.label,
        "rewrite_accuracy": sum(r["rewrite_ok"] for r in rows) / n,
        "answer_accuracy": sum(r["correct"] for r in rows) / n,
        "avg_time": statistics.mean(times), "max_time": max(times),
        "over_3s": sum(1 for t in times if t > 3.0), "rows": rows,
    }
    print(f"\n--- [{args.label}] 汇总 ---")
    print(f"  多轮改写正确率 {summary['rewrite_accuracy']:.0%} "
          f"({sum(r['rewrite_ok'] for r in rows)}/{n})")
    print(f"  回答准确率     {summary['answer_accuracy']:.0%} "
          f"({sum(r['correct'] for r in rows)}/{n})")
    print(f"  响应时间       平均 {summary['avg_time']:.2f}s | 最慢 {max(times):.2f}s | "
          f"超 3 秒 {summary['over_3s']}/{n}")

    out = HERE / "result_多轮对话.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细已写入 {out}")


if __name__ == "__main__":
    main()
