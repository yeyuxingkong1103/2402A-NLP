# 工单编号：人工智能NLP-RAG-PDF 文档的表格解析及检索优化
"""多文档准确率评测：对两份招股说明书跑工单 14 个问题

用法：
    python eval_accuracy.py --src ../研发 --label 优化后
    python eval_accuracy.py --src ../工单2/研发 --label 优化前   # 对比工单2 的版本

指标：
    回答准确率  答案命中该题全部事实点的题目占比（工单要求 >= 90%）
    检索命中率  正确来源页出现在召回结果中的题目占比（Hit@K）
    检索精确率  召回块里来自正确答案页的比例（信噪比）
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
from ground_truth import (GROUND_TRUTH, QUESTION_TEXT,          # noqa: E402
                          check_answer, check_retrieval)

DOCS = {
    "招股说明书1": r"D:\BW\RAG 工单\附件\招股说明书1.pdf",
    "招股说明书2": r"D:\BW\RAG 工单\附件\招股说明书2.pdf",
}
# 各文档对应的公司名，用于多文档路由
COMPANIES = {
    "招股说明书1": "武汉兴图新科电子股份有限公司",
    "招股说明书2": "武汉力源信息技术股份有限公司",
}


def _make_multi(rag_engine):
    """拿被测代码库的多文档引擎；旧版本没有就补一个等价的路由垫片。

    这样对比"优化前/优化后"时，唯一的变量就是表格解析本身，路由逻辑一致。
    """
    if hasattr(rag_engine, "MultiDocEngine"):
        return rag_engine.MultiDocEngine()

    class _RouteShim:
        def __init__(self):
            self.entries = []

        def add(self, company, engine):
            if company and engine is not None:
                self.entries.append((company, engine))
            return self

        def route(self, question):
            text = (question or "").replace(" ", "")
            matched = [e for name, e in self.entries if name and name in text]
            return matched or [e for _, e in self.entries]

        def answer(self, question, **kwargs):
            return self.route(question)[0].answer(question, **kwargs)

        def answer_without_rag(self, question):
            return self.route(question)[0].answer_without_rag(question)

    return _RouteShim()


def build_engines(src_dir):
    """为每份 PDF 建/加载索引，返回多文档引擎。"""
    sys.path.insert(0, str(Path(src_dir).resolve()))
    import config, document, rag_engine, vector_store             # noqa: PLC0415

    multi = _make_multi(rag_engine)
    for stem, pdf in DOCS.items():
        store = vector_store.BGEM3VectorStore()
        cache = Path(config.CACHE_ROOT) / stem
        if cache.exists() and store.load(cache):
            print(f"  [缓存] {stem} ({len(store.chunks)} 块)")
        else:
            print(f"  [建索引] {stem} ...", flush=True)
            pages, tables = document.load_pdf(pdf)
            store.build(document.build_chunks(pages, tables))
            store.save(cache)
            print(f"          {len(store.chunks)} 块")
        multi.add(COMPANIES[stem], rag_engine.RAGEngine(store, top_k=config.TOP_K))
    return multi, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--label", default="未命名")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"=== 评测 [{args.label}] 代码库：{args.src} ===")
    engine, config = build_engines(args.src)

    print("  [预热] 首次推理 ...", flush=True)
    engine.answer("公司名称是什么？")

    rows, times = [], []
    for qid in sorted(GROUND_TRUTH):
        question = QUESTION_TEXT[qid]
        started = time.time()
        result = engine.answer(question)
        elapsed = time.time() - started
        times.append(elapsed)

        pages = [c["page"] for c, _ in result["contexts"]]
        ok, hit, total = check_answer(qid, result["answer"])
        src_pages = set(GROUND_TRUTH[qid]["source_pages"])
        rows.append({
            "id": qid, "question": question, "answer": result["answer"],
            "pages": pages, "elapsed": elapsed,
            "correct": ok, "facts_hit": hit, "facts_total": total,
            "retrieval_hit": check_retrieval(qid, pages),
            "context_precision": (sum(1 for p in pages if p in src_pages) / len(pages)
                                  if pages else 0.0),
        })
        print(f"  ID {qid:<4} [{'OK' if ok else 'NG'}] 事实 {hit}/{total} "
              f"检索{'中' if rows[-1]['retrieval_hit'] else '缺'} {elapsed:.2f}s", flush=True)

    n = len(rows)
    summary = {
        "label": args.label, "src": str(Path(args.src).resolve()),
        "accuracy": sum(r["correct"] for r in rows) / n,
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / n,
        "context_precision": statistics.mean(r["context_precision"] for r in rows),
        "avg_context_chunks": statistics.mean(len(r["pages"]) for r in rows),
        "avg_time": statistics.mean(times), "min_time": min(times),
        "max_time": max(times), "over_3s": sum(1 for t in times if t > 3.0),
        "rows": rows,
    }
    print(f"\n--- [{args.label}] 汇总 ---")
    print(f"  回答准确率 {summary['accuracy']:.0%} "
          f"({sum(r['correct'] for r in rows)}/{n})")
    print(f"  检索命中率 {summary['retrieval_hit_rate']:.0%} "
          f"({sum(r['retrieval_hit'] for r in rows)}/{n})")
    print(f"  检索精确率 {summary['context_precision']:.0%} | "
          f"平均送入 {summary['avg_context_chunks']:.1f} 个块")
    print(f"  响应时间   平均 {summary['avg_time']:.2f}s | 最快 {min(times):.2f}s | "
          f"最慢 {max(times):.2f}s | 超 3 秒 {summary['over_3s']}/{n}")

    out = Path(args.out) if args.out else HERE / f"result_{args.label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细已写入 {out}")


if __name__ == "__main__":
    main()
