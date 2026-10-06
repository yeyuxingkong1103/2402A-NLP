# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""准确率评测：对指定代码库跑工单 10 个问题，输出可复现的准确率与响应时间

用法：
    python eval_accuracy.py --src ../工单1/研发 --label 优化前
    python eval_accuracy.py --src ../研发       --label 优化后

指标：
    回答准确率  答案命中该题全部事实点的题目占比（工单要求 >= 90%）
    检索命中率  正确来源页出现在召回结果中的题目占比（Hit@K）
    检索精确率  召回块里来自正确答案页的比例（信噪比，越高说明噪声越少）
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
from ground_truth import GROUND_TRUTH, check_answer, check_retrieval   # noqa: E402

PDF = r"D:\BW\RAG 工单\附件\招股说明书1.pdf"


def load_engine(src_dir):
    """从指定代码库加载引擎（每个代码库用自己的 kb_cache）。"""
    src = str(Path(src_dir).resolve())
    sys.path.insert(0, src)
    import config, document, vector_store, rag_engine            # noqa: PLC0415

    store = vector_store.BGEM3VectorStore()
    cache = Path(config.CACHE_ROOT) / Path(PDF).stem
    if cache.exists() and store.load(cache):
        print(f"  [缓存] {cache}")
    else:
        print("  [建索引] 解析 + 编码 ...")
        pages, tables = document.load_pdf(PDF)
        store.build(document.build_chunks(pages, tables))
        store.save(cache)
    return rag_engine.RAGEngine(store, top_k=config.TOP_K), config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="代码库目录")
    ap.add_argument("--label", default="未命名", help="对比标签，如 优化前/优化后")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    print(f"=== 评测 [{args.label}] 代码库：{args.src} ===")
    engine, config = load_engine(args.src)

    # 预热：首次调用要加载模型到 GPU（实测 4 秒以上），不计入响应时间
    print("  [预热] 首次推理 ...", flush=True)
    engine.answer("公司名称是什么？")

    rows, times = [], []
    for qid in sorted(GROUND_TRUTH):
        question = next(q["question"] for q in config.TEST_QUESTIONS if q["id"] == qid)
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
        # 控制台是 GBK，避免用 ✓/✗ 这类非 GBK 字符
        print(f"  ID {qid:<4} [{'OK' if ok else 'NG'}] 事实 {hit}/{total} "
              f"检索{'中' if rows[-1]['retrieval_hit'] else '缺'} {elapsed:.2f}s", flush=True)

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    ret = sum(r["retrieval_hit"] for r in rows) / n
    prec = statistics.mean(r["context_precision"] for r in rows)
    over = sum(1 for t in times if t > 3.0)
    avg_ctx = statistics.mean(len(r["pages"]) for r in rows)

    summary = {
        "label": args.label, "src": str(Path(args.src).resolve()),
        "accuracy": acc, "retrieval_hit_rate": ret, "context_precision": prec,
        "avg_context_chunks": avg_ctx,
        "avg_time": statistics.mean(times), "min_time": min(times),
        "max_time": max(times), "over_3s": over, "rows": rows,
    }
    print(f"\n--- [{args.label}] 汇总 ---")
    print(f"  回答准确率 {acc:.0%} ({sum(r['correct'] for r in rows)}/{n})")
    print(f"  检索命中率 {ret:.0%} ({sum(r['retrieval_hit'] for r in rows)}/{n})")
    print(f"  检索精确率 {prec:.0%} | 平均送入 {avg_ctx:.1f} 个块")
    print(f"  响应时间   平均 {summary['avg_time']:.2f}s | 最快 {min(times):.2f}s | "
          f"最慢 {max(times):.2f}s | 超 3 秒 {over}/{n}")

    out = Path(args.out) if args.out else HERE / f"result_{args.label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细已写入 {out}")


if __name__ == "__main__":
    main()
