# 工单编号：人工智能NLP-RAG-功能测试及评估
"""ccf_competition 检索与评估主脚本（工单7 的核心产出）

对 研发/ccf_testset.py 的 10 个问题跑一遍被测系统（01-06 实现的 RAG），产出三样东西：

  1. 检索结果：每题 Top-K 召回的块（文档/页码/类型/分数），落进 result JSON
  2. 检索评估：把召回结果与标准页码做确定性比对，算 6 个指标
  3. 答案评估：LLM-as-Judge 打 grounded / relevant 两分，并与「纯 LLM 不检索」
     的同题回答做对照 —— 只有对照才能说明检索到底起了多大作用

6 个检索指标的定义见 设计/设计说明.md 第四节（全部可复现，不依赖大模型判断）。

用法：
    python eval_ccf.py                 # 完整跑（含大模型）
    python eval_ccf.py --no-llm        # 只跑检索指标，秒级出结果
    python eval_ccf.py --ids 1,2,10    # 只跑指定题号
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "研发"
# 显式加上两个目录：研发 放被测系统，测试 放本脚本的同级模块（render_detail）。
# 不依赖「运行时 cwd 恰好是脚本目录」这种默认行为。
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEV))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ccf_testset import INDUSTRY_OF, TEST_SET  # noqa: E402
from config import CACHE_ROOT, TOP_K           # noqa: E402
from render_detail import render_markdown      # noqa: E402
from evaluator import _judge                   # noqa: E402
from rag_engine import LLMError, RAGEngine, build_context   # noqa: E402
from vector_store import BGEM3VectorStore      # noqa: E402

KB_NAME = "ccf_competition"
PREVIEW = 120          # 报告里每块正文的截断长度


def _safe(fn, *a, **kw):
    """执行一次大模型调用，失败返回 (None, 错误信息) 而不是抛出去。

    一轮评估要跑 40 多次大模型调用，任何一次网络抖动都不该让整轮白跑；
    失败会被记进 JSON 并在报告里如实列出，不静默跳过。
    """
    try:
        return fn(*a, **kw), None
    except LLMError as exc:
        return None, str(exc)


def _norm(text):
    """归一化：去掉所有空白。

    PDF 抽取出来的数字常带空格（实测中国人寿「612,265 百万元」、中国平安
    「21.7% ；」），不归一化的话字面比对会假性漏检。
    """
    return re.sub(r"\s+", "", text or "")


def fact_hits(facts, text):
    """命中多少个事实点。每个事实点是「可接受写法」列表，命中任一即算通过。"""
    norm = _norm(text)
    return sum(1 for group in facts if any(_norm(x) in norm for x in group))


def retrieval_metrics(item, hits):
    """把召回结果与标准页码/文档比对，算检索指标。"""
    keys = [(c.get("doc"), c.get("page")) for c, _ in hits]
    gold = {(d, p) for d, pages in item["gold_pages"].items() for p in pages}
    gold_docs = set(item["docs"])
    got_docs = {d for d, _ in keys}

    hit_gold = [k for k in keys if k in gold]
    first = next((i for i, k in enumerate(keys) if k in gold), None)
    context = "\n".join(c.get("text", "") for c, _ in hits)

    m = {
        "doc_hit": len(gold_docs & got_docs) / len(gold_docs),
        "doc_missing": sorted(gold_docs - got_docs),
        "page_recall": len(set(hit_gold)) / len(gold) if gold else None,
        "page_precision": len(hit_gold) / len(keys) if keys else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "fact_recall": fact_hits(item["facts"], context) / len(item["facts"]),
        "industry_cover": sorted({INDUSTRY_OF[d] for d in got_docs if d in INDUSTRY_OF}),
    }
    need = item.get("require_industries")
    m["industry_ok"] = (set(need) <= set(m["industry_cover"])) if need else None
    return m


def run_retrieval(engine, item):
    """只做检索，不调大模型。返回 (hits, contexts_text, elapsed, query_understanding)。

    走 engine.retrieve()，也就是 answer() 内部那条路径（含 Query 理解的多路召回）。
    这一点很关键：工单7 要评估的是被测系统真实的检索行为，不能另起一条
    只用原始问句的简化路径去测，否则指标和实际表现对不上。
    """
    started = time.time()
    hit_list, qu = engine.retrieve(item["question"])
    return (hit_list, "\n".join(c.get("text", "") for c, _ in hit_list),
            time.time() - started, qu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="跳过所有大模型调用")
    ap.add_argument("--ids", default="", help="只跑指定题号，逗号分隔")
    ap.add_argument("--tag", default="", help="结果文件名后缀，便于保留多组结果")
    ap.add_argument("--timeout", type=int, default=120,
                    help="单次大模型调用超时。默认 120s，比被测系统默认的 30s 宽 —— "
                         "纯 LLM 对照回答要跑推理，30s 会被截断，"
                         "那会让 RAG 显得比实际更好，对比不公平")
    ap.add_argument("--render-only", metavar="JSON", default="",
                    help="只把已有的结果 JSON 重新渲染成明细 markdown，不重跑大模型")
    args = ap.parse_args()

    if args.render_only:
        src = Path(args.render_only)
        out = json.loads(src.read_text(encoding="utf-8"))
        md_path = HERE / f"测试结果明细{args.tag}.md"
        md_path.write_text(render_markdown(out), encoding="utf-8")
        print(f"[完成] 明细已重新渲染到 {md_path}")
        return 0

    if args.no_llm and not args.tag:
        # 防止 --no-llm 覆盖掉完整评估的结果文件：完整结果要跑十几分钟，
        # 被一个几秒钟的检索-only 跑覆盖掉就白等了（开发时真踩过一次）。
        args.tag = "_retrieval"

    wanted = {int(x) for x in args.ids.split(",") if x.strip()} if args.ids else None
    cases = [t for t in TEST_SET if not wanted or t["id"] in wanted]

    store = BGEM3VectorStore()
    if not store.load(Path(CACHE_ROOT) / KB_NAME):
        print(f"[错误] 知识库 {KB_NAME} 不可用，请先运行 研发/build_kb.py 建库")
        return 1
    docs_in_kb = sorted({c.get("doc") for c in store.chunks})
    print(f"[知识库] {len(store.chunks)} 块 / {len(docs_in_kb)} 份文档", flush=True)

    engine = RAGEngine(store, top_k=TOP_K, timeout=args.timeout)
    rows = []
    for item in cases:
        row = {"id": item["id"], "qtype": item["qtype"], "origin": item["origin"],
               "question": item["question"], "docs": item["docs"],
               "gold_pages": item["gold_pages"]}

        # ---- 第一阶段：检索（有 LLM 时与生成同一次调用，共用同一次 Query 理解）----
        try:
            if args.no_llm:
                hits, _ctx, ret_sec, qu = run_retrieval(engine, item)
                rag_res = None
            else:
                rag_res = engine.answer(item["question"])
                hits = rag_res["contexts"]
                ret_sec, qu = rag_res["retrieve_seconds"], rag_res
        except LLMError as exc:                     # 单题失败不拖垮整轮评估
            rows.append({**row, "error": f"检索阶段失败：{exc}"})
            print(f"[{item['id']:>2}] 检索阶段失败：{exc}", flush=True)
            continue

        # 用与 answer() 完全相同的上下文串喂裁判 —— 见 rag_engine.build_context 的说明
        context = build_context(hits)
        metrics = retrieval_metrics(item, hits)
        row.update({
            "retrieval": metrics, "retrieval_seconds": ret_sec,
            "rewrite": qu["rewrite"], "sub_questions": qu["sub_questions"],
            "entities": qu["entities"],
            "hits": [{"doc": c.get("doc"), "page": c.get("page"),
                      "type": c.get("type"), "score": round(s, 4),
                      "preview": _norm(c.get("text", ""))[:PREVIEW]}
                     for c, s in hits],
        })

        if args.no_llm:
            print(f"[{item['id']:>2}] 检索 {ret_sec:.2f}s | 文档命中 {metrics['doc_hit']:.0%} "
                  f"页召回 {(metrics['page_recall'] or 0):.0%} "
                  f"页精确 {metrics['page_precision']:.0%} "
                  f"事实点 {metrics['fact_recall']:.0%} MRR {metrics['mrr']:.2f} "
                  f"| 召回文档 {sorted({h['doc'] for h in row['hits']})}", flush=True)
            rows.append(row)
            continue

        # ---- 第二阶段：答案评估 ----
        row["rag_answer"] = rag_res["answer"]
        row["rag_seconds"] = rag_res["elapsed"]
        row["fact_recall_in_answer"] = (
            fact_hits(item["facts"], rag_res["answer"]) / len(item["facts"]))
        row["judge"], row["judge_error"] = _safe(_judge, item["question"],
                                                 context, rag_res["answer"])

        plain_res, plain_err = _safe(engine.answer_without_rag, item["question"])
        if plain_res is None:
            row["plain_error"] = plain_err
        else:
            row["plain_answer"] = plain_res["answer"]
            row["plain_seconds"] = plain_res["elapsed"]
            row["plain_fact_recall_in_answer"] = (
                fact_hits(item["facts"], plain_res["answer"]) / len(item["facts"]))
            row["plain_judge"], row["plain_judge_error"] = _safe(
                _judge, item["question"], context, plain_res["answer"])

        j = row.get("judge") or {}
        print(f"[{item['id']:>2}] 检索 {ret_sec:.2f}s 生成 {rag_res['elapsed']:.1f}s | "
              f"文档命中 {metrics['doc_hit']:.0%} 页召回 "
              f"{(metrics['page_recall'] or 0):.0%} 事实点 {metrics['fact_recall']:.0%} | "
              f"裁判 有据={j.get('grounded', '?')} 切题={j.get('relevant', '?')}"
              + (f" | 纯LLM失败：{plain_err}" if plain_err else ""), flush=True)
        rows.append(row)

    summary = summarize(rows, with_llm=not args.no_llm)
    out = {"kb": KB_NAME, "chunks": len(store.chunks), "docs_in_kb": docs_in_kb,
           "top_k": TOP_K, "with_llm": not args.no_llm,
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "summary": summary, "rows": rows}
    path = HERE / f"result_ccf_eval{args.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = HERE / f"测试结果明细{args.tag}.md"
    md_path.write_text(render_markdown(out), encoding="utf-8")
    print(f"\n[完成] 结果已写入 {path}")
    print(f"       明细已写入 {md_path}")
    print_summary(summary)
    return 0


def _mean(rows, fn):
    vals = [fn(r) for r in rows if fn(r) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows, with_llm=True):
    s = {
        "n": len(rows),
        "doc_hit": _mean(rows, lambda r: r["retrieval"]["doc_hit"]),
        "page_recall": _mean(rows, lambda r: r["retrieval"]["page_recall"]),
        "page_precision": _mean(rows, lambda r: r["retrieval"]["page_precision"]),
        "fact_recall": _mean(rows, lambda r: r["retrieval"]["fact_recall"]),
        "mrr": _mean(rows, lambda r: r["retrieval"]["mrr"]),
        "retrieval_seconds": _mean(rows, lambda r: r["retrieval_seconds"]),
        "full_doc_miss": [r["id"] for r in rows if r["retrieval"]["doc_hit"] < 1.0],
        "zero_page_hit": [r["id"] for r in rows if r["retrieval"]["mrr"] == 0.0],
    }
    cross = [r for r in rows if r["qtype"] == "跨文档"]
    if cross:
        s["cross_doc_avg_doc_hit"] = _mean(cross, lambda r: r["retrieval"]["doc_hit"])
        s["industry_fail"] = [r["id"] for r in cross
                              if r["retrieval"]["industry_ok"] is False]
    if with_llm:
        s["grounded"] = _mean(rows, lambda r: (r.get("judge") or {}).get("grounded"))
        s["relevant"] = _mean(rows, lambda r: (r.get("judge") or {}).get("relevant"))
        s["plain_grounded"] = _mean(
            rows, lambda r: (r.get("plain_judge") or {}).get("grounded"))
        s["plain_relevant"] = _mean(
            rows, lambda r: (r.get("plain_judge") or {}).get("relevant"))
        s["answer_seconds"] = _mean(rows, lambda r: r.get("rag_seconds"))
        s["plain_seconds"] = _mean(rows, lambda r: r.get("plain_seconds"))
        # 答案事实点召回对 RAG 与纯 LLM 是同一把尺子（只看答案里有没有出现
        # 该题要求的事实），比 grounded 更适合做横向对照 —— grounded 的定义是
        # 「能否在检索到的上下文中找到依据」，不检索的一方天然吃亏。
        s["answer_fact_recall"] = _mean(rows, lambda r: r.get("fact_recall_in_answer"))
        s["plain_answer_fact_recall"] = _mean(
            rows, lambda r: r.get("plain_fact_recall_in_answer"))
        s["failed"] = [r["id"] for r in rows
                       if r.get("error") or r.get("plain_error") or r.get("judge_error")]
    return s


def print_summary(s):
    print("\n" + "=" * 66)
    print("检索评估汇总")
    print("=" * 66)
    print(f"题数 {s['n']}")
    print(f"文档命中率   {s['doc_hit']:.1%}")
    print(f"页召回率     {s['page_recall']:.1%}")
    print(f"页精确率     {s['page_precision']:.1%}")
    print(f"事实点召回   {s['fact_recall']:.1%}")
    print(f"MRR          {s['mrr']:.3f}")
    print(f"检索耗时     {s['retrieval_seconds']:.2f}s")
    if "grounded" in s:
        print("-" * 66)
        print(f"答案有据率   RAG {s['grounded']:.1%}  vs  纯LLM {s['plain_grounded']:.1%}")
        print(f"答案切题率   RAG {s['relevant']:.1%}  vs  纯LLM {s['plain_relevant']:.1%}")
        print(f"答案事实点   RAG {s['answer_fact_recall']:.1%}  vs  "
              f"纯LLM {s['plain_answer_fact_recall']:.1%}")
        print(f"生成耗时     RAG {s['answer_seconds']:.1f}s  纯LLM {s['plain_seconds']:.1f}s")
    print("-" * 66)
    print(f"未完全命中目标文档的题：{s['full_doc_miss'] or '无'}")
    print(f"完全没召回标准页的题：{s['zero_page_hit'] or '无'}")
    if "industry_fail" in s:
        print(f"跨文档题行业覆盖不全：{s['industry_fail'] or '无'}")
    if s.get("failed"):
        print(f"存在调用失败的题：{s['failed']}（详见 JSON 的 error 字段）")


if __name__ == "__main__":
    raise SystemExit(main())
