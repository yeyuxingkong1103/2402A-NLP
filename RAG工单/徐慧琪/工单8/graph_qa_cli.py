# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
CLI：批量跑 eval_question 的全部问题，留痕逐题结果。

用法：
  python graph_qa_cli.py --all                    # 全部 14 题
  python graph_qa_cli.py --all --limit 3          # 只跑前 3 题
  python graph_qa_cli.py --all --out data/eval/graph_qa_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

from src import bootstrap  # noqa: F401
from src import config  # noqa: F401

GROUND_TRUTH = os.path.join("data", "eval", "ground_truth.json")
DEFAULT_OUT = os.path.join("data", "eval", "graph_qa_results.json")


def load_ground_truth(path: str | None = None) -> list[dict]:
    path = path or GROUND_TRUTH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少问题集 {path}（Task 2 生成）")
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def run_question(item: dict, kg=None, ask_fn=None) -> dict:
    """执行单题并展平为可序列化记录。"""
    from src import graph_qa
    ask_fn = ask_fn or graph_qa.ask

    question = item["question"]
    base = {"id": item.get("id"), "question": question,
            "qtype": item.get("qtype"), "origin": item.get("origin"),
            "reference": item.get("reference"),
            "key_facts": item.get("key_facts", []), "error": None}

    try:
        res = ask_fn(question, kg=kg) if kg is not None else ask_fn(question)
        d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        return {**base, **d}
    except Exception as exc:                       # noqa: BLE001
        return {**base, "answer": "", "mode": "error",
                "timings": {}, "graph_facts": [], "text_hits": [],
                "subgraph": {}, "degraded": True,
                "degraded_reason": str(exc),
                "error": f"{type(exc).__name__}: {exc}"}


def run_all(items: list[dict] | None = None, out_path: str | None = None,
            limit: int | None = None, ask_fn=None, kg=None,
            verbose: bool = True) -> list[dict]:
    """批量执行；单题失败不中断整批。"""
    from src import graph_qa

    items = items if items is not None else load_ground_truth()
    if limit is not None:
        items = items[:limit]
    out_path = out_path or DEFAULT_OUT
    # 注：注入 ask_fn（测试桩）时不需要加载真实图谱，避免无谓开销
    if kg is None and ask_fn is None:
        kg = graph_qa.get_kg()

    results: list[dict] = []
    t0 = time.time()
    for i, item in enumerate(items, start=1):
        t1 = time.time()
        rec = run_question(item, kg=kg, ask_fn=ask_fn)
        rec["wall_seconds"] = round(time.time() - t1, 2)
        results.append(rec)
        if verbose:
            mark = "⚠️" if rec.get("error") or rec.get("degraded") else "✅"
            total = (rec.get("timings") or {}).get("total", 0)
            print(f"{mark} Q{rec['id']:02d} [{rec.get('mode')}] "
                  f"{total:.2f}s  {rec['question'][:38]}…", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if verbose:
        ok = sum(1 for r in results if not r.get("error"))
        avg = sum((r.get("timings") or {}).get("total", 0)
                  for r in results) / max(len(results), 1)
        print(f"\n完成 {len(results)} 题（成功 {ok}），"
              f"平均耗时 {avg:.2f}s，总用时 {(time.time() - t0) / 60:.1f}min")
        print(f"结果已写入 {out_path}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="工单08 批量评测")
    ap.add_argument("--all", action="store_true", help="跑全部问题")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.all:
        ap.print_help()
        return 2

    from src import graph_qa
    print("预热中…")
    print(graph_qa.warmup())
    run_all(limit=args.limit, out_path=args.out)
    return 0


if __name__ == "__main__":
    # 入口一律跑在大栈线程里：主线程只有约 1MB 栈，transformers 的深链导入
    # （sklearn→pandas）会在此栈上触发 access violation（本机实测段错误，
    # 崩点随运行漂移）。与 build_index.py / retriever.py 等入口保持同一约定。
    bootstrap.run_with_large_stack(main)
