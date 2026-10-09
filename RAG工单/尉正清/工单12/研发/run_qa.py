# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""把 16 个测试问题跑一遍，把答案与检索到的上下文落盘

产出 `测试/results/{kb}.json`，给对比脚本和 RAGAS 评估用。

⚠️ **两条路跑在不同环境里**（原因见 优化/过程问题记录.md 问题 2）：

    --kb rag       → rag_gd    （FlagEmbedding + BGE-M3 + 重排序）
    --kb lightrag  → rag_gd1   （lightrag-hku + neo4j）

所以这个脚本要跑两次，每次用对应的 python：

    D:/Anaconda/envs/rag_gd/python.exe  run_qa.py --kb rag
    D:/Anaconda/envs/rag_gd1/python.exe run_qa.py --kb lightrag

用法：
    python run_qa.py --kb rag [--limit N] [--out 路径]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import testset                                                  # noqa: E402
from query import query_lightrag, query_rag                     # noqa: E402

RESULTS = HERE.parent / "测试" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", choices=["rag", "lightrag"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="复用已跑过的题（大模型偶发失败时不用从头再来）")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else RESULTS / f"{args.kb}.json"
    done = {}
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["id"]: r for r in prev.get("records", [])}
        print(f"[续跑] 复用已有 {len(done)} 题")

    questions = testset.TEST_SET[:args.limit] if args.limit else testset.TEST_SET
    ask = query_rag if args.kb == "rag" else query_lightrag

    records, t0 = [], time.time()
    for i, t in enumerate(questions, 1):
        if t["id"] in done:
            records.append(done[t["id"]])
            continue
        print(f"[{i}/{len(questions)}] id={t['id']} {t['question'][:44]}...", flush=True)
        try:
            r = ask(t["question"])
        except Exception as exc:                       # noqa: BLE001
            # 单题失败不能让整轮白跑：记下来继续，报告里如实标出
            print(f"    ✗ {type(exc).__name__}: {str(exc)[:90]}")
            r = {"kb": args.kb, "answer": "", "contexts": [], "seconds": 0.0,
                 "error": f"{type(exc).__name__}: {exc}"}
        records.append({
            "id": t["id"], "doc": t["doc"], "question": t["question"],
            "gold_answer": t["gold_answer"],
            "gold_pages": t["gold_pages"],
            "answer": r["answer"], "contexts": r["contexts"],
            "metas": r.get("metas", []), "seconds": r["seconds"],
            "kg_chars": r.get("kg_chars", 0),
            "error": r.get("error", ""),
        })
        print(f"    ✓ {r['seconds']}s，{len(r['contexts'])} 段上下文，"
              f"回答 {len(r['answer'])} 字", flush=True)
        # 每题落盘一次：跑一半断了也不丢
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"kb": args.kb, "n": len(records), "records": records},
            ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in records if r["answer"] and not r["error"])
    print(f"\n完成：{ok}/{len(records)} 题有回答，总耗时 {time.time() - t0:.0f}s")
    print(f"已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
