"""批次 10 任务 2：召回窗口前后对比脚本（可复用调参工具）。

对同一批问题，分别用旧窗口（20/20/20）和新窗口（40/40/40）跑检索，
对比 golden 排名与 top10 命中，验证窗口放大对 Recall 的收益。

用法：
    python evaluation/compare_window.py [--questions asof-031 confuse-046]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from run_eval import prepare_env, first_golden_rank  # noqa: E402

prepare_env()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", nargs="*", default=["asof-031", "confuse-046", "confuse-049"])
    parser.add_argument("--old", type=int, default=20, help="旧窗口大小")
    parser.add_argument("--new", type=int, default=40, help="新窗口大小")
    args = parser.parse_args()

    eval_set = PROJECT_ROOT / "data" / "evaluation" / "eval_set_v1.jsonl"
    items = {
        json.loads(line)["id"]: json.loads(line)
        for line in eval_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    from app.retrieval.assembly import build_default_retrieval_service

    service = build_default_retrieval_service()

    for qid in args.questions:
        item = items.get(qid)
        if not item:
            print(f"[跳过] {qid} 不在评测集中")
            continue
        print("=" * 90)
        print(f"{qid}：{item['question']}  as_of={item.get('as_of_date')}")
        print(f"golden：{[(g['law'], g['article']) for g in item.get('golden', [])]}")
        for label, window in (("旧窗口", args.old), ("新窗口", args.new)):
            result = service.retrieve(
                item["question"],
                vector_recall_limit=window,
                keyword_recall_limit=window,
                rerank_candidate_limit=window,
                rerank_top_n=10,
                as_of_date=item.get("as_of_date"),
                jurisdiction="中国大陆",
                user_id="window_compare",
                session_id=f"window_compare_{qid}_{window}",
            )
            rank = first_golden_rank(result.articles, item.get("golden") or [])
            print(f"  [{label} {window}/{window}/{window}] golden排名={rank}  stats={result.stats}")
            for i, art in enumerate(result.articles[:10], start=1):
                marker = " <-- golden" if rank == i else ""
                print(f"    {i:>2}. {art.document_title[:30]:<32} {art.article_number or '-':<8} rerank={art.rerank_score or 0:.4f}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
