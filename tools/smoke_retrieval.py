"""检索冒烟测试：对每个角色跑若干问题，并列比较 dense / sparse / bm25 / hybrid。

用法：
    python tools/smoke_retrieval.py
    python tools/smoke_retrieval.py --query "试用期最长多久" --role lawyer --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from role_rag.config import get_config  # noqa: E402
from role_rag.logging_conf import setup_logging  # noqa: E402
from role_rag.retrieval.retriever import get_retriever  # noqa: E402

DEFAULT_QUERIES = {
    "financial_planner": [
        "养老金三支柱分别是什么？个人养老金每年能交多少？",
        "定投为什么能摊平成本？什么时候该止盈？",
        "亏损 30% 需要涨多少才能回本？",
    ],
    "scientist": [
        "p 值 0.03 到底说明了什么？",
        "随机对照试验为什么要随机化和盲法？",
        "比较 20 个假设时为什么容易出现假阳性？",
    ],
    "lawyer": [
        "试用期最长可以约定多久？试用期工资有下限吗？",
        "民间借贷利率超过 LPR 四倍会怎样？",
        "普通诉讼时效是几年？从什么时候开始算？",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role RAG_try 检索冒烟测试")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--role", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="并列比较四种检索模式")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser.parse_args()


def show(result, top_k: int) -> None:
    print(f"  mode={result.mode:<7} fusion={result.fusion:<7} cached={result.cached} "
          f"candidates={result.candidates} timings={ {k: round(v, 3) for k, v in result.timings.items()} }")
    for position, item in enumerate(result.results[:top_k], start=1):
        routes = " ".join(
            f"{name}#{int(values['rank'])}({values['raw']:.3f})"
            for name, values in item.routes.items()
        )
        print(f"    {position}. {item.score:.5f} | 《{item.doc_title}》›{item.section} | {routes}")
        preview = item.text.replace("\n", " ")[:88]
        print(f"       {preview}…")


def main() -> int:
    args = parse_args()
    setup_logging()
    config = get_config()
    retriever = get_retriever(config)

    if args.query:
        if not args.role:
            print("--query 必须配合 --role 使用", file=sys.stderr)
            return 2
        print(f"\n=== 查询：{args.query}（角色 {args.role}）===")
        if args.compare:
            report = retriever.compare(args.query, args.role, top_k=args.top_k)
            for mode, payload in report["modes"].items():
                print(f"\n--- {mode} | candidates={payload['candidates']} ---")
                for position, item in enumerate(payload["results"], start=1):
                    routes = item.get("routes", {})
                    detail = " ".join(f"{k}#{int(v['rank'])}" for k, v in routes.items()) if routes else ""
                    print(f"  {position}. {item['score']:.5f} | 《{item['doc_title']}》›{item['section']} | {detail}")
                    print(f"     {item['preview'][:88]}…")
            print("\n查询稀疏词权重 Top12：", report["sparse_terms"])
        else:
            for mode in ("dense", "sparse", "bm25", "hybrid"):
                print(f"\n--- {mode} ---")
                show(retriever.search(args.query, args.role, top_k=args.top_k, mode=mode,
                                      use_cache=False, debug=True), args.top_k)
        return 0

    print("=" * 92)
    print("三路混合检索冒烟测试")
    print("=" * 92)
    for role_id, queries in DEFAULT_QUERIES.items():
        for query in queries:
            print(f"\n[{role_id}] {query}")
            show(retriever.search(query, role_id, top_k=args.top_k, mode="hybrid", use_cache=False), args.top_k)

    print("\n" + "=" * 92)
    print("缓存验证（同一 query 再查一次）")
    first = retriever.search("试用期最长可以约定多久？", "lawyer", top_k=3, use_cache=True)
    second = retriever.search("试用期最长可以约定多久？", "lawyer", top_k=3, use_cache=True)
    print(f"第一次 cached={first.cached}，第二次 cached={second.cached}"
          f"（Redis 缓存键前缀 cache:retr:）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
