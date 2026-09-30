"""检索效果评测：小规模标注集上的 Recall@K / MRR / 命中率对比。

用途：调整 chunk_size、RRF 权重、top_k 时，用同一套标注集衡量改动是否真的更好。

用法：
    python tools/eval_retrieval.py                    # 对比四种模式
    python tools/eval_retrieval.py --mode hybrid --top-k 6
    python tools/eval_retrieval.py --json > docs/eval_report.json

标注集格式：{角色: [{query, 期望命中的文档标题关键字}, ...]}
判定规则：只要返回结果中存在「文档标题包含关键字」的条目，即认为该 query 命中。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from role_rag.config import get_config  # noqa: E402
from role_rag.logging_conf import setup_logging  # noqa: E402
from role_rag.retrieval.retriever import get_retriever  # noqa: E402

LABELS: dict[str, list[dict[str, str]]] = {
    "financial_planner": [
        {"query": "养老金三支柱是什么？个人养老金每年交多少？", "expect": "养老规划"},
        {"query": "定投为什么能摊平成本？", "expect": "基金类型与定投"},
        {"query": "亏损 30% 需要涨多少才能回本？", "expect": "风险测评"},
        {"query": "应急金应该准备几个月的支出？", "expect": "资产配置基础"},
        {"query": "风险测评 R1 到 R5 有什么区别？", "expect": "风险测评"},
    ],
    "scientist": [
        {"query": "什么是可证伪性？", "expect": "科学方法与可证伪性"},
        {"query": "p 值 0.03 说明了什么？", "expect": "实验设计与统计推断"},
        {"query": "为什么会出现发表偏倚和撤稿？", "expect": "同行评审"},
        {"query": "相关不等于因果的例子", "expect": "逻辑谬误"},
        {"query": "统计功效不足会有什么后果？", "expect": "实验设计与统计推断"},
    ],
    "lawyer": [
        {"query": "试用期最长可以约定多久？", "expect": "劳动争议"},
        {"query": "加班费怎么算？", "expect": "劳动争议"},
        {"query": "一般诉讼时效是几年？", "expect": "证据规则与诉讼时效"},
        {"query": "民间借贷利率上限是多少？", "expect": "民间借贷"},
        {"query": "合同只有签字没有盖章有效吗？", "expect": "合同基础"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role RAG_try 检索评测")
    parser.add_argument("--mode", default="", help="只评测单一模式（默认对比全部）")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def evaluate(retriever, mode: str, top_k: int) -> dict:
    per_role: dict[str, dict] = {}
    for role_id, items in LABELS.items():
        recalls, mrrs = [], []
        for item in items:
            result = retriever.search(item["query"], role_id, top_k=top_k, mode=mode, use_cache=False)
            ranks = [
                index for index, hit in enumerate(result.results, start=1)
                if item["expect"] in hit.doc_title
            ]
            recalls.append(1.0 if ranks else 0.0)
            mrrs.append(1.0 / ranks[0] if ranks else 0.0)
        per_role[role_id] = {
            "queries": len(items),
            "recall": round(mean(recalls), 4),
            "mrr": round(mean(mrrs), 4),
        }
    overall = {
        "queries": sum(item["queries"] for item in per_role.values()),
        "recall": round(mean(item["recall"] for item in per_role.values()), 4),
        "mrr": round(mean(item["mrr"] for item in per_role.values()), 4),
    }
    return {"mode": mode, "top_k": top_k, "overall": overall, "per_role": per_role}


def main() -> int:
    args = parse_args()
    setup_logging()
    config = get_config()
    retriever = get_retriever(config)
    modes = [args.mode] if args.mode else ["dense", "sparse", "bm25", "hybrid"]

    reports = [evaluate(retriever, mode, args.top_k) for mode in modes]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0

    print(f"评测标注集：{sum(len(v) for v in LABELS.values())} 条 query / {len(LABELS)} 个角色，Top-{args.top_k}")
    print("-" * 78)
    print(f"{'模式':<10}{'Recall@K':>12}{'MRR':>10}   分角色 MRR")
    for report in reports:
        detail = " ".join(f"{role}:{item['mrr']:.2f}" for role, item in report["per_role"].items())
        print(f"{report['mode']:<10}{report['overall']['recall']:>12.4f}"
              f"{report['overall']['mrr']:>10.4f}   {detail}")
    print("-" * 78)
    best = max(reports, key=lambda item: item["overall"]["mrr"])
    print(f"MRR 最高的模式：{best['mode']}（{best['overall']['mrr']:.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
