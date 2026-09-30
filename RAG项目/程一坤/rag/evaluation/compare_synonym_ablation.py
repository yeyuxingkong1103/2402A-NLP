"""同义术语扩写的检索层消融对比（批次 13）。

目的：把「术语扩写开启 / 关闭」对检索指标的影响单独量出来，不掺 LLM 随机性。

用法（每个进程只跑一个 arm，因为装配发生在 import 期）：
    python evaluation/compare_synonym_ablation.py --arm off --tag pass1
    python evaluation/compare_synonym_ablation.py --arm on  --tag pass1

产物：reports/synonym_ablation_<arm>_<tag>.json
    · 每题：golden_rank / hit_at_5 / top10 / absent_violation / 扩展词条 / 扩展条数
    · 汇总：Recall@5 / MRR@10 / 越界数 / 两路召回池里 golden 是否进池

说明：
- 只跑检索链路（不调 LLM），所以指标是确定性的；两遍结果应完全一致，
  这正是"消融两遍"要的结论（增益不是随机波动）。
- 追问题同样先预热短期记忆（写入 context 轮），两 arm 一致，保证可比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_EVAL_SET = PROJECT_ROOT / "data" / "evaluation" / "eval_set_v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"

ARM_ENV = {"on": "true", "off": "false"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同义扩写检索层消融")
    parser.add_argument("--arm", choices=("on", "off"), required=True, help="是否开启术语扩写")
    parser.add_argument("--tag", default="pass1", help="跑次标记（消融第二遍用 pass2）")
    parser.add_argument(
        "--eval-set",
        nargs="+",
        default=[str(DEFAULT_EVAL_SET)],
        help="评测集文件，可传多个（如主集 + 待审候选集，做预览测量）",
    )
    parser.add_argument("--questions", nargs="*", default=None, help="只跑指定题号")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # arm 必须在 import 后端配置之前定好：settings 是 import 期构造的单例
    os.environ["SYNONYM_EXPANSION_ENABLED"] = ARM_ENV[args.arm]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_eval import (  # noqa: E402
        absent_violation,
        first_golden_rank,
        prepare_env,
        with_retry,
    )

    prepare_env()
    # prepare_env 用 setdefault 读 .env，不会覆盖上面显式设置的值
    os.environ["SYNONYM_EXPANSION_ENABLED"] = ARM_ENV[args.arm]

    from app.core.config import settings  # noqa: E402
    from app.retrieval.assembly import build_default_retrieval_service  # noqa: E402

    enabled = settings.synonym_expansion_enabled
    if enabled != (args.arm == "on"):
        print(f"[FATAL] 开关状态与 arm 不一致：arm={args.arm} enabled={enabled}")
        return 2

    items = []
    for eval_set in args.eval_set:
        items.extend(
            json.loads(line)
            for line in Path(eval_set).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.questions:
        wanted = set(args.questions)
        items = [item for item in items if item["id"] in wanted]

    service = build_default_retrieval_service()
    expander = getattr(service.keyword_searcher, "query_expander", None)
    table_version = getattr(expander, "version", None) if expander else None

    def expansion_phrases(question: str) -> list[str]:
        searcher = getattr(service, "keyword_searcher", None)
        provider = getattr(searcher, "expansion_phrases", None)
        return list(provider(question)) if callable(provider) else []

    details: list[dict] = []
    for index, item in enumerate(items, start=1):
        user_id, session_id = "ablation_runner", f"ablation_{item['id']}"
        # 追问题：先预热短期记忆，让查询改写链路与真实评测一致
        if item.get("context") and service.short_term_memory:
            service.short_term_memory.append_message(
                user_id, session_id, {"role": "user", "content": item["context"]}
            )
        try:
            # 与 run_eval 同口径重试：Embedding/重排上游偶发失败不能算进指标
            retrieval = with_retry(
                lambda: service.retrieve(
                    item["question"],
                    rerank_top_n=10,
                    as_of_date=item.get("as_of_date"),
                    jurisdiction="中国大陆",
                    user_id=user_id,
                    session_id=session_id,
                ),
                attempts=5,
            )
        except Exception as error:  # noqa: BLE001
            details.append({"id": item["id"], "errors": [f"{type(error).__name__}: {error}"]})
            print(f"[{args.arm}] {index}/{len(items)} {item['id']} 失败：{type(error).__name__}", flush=True)
            continue

        articles = retrieval.articles
        rank = first_golden_rank(articles, item["golden"]) if item.get("golden") else None
        detail = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expect_refusal": bool(item.get("expect_refusal")),
            "golden": item.get("golden") or [],
            "golden_rank": rank,
            "hit_at_5": bool(rank and rank <= 5),
            "mrr10": round(1.0 / rank, 4) if rank and rank <= 10 else 0.0,
            "retrieved": [
                {
                    "rank": i,
                    "law": a.document_title,
                    "article": a.article_number,
                    "vector": round(a.vector_score, 4) if a.vector_score is not None else None,
                    "keyword": round(a.keyword_score, 4) if a.keyword_score is not None else None,
                    "rerank": round(a.rerank_score, 4) if a.rerank_score is not None else None,
                }
                for i, a in enumerate(articles, start=1)
            ],
            "retrieval_stats": retrieval.stats,
            "expansion_phrases": expansion_phrases(item["question"]),
        }
        if item.get("expect_absent"):
            detail["absent_violation"] = absent_violation(articles, item["expect_absent"])
        details.append(detail)
        print(f"[{args.arm}] {index}/{len(items)} {item['id']} rank={rank}", flush=True)

    summary = summarize(details)
    payload = {
        "arm": args.arm,
        "tag": args.tag,
        "synonym_expansion_enabled": enabled,
        "synonym_table_version": table_version,
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "eval_set": list(args.eval_set),
        "summary": summary,
        "details": details,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"synonym_ablation_{args.arm}_{args.tag}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 汇总 ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"产物：{output_path}")
    return 0


def summarize(details: list[dict]) -> dict:
    graded = [d for d in details if d.get("golden") and d.get("golden_rank") is not None]
    recalled = [d for d in details if d.get("golden")]  # golden 进 top10 与否都要计入 Recall 分母
    hit5 = sum(1 for d in recalled if d.get("hit_at_5"))
    mrr = sum(d.get("mrr10", 0.0) for d in recalled)
    violations = [d["id"] for d in details if d.get("absent_violation")]
    expanded = [d["id"] for d in details if d.get("expansion_phrases")]
    return {
        "total": len(details),
        "graded": len(recalled),
        "recall_at_5": round(hit5 / len(recalled), 4) if recalled else 0.0,
        "mrr_at_10": round(mrr / len(recalled), 4) if recalled else 0.0,
        "absent_violation_count": len(violations),
        "absent_violation_ids": violations,
        "golden_top10_hits": len(graded),
        "questions_expanded": len(expanded),
        "expansion_question_ids": expanded,
    }


if __name__ == "__main__":
    raise SystemExit(main())
