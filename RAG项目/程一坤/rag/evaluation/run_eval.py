"""评测运行器（阶段 8.3）：把检索与回答质量变成可重复测量的数字。

一条命令：
    python evaluation/run_eval.py

产物（默认写到项目根 reports/）：
    eval_<时间戳>.json   机器可读（含每条明细）
    eval_<时间戳>.md     人可读（分类型指标 + 最差样本）

四个指标（口径见 data/evaluation/_schema.md）：
    1. Recall@5            期望条号是否落在检索 top5
    2. MRR@10              期望条号首次命中的倒数排名均值（只对非拒答题）
    3. 引用正确率          回答里的 [n] 是否都指向真实存在的法源
    4. 拒答准确率          refusal 题是否按口径被正确拒答

本文件只做"装配 + CLI"（批次 24 拆分后 628 → 约 180 行，守住"单文件 ≤300 行"，
见 docs/目录与命名约定.md §3.4）。其余职责各自成模块，**都是逐字搬移、零逻辑改动**：
    legal_matching.py    法规名/条号匹配（检索侧判定）
    answer_judging.py    拒答口径 + 引用越界审计（回答侧判定）
    item_runner.py       单问题题型的执行（with_retry + run_item）
    aggregate.py         指标汇总 + 最差样本（summarize / worst_samples）
    render_report.py     Markdown 渲染（render_markdown）
    multi_turn_grading.py  多轮专用判定/汇总/渲染（批次 23）
    multi_turn_runner.py   多轮题型的执行（批次 23）

不引入新依赖（纯标准库 + 现有后端代码）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = PROJECT_ROOT / "data" / "evaluation" / "eval_set_v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"

# 本目录也要进 sys.path：同级模块（legal_matching / aggregate / multi_turn_runner …）
# 都按顶层模块名互相导入，这样无论以 `python evaluation/run_eval.py` 还是
# `import evaluation.run_eval` 的方式运行，都导入同一个模块对象（不会出现两份模块状态）。
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from aggregate import summarize, worst_samples  # noqa: E402 - 必须在上面的 sys.path 补全之后
from answer_judging import (  # noqa: E402
    citation_audit,
    refusal_failure_reason,
    refused_by_text,
)
from item_runner import _require_clean_result, run_item, with_retry  # noqa: E402
from legal_matching import (  # noqa: E402
    absent_violation,
    article_matches,
    canonical_title,
    first_golden_rank,
    normalize_article,
    title_matches,
)
from multi_turn_grading import (  # noqa: E402
    EVAL_USER_ID,
    cleanup_eval_session,
    eval_session_id,
    resolve_session_store,
)
from multi_turn_runner import run_multi_turn_item  # noqa: E402
from render_report import render_markdown  # noqa: E402
from run_config import collect_run_config  # noqa: E402

# ↑ 以上名字在本模块是"再导出"（re-export），不要删、也不要当作未使用而清理：
#   拆分前它们是本模块的定义，调用方从这里取；拆分后仍有 6 个脚本这样写：
#   calibrate_refusal.py（DEFAULT_* / prepare_env / with_retry）、
#   compare_synonym_ablation.py（absent_violation / first_golden_rank / prepare_env / with_retry）、
#   compare_window.py（prepare_env / first_golden_rank）、
#   diagnose_rerank.py（prepare_env / canonical_title / title_matches / article_matches）、
#   faithfulness.py（prepare_env / with_retry）、selfcheck_faithfulness.py（prepare_env）。
#   保持可导入 = 调用方零改动（`docs/目录与命名约定.md`：拆分时保持对外函数名不变）。


# ---------------------------------------------------------------------------
# 环境准备：载入 .env、剔除沙箱代理（代理会劫持 Embedding/LLM 请求）
# ---------------------------------------------------------------------------
def prepare_env() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行评测集，产出 JSON + Markdown 报告。")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全跑）")
    parser.add_argument("--only", help="只跑 id 含该子串的题目")
    parser.add_argument("--tag", default="", help="报告文件名后缀标记（如 before/after）")
    parser.add_argument(
        "--no-context-writeback",
        action="store_true",
        help="multi_turn 反向对照：不回写短期记忆（改写应彻底失效，rewrite_rate 掉到 0）",
    )
    args = parser.parse_args(argv)

    prepare_env()
    # build_default_chat_service 在批次 19 已拆到 app.chat.bootstrap（原在 app.chat.service）
    from app.chat.bootstrap import build_default_chat_service
    from app.retrieval.assembly import build_default_retrieval_service

    items = [
        json.loads(line)
        for line in args.eval_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.only:
        items = [item for item in items if args.only in item["id"]]
    if args.limit:
        items = items[: args.limit]

    retrieval_service = build_default_retrieval_service()
    chat_service = build_default_chat_service()

    details: list[dict[str, Any]] = []
    api_failure_counter: dict[str, Any] = {"total": 0, "by_type": {}}
    unresolved_api_failures: list[str] = []
    final_api_failures: list[str] = []
    # 会话清理用的存储（题内 finally 已清一次，这里兜底再清一次）
    session_store = resolve_session_store(retrieval_service)
    for index, item in enumerate(items, start=1):
        start = time.time()
        print(f"[{index}/{len(items)}] {item['id']} …", flush=True)
        detail: dict[str, Any] | None = None
        for item_attempt in range(1, 6):
            try:
                if item.get("type") == "multi_turn":
                    # 多轮题：逐轮跑 + 回写短期记忆 + 改写断言（实现见 multi_turn_runner.py）
                    detail = run_multi_turn_item(
                        item,
                        chat_service=chat_service,
                        retrieval_service=retrieval_service,
                        retrieve_top_n=10,
                        retry=with_retry,
                        citation_audit=citation_audit,
                        refused_by_text=refused_by_text,
                        golden_rank=first_golden_rank,
                        clean_result=lambda result: _require_clean_result(
                            result, api_failure_counter
                        ),
                        # 反向对照开关：关掉回写后改写应彻底失效（见 _schema.md 对照口径）
                        writeback=not args.no_context_writeback,
                    )
                else:
                    detail = run_item(
                        item,
                        chat_service=chat_service,
                        retrieval_service=retrieval_service,
                        retrieve_top_n=10,
                        api_failure_counter=api_failure_counter,
                    )
            except Exception:
                detail = {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item.get("question", ""),
                    "errors": [traceback.format_exc(limit=2)],
                }
            errors = detail.get("errors", []) if detail else []
            is_fallback_failure = any("RerankerFallbackError" in error for error in errors)
            if not is_fallback_failure:
                break
            unresolved_api_failures.append(item["id"])
            if item_attempt < 5:
                print(
                    f"  [Reranker API 失败，整题重试 {item_attempt}/5] {item['id']}",
                    flush=True,
                )
            else:
                final_api_failures.append(item["id"])
        if detail is None:
            detail = {
                "id": item["id"],
                "type": item["type"],
                "question": item.get("question", ""),
                "errors": ["item_execution_failed"],
            }
        # 兜底清理：异常穿透时也不能在 Redis 留下 eval_<id> 会话
        residue = cleanup_eval_session(session_store, EVAL_USER_ID, eval_session_id(item["id"]))
        if residue and residue != "short_term_memory_unavailable":
            detail.setdefault("errors", []).append(f"session_cleanup_failed: {residue}")
        detail["elapsed_seconds"] = round(time.time() - start, 2)
        details.append(detail)

    if final_api_failures:
        print(
            "评测作废：以下题目在 5 次整题重试后仍发生 Reranker API 失败："
            + ", ".join(final_api_failures),
            file=sys.stderr,
        )
        return 2

    payload: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "eval_set": str(args.eval_set),
        # 治理规则第 10 条：当轮变量（LLM 模型 / 提示词 md5 / 召回窗口 / 拒答阈值）随报告落盘，
        # 便于跨轮 diff——否则指标漂移时无法判断"是改动引起的还是变量换了"（见批次 29）
        "run_config": collect_run_config(PROJECT_ROOT, args.eval_set),
        "api_failures": api_failure_counter,
        "api_failure_retry_items": unresolved_api_failures,
        "summary": summarize(details),
        "worst_samples": worst_samples(details, 5),
        "details": details,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = args.output_dir / f"eval_{stamp}{suffix}.json"
    md_path = args.output_dir / f"eval_{stamp}{suffix}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    (args.output_dir / f"latest_eval{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / f"latest_eval{suffix}.md").write_text(render_markdown(payload), encoding="utf-8")

    summary = payload["summary"]
    print("\n==== 评测完成 ====")
    print(f"Recall@5={summary['recall_at_5']:.4f}  MRR@10={summary['mrr_at_10']:.4f}  "
          f"引用正确率={summary['citation_accuracy']:.4f}  拒答准确率={summary['refusal_accuracy']:.4f}")
    multi = summary.get("multi_turn") or {}
    if multi.get("count"):
        rate = multi.get("rewrite_rate")
        turn2 = multi.get("turn2_hit_at_5")
        print(f"multi_turn：改写命中率={rate}（分母 {multi['rewrite_rate_denominator']}）  "
              f"第2轮 top5={turn2}（分母 {multi['turn2_hit_at_5_denominator']}）  "
              f"回写={'开' if multi.get('context_writeback') == [True] else '关/混合'}")
    print(f"报告：{json_path}")
    print(f"报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
