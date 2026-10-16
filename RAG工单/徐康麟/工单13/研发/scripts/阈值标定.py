# -*- coding: utf-8 -*-
"""工单13 优化阈值标定：``研发/scripts/阈值标定.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

作用：对**可调阈值**做真实扫描，给出「耗时 - 收益 - 是否改变结果」的三方对照，
用来证明配置里的阈值不是拍脑袋定的。当前覆盖：

* ``rerank_budget``（重排候选裁剪上限）：扫描 60 / 40 / 30 / 20 / 10。
  只看**检索**这一段的耗时（生成与阈值无关，不必重复跑 LLM ⇒ 扫描成本低且噪声小），
  同时记录每题 Top-5 ``chunk_id``，用于判定「裁剪是否改变了结果」（改了就必须在报告里写明）。

口径：
    * 每题每档跑 ``--rounds`` 轮，第 1 轮丢弃（索引/分词/嵌入后端的首次访问成本）；
    * 检索耗时取 ``retriever`` 的 ``retrieve()`` 墙钟（含查询嵌入 + 双通道 + 融合重排），
      经 ``app.core.perf.TIMER`` 的 ``stage::2_检索`` 与 ``span::hybrid_search`` 双重核对；
    * 结果一致性按 **Top-k chunk_id 列表**逐题比较（不是只看第一题）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/阈值标定.py --budgets 60,40,30,20,10 \\
        --json=优化/评估结果/重排预算标定.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import perf as perf_mod                                        # noqa: E402
from app.core.config import get_config                                       # noqa: E402
from app.core.errors import IndexMissingError                                # noqa: E402
from app.core.logging_conf import setup_logging                              # noqa: E402
from app.core.qa_engine import QAEngine                                      # noqa: E402
from app.core.retriever import hybrid_search, load_bundle                    # noqa: E402
from app.core.query_understanding import analyze                             # noqa: E402
from app.storage.sqlite_manager import SqliteManager                         # noqa: E402

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"
DEFAULT_QUESTIONS = REPO_ROOT / "测试" / "测试数据" / "测试问题集.jsonl"

EXIT_OK, EXIT_PARAM, EXIT_INDEX = 0, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="阈值标定.py",
                                     description=f"{WORK_ORDER} —— 优化阈值真实扫描标定")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--budgets", default="60,40,30,20,10", help="rerank_budget 扫描档位")
    parser.add_argument("--rounds", type=int, default=3, help="每档轮数（第 1 轮丢弃）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gap-s", type=float, default=0.05)
    parser.add_argument("--json", dest="json_out", default=None)
    return parser


def read_questions(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if str(row.get("question") or "").strip():
                rows.append(row)
    return rows[:limit] if limit and limit > 0 else rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    perf_mod.apply_profile("optimized")
    cfg = get_config()
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    log = setup_logging(cfg, run_id="calib")
    questions = read_questions(Path(args.questions))
    if not questions:
        print(f"[参数错误] 测试问题集为空：{args.questions}", file=sys.stderr)
        return EXIT_PARAM
    try:
        engine = QAEngine(cfg=cfg, logger=log, storage=None)
        engine.warmup(with_generate=False)              # 只要索引与嵌入就绪；本脚本不调 LLM
    except IndexMissingError as exc:
        print(f"[索引缺失] {exc}", file=sys.stderr)
        return EXIT_INDEX
    bundle = load_bundle(cfg, logger=log)
    print(f"[就绪] 索引 {bundle.size} 块；题数 {len(questions)}；扫描档位 {args.budgets}")

    budgets = [int(x) for x in str(args.budgets).replace("，", ",").split(",") if x.strip()]
    rounds = max(2, int(args.rounds))
    report: dict[str, Any] = {
        "工单": WORK_ORDER,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "脚本": str(Path(__file__).resolve()),
        "口径": {
            "度量对象": "retriever.hybrid_search（查询嵌入 + 向量通道 + BM25 通道 + RRF 融合 + 特征重排）",
            "为什么只量检索": "rerank_budget 只影响检索段，与 LLM 生成无关；只量检索可把扫描成本降到 1/10、噪声更低",
            "top_k": int(args.top_k), "轮数": rounds, "丢弃轮数": 1,
            "结果一致性判据": "逐题比较 Top-k chunk_id 列表（全部相同才算『不改变结果』）",
        },
        "档位": [],
    }

    baseline_ids: dict[str, list[str]] = {}
    for budget in budgets:
        os.environ["RAG_PERF__RERANK_BUDGET"] = str(budget)
        per_round: list[list[float]] = []
        ids_by_q: dict[str, list[str]] = {}
        pruned_values: list[int] = []
        for round_no in range(1, rounds + 1):
            costs: list[float] = []
            for index, row in enumerate(questions, 1):
                question = str(row["question"])
                analysis = analyze(question, corpus_issuers=bundle.corpus_issuers, logger=log)
                t0 = time.perf_counter()
                hits = hybrid_search(bundle, question, cfg=cfg, analysis=analysis,
                                     top_k=int(args.top_k), logger=log)
                costs.append((time.perf_counter() - t0) * 1000)
                if round_no > 1:
                    ids_by_q[str(row.get("id"))] = [str(h.get("chunk_id")) for h in hits]
                if args.gap_s > 0 and index < len(questions):
                    time.sleep(float(args.gap_s))
            per_round.append(costs)
        kept = per_round[1:] or per_round
        flat = [v for row in kept for v in row]
        if not baseline_ids:
            baseline_ids = dict(ids_by_q)
        same = sum(1 for key, value in ids_by_q.items() if baseline_ids.get(key) == value)
        entry = {
            "rerank_budget": budget,
            "检索耗时": perf_mod.summarize(flat),
            "按轮次均值ms": [round(statistics.fmean(row), 2) for row in per_round],
            "Top-k一致题数": same,
            "Top-k一致率": round(same / len(ids_by_q), 4) if ids_by_q else 0.0,
            "与60基线完全一致": bool(ids_by_q) and same == len(ids_by_q),
        }
        report["档位"].append(entry)
        print(f"[budget={budget:>3}] 检索均值 {entry['检索耗时']['mean']:>7.2f} ms  "
              f"P95 {entry['检索耗时']['p95']:>7.2f} ms  "
              f"Top-{args.top_k} 与基线一致 {same}/{len(ids_by_q)}")
    # 还原默认，避免污染后续脚本
    os.environ.pop("RAG_PERF__RERANK_BUDGET", None)

    if args.json_out:
        path = Path(args.json_out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 标定结果 → {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
