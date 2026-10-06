# -*- coding: utf-8 -*-
"""t17 方差对照：同一道题（793）在「有生成预热」与「跳过生成预热」两种装配下各跑 N 次。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

目的：t17 给 ``QAEngine.warmup()`` 加了「一次极小生成」后，评测中**题 793** 出现「小模型改写版答案」
（丢「为各类终端用户，覆盖范围广泛」）而被官方判分器判否。本脚本用 A/B 对照回答一个必须回答的问题：
**这是预热引入的行为改变，还是小模型本身固有的采样方差？**

做法：
    * 不修改产品代码——``--arm without`` 时在脚本内把 ``QAEngine._warmup_generation`` 打桩成 no-op；
    * 同一进程内连续 ``engine.ask()`` N 次（每次请求各自采样，等价于独立观测）；
    * 每条答案用**官方判分器**（工单1 ``Evaluator.check_answer``，只读子进程）判一次，落盘 JSONL。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/留痕/t17_variance_793.py --arm with --n 6
    pwsh -NoProfile -File run_py.ps1 测试/留痕/t17_variance_793.py --arm without --n 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "研发"
sys.path.insert(0, str(DEV_DIR))

from app.core import citation as citation_mod  # noqa: E402
from app.core import evaluator_bridge  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import QAEngine, build_engine  # noqa: E402

QUESTION = "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"
GOLDEN = "电子信息行业的下游为各类终端用户，覆盖范围广泛，主要包括军队、政府机关、能源等行业企业。"
OUT_DIR = REPO_ROOT / "优化" / "评估结果" / "过程日志"


def main(argv: list[str] | None = None) -> int:
    """入口：按 arm 装配引擎 → N 次同题作答 → 官方判分 → 落盘。"""
    parser = argparse.ArgumentParser(description="t17 方差对照（题 793）")
    parser.add_argument("--arm", choices=("with", "without"), default="with")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args(argv)

    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("t17_variance_793")
    if args.arm == "without":
        # 打桩：跳过生成预热（产品代码不改，仅本次对照进程内生效）
        QAEngine._warmup_generation = lambda self: {"ok": False, "skipped": True,  # type: ignore[method-assign]
                                                   "reason": "A/B 对照：本次跳过生成预热"}
        log.log_event("t17.variance_patch", arm=args.arm, patch="_warmup_generation → no-op")

    rows: list[dict[str, object]] = []
    with log.enter("t17_variance", {"arm": args.arm, "n": args.n, "qid": 793}) as span:
        engine = build_engine(cfg=cfg, warmup=True, logger=log)
        warmup_gen = (engine.warmup_info or {}).get("generation")
        for index in range(1, int(args.n) + 1):
            started = time.perf_counter()
            answer = engine.ask(QUESTION, top_k=5)
            body = citation_mod.answer_body(answer.text)
            ok, reason = evaluator_bridge.check_answer(body, GOLDEN, logger=log)
            row = {"arm": args.arm, "iter": index, "answer_body": body,
                   "judge_ok": bool(ok), "judge_reason": str(reason)[:120],
                   "first_token_ms": float(answer.first_token_ms or 0.0),
                   "citations": [c.render() for c in answer.citations],
                   "wall_ms": round((time.perf_counter() - started) * 1000, 2)}
            rows.append(row)
            print(f"[{args.arm} #{index}] judge={'✅' if ok else '❌'} 首字={row['first_token_ms']:.0f}ms "
                  f"｜ {body}", flush=True)
        passed = sum(1 for row in rows if row["judge_ok"])
        summary = {"work_order": WORK_ORDER, "arm": args.arm, "n": int(args.n),
                   "judge_pass": passed, "judge_fail": len(rows) - passed,
                   "warmup_generation": warmup_gen, "rows": rows}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"_t17_variance793_{args.arm}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"=== arm={args.arm} 判分通过 {passed}/{len(rows)}；落盘 {out_path.name}", flush=True)
        span.set_output({"arm": args.arm, "pass": passed, "total": len(rows)})
    shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：非零退出，绝不静默
        import traceback

        print(json.dumps({"event": "t17_variance.failed", "level": "ERROR",
                          "error_type": type(exc).__name__, "message": str(exc),
                          "stack": traceback.format_exc()}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
