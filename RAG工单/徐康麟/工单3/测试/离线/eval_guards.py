# -*- coding: utf-8 -*-
"""t15 守卫标定：三条守卫（主体/时间/谓词）在 6 条冻结负例 + 14 道正题上的区分度与误伤面。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定（先出数据，再决定是否接线到 ``answerability.decide``）：
    * 负例：期望 ``block``（任一守卫命中）；
    * 正题：期望 ``pass``（**误伤必须为 0** —— 红线：宁少拦不可错拦）。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/eval_guards.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import answerability  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import setup_logging, shutdown_logging  # noqa: E402
from app.core.query_understanding import understand  # noqa: E402

NEGATIVES = REPO_ROOT / "测试" / "测试数据" / "unknown_questions.jsonl"
POSITIVES = REPO_ROOT / "测试" / "测试数据" / "eval_retrieval_14.jsonl"


def main() -> int:
    """逐题打印守卫判定与短语，并给出负例拦截率 / 正题误伤数。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    rows_neg, rows_pos = [], []
    print("=" * 108)
    print("【负例】期望 block")
    for line in NEGATIVES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        info = understand(item["question"], None, cfg=cfg, llm=None, logger=None)
        ok, reason, detail = answerability.guard_evidence(
            item["question"], item.get("file_names") or None,
            expects_numeric=bool(info.get("expects_numeric")), cfg=cfg, logger=None)
        rows_neg.append({"id": item["id"], "blocked": not ok, "reason": reason})
        print(f"  {'✅' if not ok else '❌ 未拦下'} {item['id']:<8} reason={reason:<28} "
              f"phrases={detail.get('phrases')} missing={detail.get('missing_phrases')} "
              f"co_occur={detail.get('co_occur')} subject_hits={detail.get('subject_hits')} "
              f"missing_years={detail.get('missing_years')}")

    print("\n" + "=" * 108)
    print("【正题】期望 pass（误伤必须为 0）")
    for line in POSITIVES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        info = understand(item["question"], None, cfg=cfg, llm=None, logger=None)
        ok, reason, detail = answerability.guard_evidence(
            item["question"], None, expects_numeric=bool(info.get("expects_numeric")), cfg=cfg, logger=None)
        rows_pos.append({"id": item["id"], "blocked": not ok, "reason": reason})
        flag = "✅" if ok else "❌ 误伤"
        print(f"  {flag} 题 {str(item['id']):<5} reason={reason:<28} phrases={detail.get('phrases')} "
              f"co_occur={detail.get('co_occur')} numeric_anchor={detail.get('numeric_anchor')} "
              f"missing={detail.get('missing_phrases')}")

    blocked_neg = sum(1 for r in rows_neg if r["blocked"])
    hurt_pos = [r["id"] for r in rows_pos if r["blocked"]]
    print("\n" + "─" * 108)
    print(f"负例拦截：{blocked_neg}/{len(rows_neg)}"
          f"（未拦：{[r['id'] for r in rows_neg if not r['blocked']] or '无'}）")
    print(f"正题误伤：{len(hurt_pos)}/{len(rows_pos)}（{hurt_pos or '无'}）")
    shutdown_logging()
    return 0 if blocked_neg == len(rows_neg) and not hurt_pos else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
