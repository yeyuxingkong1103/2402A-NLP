# -*- coding: utf-8 -*-
"""T6 引用可回溯判据标定：在真实数据上量出「正例必过 / 反例必拦」的阈值。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

正例 = 答案落在其真实依据块（含表格块 Markdown）；反例 = 同一答案对上无关块/无关页。
输出每题数值命中率与片段覆盖率，作为 ``citation.SENTENCE_MIN_COVERAGE`` / ``NUMBER_MIN_HIT`` 的标定依据。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/离线/calibrate_citation_support.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import citation as citation_mod, text_utils  # noqa: E402
from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import setup_logging, shutdown_logging  # noqa: E402

# (题号, 答案文本, 正例 chunk_id 前缀（真实依据）, 反例 chunk_id（无关页）)
CASES: list[tuple[str, str, str, str]] = [
    ("34", "答案：电子信息行业的上游涉及信息系统相关的电子元器件制造企业，以及机箱、机柜等金属壳体制造企业。",
     "招股说明书1_p0152_x357", "招股说明书1_p0456_x1055"),
    ("793", "答案：电子信息行业的下游主要包括军队、政府机关、能源等行业企业。",
     "招股说明书1_p0152_x357", "招股说明书1_p0135_x319"),
    ("957", "答案：武汉兴图新科电子股份有限公司在国防军队视频指挥领域已经成为重要供应商。",
     "招股说明书1_p0154_x362", "招股说明书1_p0129"),
    ("531", "答案：法定代表人：程家明", "招股说明书1_p0022_t10", "招股说明书1_p0456_x1055"),
    ("543", "答案：武汉兴图新科电子股份有限公司的注册资本是5,520.00万元。",
     "招股说明书1_p0022_t10", "招股说明书1_p0063_x164"),
    ("795", "答案：某情报、指挥、控制与通信网络一体化工程（即相当于美军的C4ISR系统）荣获国家科技进步一等奖。",
     "招股说明书1_p0094_x235", "招股说明书1_p0139_x329"),
]


def main() -> int:
    """跑标定并打印逐例结果。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    conn = sqlite3.connect(str(cfg.paths.index_dir / "rag.sqlite3"))
    print(f"阈值：覆盖率 ≥ {citation_mod.SENTENCE_MIN_COVERAGE}，数值命中率 ≥ {citation_mod.NUMBER_MIN_HIT}")
    print(f"{'题':<6}{'方向':<8}{'证据块':<28}{'判定':<8}{'说明'}")
    bad = 0
    for qid, answer, pos_id, neg_id in CASES:
        for label, chunk_id, expect in (("正例", pos_id, True), ("反例", neg_id, False)):
            row = conn.execute("SELECT content,file_name,page FROM chunks WHERE chunk_id LIKE ?",
                               (chunk_id + "%",)).fetchone()
            if row is None:
                print(f"{qid:<6}{label:<8}{chunk_id:<28}⚠️ 未找到块")
                bad += 1
                continue
            ok, why = citation_mod.answer_support_check(answer, row[0], logger=None)
            mark = "✅" if ok == expect else "❌ 不符预期"
            if ok != expect:
                bad += 1
            print(f"{qid:<6}{label:<8}{chunk_id[:14] + ':p' + str(row[2]):<28}{str(ok):<8}{mark} {why}")
    conn.close()
    print(f"\n不符预期例数：{bad}（正例必须 True、反例必须 False）")
    shutdown_logging()
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
