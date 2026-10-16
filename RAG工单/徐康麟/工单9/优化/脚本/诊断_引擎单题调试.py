# -*- coding: utf-8 -*-
"""临时调试：直接跑优化引擎并打印完整 traceback（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "研发"), str(ROOT / "研发" / "评估")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.graphrag_opt.配置 import get_config  # noqa: E402
from app.graphrag_opt.引擎 import OptimizedEngine  # noqa: E402

cfg = get_config()
eng = OptimizedEngine(cfg=cfg, profile="after")
print("warmup:", eng.warmup())
try:
    out = eng.retrieve("平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？")
    print("retrieve ok, hits =", len(out.get("hits") or []))
    print("diagnostics =", {k: v for k, v in (out.get("diagnostics") or {}).items()
                            if k not in ("linked_entities", "fusion_summary")})
except Exception:
    traceback.print_exc()
try:
    ans = eng.answer("平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？")
    print("answer method =", ans.generation_method)
    print(ans.answer[:400])
except Exception:
    traceback.print_exc()
