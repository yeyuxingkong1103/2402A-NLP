# -*- coding: utf-8 -*-
"""临时诊断：英文提问的语料路由与证据（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\gao6gongdan\工单9")
for p in (str(ROOT / "研发"), str(ROOT / "研发" / "评估")):
    sys.path.insert(0, p)

from app.graphrag_opt.配置 import get_config  # noqa: E402
from app.graphrag_opt.引擎 import OptimizedEngine  # noqa: E402

CASES = [
    "In Ping An Bank's 2019 Annual Report, what were the bank's operating income and net profit in 2019?",
    "What were China Pacific Insurance Group's total revenue and net profit in 2021?",
    "In Guotai Junan's 2021 Annual Report, what were the company's consolidated operating income and net profit attributable to shareholders?",
]
cfg = get_config()
eng = OptimizedEngine(cfg=cfg, profile="after")
eng.warmup()
for q in CASES:
    a = eng.answer(q, language="en")
    d = a.diagnostics
    print("=" * 70)
    print("Q:", q[:70])
    print("  corpus_hint=", d.get("corpus_hint"), " companies=", d.get("rewrite_added_companies"),
          " terms=", d.get("rewrite_added_terms"))
    print("  probe_fields=", d.get("field_probe_fields"), " probe_extra=", d.get("field_probe_extra"),
          " note=", d.get("field_probe_note"))
    print("  evidence=", [(c.get("page"), str(c.get("file_name"))[18:24], c.get("retrieval_channel"))
                          for c in a.evidence_chunks])
    print("  ANSWER:", str(a.answer)[:300].replace("\n", " | "))
