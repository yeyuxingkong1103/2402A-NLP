# -*- coding: utf-8 -*-
"""工单8 临时诊断脚本：对比「开启图谱」与「关闭图谱」的 Top-K 命中（不写产物，只打印）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "研发"))

from app.graphrag.engine import GraphRAGEngine  # noqa: E402

QUESTION = ("平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？"
            "同比增速分别是多少？年末不良贷款率和拨备覆盖率分别是多少？")
MARKERS = ("137,958", "1,379.58", "281.95", "183.12", "1.65")


def dump(engine: GraphRAGEngine, *, use_graph: bool) -> None:
    result = engine.retrieve(QUESTION, top_k=10, use_graph=use_graph)
    print(f"=== use_graph={use_graph} ===")
    for rank, hit in enumerate(result["hits_raw"], 1):
        content = str(hit.get("content") or "")
        marks = "".join(m for m in MARKERS if m in content) or "-"
        print(f"{rank:2d} {str(hit.get('chunk_id')):22s} {str(hit.get('type')):5s} "
              f"p{int(hit.get('page') or 0):3d} {str(hit.get('retrieval_channel')):12s} "
              f"score={float(hit.get('score') or 0):7.3f} 标记={marks}")
    print()


def main() -> int:
    engine = GraphRAGEngine()
    dump(engine, use_graph=True)
    dump(engine, use_graph=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
