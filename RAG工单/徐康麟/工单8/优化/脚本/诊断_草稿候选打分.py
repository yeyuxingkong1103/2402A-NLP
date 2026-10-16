# -*- coding: utf-8 -*-
"""工单8 临时诊断脚本：打印草稿候选句的打分明细（定位「正确答案句为何落选」）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "研发"))

from app.graphrag.engine import GraphRAGEngine, select_evidence  # noqa: E402
from app.graphrag.grounded import _ATOM_RE, _SENT_SPLIT, _question_terms, _term_weights  # noqa: E402

QUESTION = ("平安银行2019年年度报告的董事长致辞中，提到该行盈利增长的关键因素有哪些？"
            "请结合业务结构、盈利趋势与风险状况说明。")
TARGET = "零售进一步巩固收入贡献的支柱地位"


def main() -> int:
    engine = GraphRAGEngine()
    result = engine.retrieve(QUESTION, top_k=10)
    evidence = select_evidence(result["hits_raw"], top_k=3, max_per_page=2, prefer_text=True)
    terms = _question_terms(QUESTION)
    weights = _term_weights(terms)
    print("实词与 IDF 权重：", list(zip(terms, [round(w, 2) for w in weights])))
    print("证据块：", [(h["chunk_id"], h["type"]) for h in evidence])
    for rank, chunk in enumerate(evidence):
        content = re.sub(r"[ \t\u3000]+", " ", str(chunk.get("content") or ""))
        print(f"--- #{rank} {chunk['chunk_id']} {chunk['type']} 含目标句={TARGET in content} 长度={len(content)}")
        rows = []
        for sentence in _SENT_SPLIT.split(content):
            sentence = sentence.strip(" |")
            if len(sentence) < 6 or len(sentence) > 90:
                continue
            compact = re.sub(r"\s+", "", sentence)
            matched = [w for t, w in zip(terms, weights) if t in compact or t in sentence]
            mass = sum(matched)
            if len(matched) < 2 or mass < 0.75:
                continue
            rows.append((mass * 2 + (1.0 if _ATOM_RE.search(sentence) else 0.0)
                         + (0.6 if chunk["type"] == "table" else 0.0) - rank * 0.25,
                         len(matched), mass, sentence[:70]))
        rows.sort(reverse=True)
        for score, hits, mass, sentence in rows[:6]:
            mark = "★" if TARGET[:6] in sentence else " "
            print(f"   {mark} score={score:6.2f} hits={hits} mass={mass:5.2f} | {sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
