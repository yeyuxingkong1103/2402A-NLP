# -*- coding: utf-8 -*-
"""工单8 临时诊断脚本：追踪答案关键词落在第几名命中块（不写产物，只打印）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "研发"))

from app.graphrag.engine import GraphRAGEngine  # noqa: E402

CASES = [
    ("Q01", "平安银行2019年年度报告的董事长致辞中，提到该行盈利增长的关键因素有哪些？请结合业务结构、盈利趋势与风险状况说明。",
     ["零售进一步巩固收入贡献的支柱地位", "对公、资金同业", "逾期60天以上贷款占比", "1,379.58", "281.95"]),
    ("Q02", "招商银行2019年年度报告中提到的创新商业模式有哪些？",
     ["逾越者联盟", "蛋壳平台", "咖啡零售", "聚焦高频生活场景"]),
    ("Q07", "平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？同比增速分别是多少？年末不良贷款率和拨备覆盖率分别是多少？",
     ["1,379.58", "281.95", "18.2%", "13.6%", "1.65", "183.12"]),
]


def main() -> int:
    engine = GraphRAGEngine()
    for qid, question, keys in CASES:
        result = engine.retrieve(question, top_k=10)
        print(f"=== {qid} | 命中 {len(result['hits_raw'])} 条 ===")
        for rank, hit in enumerate(result["hits_raw"], 1):
            content = str(hit.get("content") or "")
            found = [k for k in keys if k in content]
            print(f"{rank:2d} {str(hit.get('chunk_id')):22s} type={str(hit.get('type')):5s} "
                  f"p{int(hit.get('page') or 0):3d} ch={str(hit.get('retrieval_channel')):12s} "
                  f"score={float(hit.get('score') or 0):7.3f} 关键词命中={len(found)}/{len(keys)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
