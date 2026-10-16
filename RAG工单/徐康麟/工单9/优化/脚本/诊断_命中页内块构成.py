# -*- coding: utf-8 -*-
"""临时诊断：Top 命中页上到底有哪些块、关键词落在哪一块（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\gao6gongdan\工单9\研发")
sys.path.insert(0, r"E:\gao6gongdan\工单9\研发\评估")

import ragas近似 as RA  # noqa: E402
from app.core import retriever as RT  # noqa: E402

bundle = RT.load_bundle(logger=None)
CASES = [
    ("招商银行", 5, ["聚焦高频生活场景", "逾越者联盟", "咖啡零售、出行预订、影票销售", "开放的新生态模式", "蛋壳平台", "联合合作伙伴共同为用户提供优质服务"]),
    ("邮储银行", 18, ["610.36亿元", "16.52%", "81.80亿元", "2.96%"]),
    ("国泰君安", 4, ["428亿元", "150亿元", "22%和35%", "连续14年获得A类AA级监管评级", "11.05%"]),
]
for company, page, kws in CASES:
    rows = [r for r in bundle.chunks.values()
            if company in str(r.get("file_name") or "") and int(r.get("page") or 0) == page]
    print(f"\n===== {company} 第 {page} 页：共 {len(rows)} 块")
    for r in rows:
        text = str(r.get("content") or "")
        hit = [k for k in kws if RA.contains(RA.norm(text), k)]
        print(f"   {r.get('chunk_id'):<34} type={str(r.get('type')):<6} len={len(text):<5} 命中={hit}")
    allk = [k for k in kws if any(RA.contains(RA.norm(str(r.get('content') or '')), k) for r in rows)]
    print(f"   本页可命中关键词：{allk}")
    print(f"   本页未覆盖：{[k for k in kws if k not in allk]}")
