# -*- coding: utf-8 -*-
"""临时诊断：缺失关键词在语料里到底落在哪一页（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\gao6gongdan\工单9\研发")
sys.path.insert(0, r"E:\gao6gongdan\工单9\研发\评估")

import 判分口径 as J
import ragas近似 as RA  # noqa: E402
from app.core import retriever as RT  # noqa: E402

bundle = RT.load_bundle(logger=None)
print("索引块数 =", bundle.size)

refs = {}
for line in Path(r"E:\gao6gongdan\工单9\研发\评估\参考答案集.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        refs[d["id"]] = d

for qid in ("Q03", "Q04", "Q09", "Q13", "Q02", "Q15"):
    item = refs[qid]
    files = J.splitters()["split_source_files"](item.get("source_file"))
    want = [f for f in files]
    print(f"\n===== {qid}  期望页区间={item.get('expected_pages_hint')}  文件数={len(files)}")
    for kw in item.get("answer_keywords") or []:
        hits = []
        for cid, row in bundle.chunks.items():
            fn = str(row.get("file_name") or "")
            if want and fn not in want:
                continue
            if RA.contains(RA.norm(str(row.get("content") or "")), kw):
                hits.append((fn.split("__")[3] if fn.count("__") >= 3 else fn[:14], row.get("page")))
        # 去重 + 限条数
        uniq = sorted(set(hits))[:6]
        print(f"   {kw!r:40} 命中 {len(set(hits))} 块 -> {uniq}")
