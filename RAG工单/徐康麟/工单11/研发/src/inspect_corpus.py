# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""工单7 语料分块勘察：为数据集生成口径提供真实统计依据（只读）。"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config  # noqa: E402


def main() -> int:
    cfg = Config()
    src = cfg.path("语料分块")
    chunks = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"总分块数 = {len(chunks)}")
    print("按类型：", dict(Counter(c.get("type") for c in chunks)))
    print("按模态：", dict(Counter(c.get("modality") for c in chunks)))
    print("section 非空占比：", round(sum(1 for c in chunks if (c.get("section") or "").strip()) / len(chunks), 4))

    cc = [c.get("char_count") or len(c.get("content") or "") for c in chunks]
    cc.sort()
    print("字符数分位：", {q: cc[int(len(cc) * q) - 1] for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)})

    files = Counter(c.get("file_name") for c in chunks)
    print("\n文档清单（文件名 -> 块数）：")
    for k, v in files.items():
        parts = k.split("__")
        print(f"  {v:5d}  {k}   -> 简称={parts[3] if len(parts) > 3 else '?'} 年份={parts[4] if len(parts) > 4 else '?'} 代码={parts[2] if len(parts) > 2 else '?'}")

    shows = {"table": 0, "text": 0}
    print("\n样例（每种类型 3 条）：")
    for c in chunks:
        t = c.get("type")
        if t in shows and shows[t] < 3:
            shows[t] += 1
            print(f"--- type={t} chunk_id={c.get('chunk_id')} page={c.get('page')} section={c.get('section')!r} structured={str(c.get('structured'))[:200]}")
            print("    content:", (c.get("content") or "")[:300].replace("\n", " | "))

    # 指标-数值模式命中率（决定能否用「指标+数值」构造弱监督查询）
    pat = re.compile(r"([\u4e00-\u9fa5A-Za-z]{2,16})[：:\s]{0,3}([0-9][0-9,\.]{2,})\s*(亿元|万元|百万元|万亿|元|%|个百分点)?")
    hit = sum(1 for c in chunks if pat.search(c.get("content") or ""))
    print(f"\n含「指标+数值」模式的分块数 = {hit} ({hit/len(chunks):.1%})")
    names = Counter()
    for c in chunks:
        for m in pat.finditer(c.get("content") or ""):
            names[m.group(1)] += 1
    print("高频指标名 Top40：", names.most_common(40))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
