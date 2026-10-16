# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""诊断脚本：核查留出集与训练集的交集、锚点是否含答案数值。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

D = Path(r"E:\gao6gongdan\工单11\研发\data\dataset")


def jl(name: str) -> list[dict]:
    return [json.loads(x) for x in (D / name).read_text(encoding="utf-8").splitlines() if x.strip()]


hold = jl("留出测试集.jsonl")
train = jl("train.jsonl")
gold = {r["gold_chunk_id"] for r in hold}
pos_used = Counter(r["正向分块"] for r in train)
neg_used = Counter(r.get("负向分块") for r in train if r.get("负向分块"))
inter_pos = gold & set(pos_used)
inter_neg = gold & set(neg_used)
print(f"留出金标准 {len(gold)} 个；与 train 正向分块交集 {len(inter_pos)}；与负向分块交集 {len(inter_neg)}")
for cid in list(inter_pos)[:5]:
    r = [x for x in hold if x["gold_chunk_id"] == cid][0]
    print(f"  [正向重叠] {cid} | 查询={r['query'][:40]} | 文档={r['文档']} 页={r['页码']}")
for cid in list(inter_neg)[:5]:
    r = [x for x in hold if x["gold_chunk_id"] == cid][0]
    print(f"  [负向重叠] {cid} | 查询={r['query'][:40]} | 文档={r['文档']} 页={r['页码']}")

# 锚点是否含答案数值
bad = 0
for r in train:
    v = r.get("数值")
    if v and r.get("指标"):
        num = str(v).rstrip("%").replace("亿元", "").replace("万元", "").replace("元", "")
        if num and num in r["anchor"]:
            bad += 1
            if bad <= 6:
                print(f"  [含数值] 指标={r['指标']} 数值={v} 锚点={r['anchor']}")
print(f"锚点含答案数值的记录数：{bad}")
