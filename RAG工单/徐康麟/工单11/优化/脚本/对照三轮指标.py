# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""对照打印三轮模型的评估指标（微调前 / 轮次1 / 轮次2）。"""
from __future__ import annotations

import io
import json
from pathlib import Path

D = Path(r"E:\gao6gongdan\工单11\研发\评估")


def load(name: str) -> dict:
    return json.load(io.open(D / f"评估结果_{name}.json", encoding="utf-8"))


def show(tag: str, m: dict) -> dict:
    h, r, s = m["留出测试集"], m["真实问题"], m["相似度指标"]
    print(f"{tag:<10} 留出集 R@1={h['Recall@1']:.4f} R@5={h['Recall@5']:.4f} R@10={h['Recall@10']:.4f} "
          f"MRR={h['MRR']:.4f} nDCG@10={h['nDCG@10']:.4f} 平均命中排名={h['平均命中排名']}")
    print(f"{'':<10} 真题   命中@1={r['Recall@1']:.3f} @3={r['Recall@3']:.3f} @5={r['Recall@5']:.3f} "
          f"@10={r['Recall@10']:.3f} MRR={r['MRR']:.3f} 关键词覆盖@10={r['关键词覆盖@10']:.3f}")
    print(f"{'':<10} 相似度 正例={s['正例对平均相似度']:.4f} 负例={s['负例对平均相似度']:.4f} "
          f"间隔={s['相似度间隔']:.4f} 判别准确率={s['正负例判别准确率']:.4f}")
    return {"R1": h["Recall@1"], "R5": h["Recall@5"], "R10": h["Recall@10"], "MRR": h["MRR"],
            "nDCG": h["nDCG@10"], "命中1": r["Recall@1"], "命中10": r["Recall@10"], "真题MRR": r["MRR"],
            "覆盖": r["关键词覆盖@10"], "间隔": s["相似度间隔"], "AUC": s["正负例判别准确率"]}


if __name__ == "__main__":
    a = show("微调前", load("微调前"))
    b = show("轮次1(3e-5/117步)", load("微调后"))
    c = show("轮次2(1e-5/70步)", load("迭代2"))
    print("\n=== 相对微调前的变化 ===")
    for tag, m in (("轮次1", b), ("轮次2", c)):
        print(f"{tag}: " + " ".join(f"{k}={m[k]-a[k]:+.4f}" for k in a))
