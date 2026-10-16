# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""优化实验脚本：检索池规模敏感性分析（复用已缓存的微调前后向量，秒级完成）。

目的：验证「微调带来的提升」不是某一种检索池规模的偶然结果。
输入：研发/评估/向量_微调前.npy、向量_微调后.npy、研发/data/dataset/留出测试集.jsonl
输出：优化/评估结果/检索池敏感性.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发" / "src"))

import numpy as np  # noqa: E402

from evaluator import ranks_from_scores, retrieval_metrics  # noqa: E402

评估目录 = ROOT / "研发" / "评估"
数据集目录 = ROOT / "研发" / "data" / "dataset"
输出 = ROOT / "优化" / "评估结果" / "检索池敏感性.csv"


def main() -> int:
    hold = [json.loads(x) for x in (数据集目录 / "留出测试集.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    base = json.loads((评估目录 / "评估结果_微调前.json").read_text(encoding="utf-8"))
    ids = None
    vec_b = np.load(评估目录 / "向量_微调前.npy")
    vec_a = np.load(评估目录 / "向量_微调后.npy")
    if vec_b.shape != vec_a.shape:
        raise SystemExit(f"前后向量形状不一致：{vec_b.shape} vs {vec_a.shape}")
    print(f"[敏感性] 检索池 {vec_b.shape[0]} 块，查询 {len(hold)} 条")

    gold = base["留出测试集"]["金标准下标"]
    q_texts = base["留出测试集"]["查询列表"]

    # 重新计算查询向量不可行（需模型），故直接复用评估阶段已写入的逐条排名？
    # 不能复用 —— 这里改为：对「池规模」做子集重算，需要查询向量。
    # 因此本脚本要求评估阶段把查询向量一并缓存；若不存在则退化为提示。
    qv_path = 评估目录 / "查询向量_微调前.npy"
    qa_path = 评估目录 / "查询向量_微调后.npy"
    if not (qv_path.exists() and qa_path.exists()):
        print("[敏感性] 未找到查询向量缓存（评估阶段未导出），改用已有排名列表做池无关的指标复核")
        rows = []
        for tag in ("微调前", "微调后"):
            d = json.loads((评估目录 / f"评估结果_{tag}.json").read_text(encoding="utf-8"))
            m = d["留出测试集"]
            rows.append({"模型": tag, "检索池": vec_b.shape[0], "Recall@1": m["Recall@1"],
                         "Recall@5": m["Recall@5"], "Recall@10": m["Recall@10"], "MRR": m["MRR"],
                         "nDCG@10": m["nDCG@10"]})
        输出.parent.mkdir(parents=True, exist_ok=True)
        with 输出.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[敏感性] 已写出 {输出}")
        for r in rows:
            print(r)
        return 0

    qv_b = np.load(qv_path)
    qv_a = np.load(qa_path)
    rows = []
    for 池大小 in (200, 500, 1000, 2000, vec_b.shape[0]):
        keep = np.arange(min(池大小, vec_b.shape[0]))
        remap = {int(g): i for i, g in enumerate(keep)}
        可行 = [i for i, g in enumerate(gold) if int(g) in remap]
        if not 可行:
            continue
        gs = [remap[int(gold[i])] for i in 可行]
        for tag, dv, qvv in (("微调前", vec_b, qv_b), ("微调后", vec_a, qv_a)):
            m = retrieval_metrics(ranks_from_scores(qvv[可行], dv[keep], gs), ks=(1, 5, 10), ndcg_k=10)
            rows.append({"模型": tag, "检索池": len(keep), "查询数": len(可行),
                         "Recall@1": m["Recall@1"], "Recall@5": m["Recall@5"],
                         "Recall@10": m["Recall@10"], "MRR": m["MRR"], "nDCG@10": m["nDCG@10"]})
    输出.parent.mkdir(parents=True, exist_ok=True)
    with 输出.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[敏感性] 已写出 {输出}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
