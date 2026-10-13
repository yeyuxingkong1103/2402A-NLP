# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 金标准自检（评估前的质量闸门）

为什么需要这一步
----------------
上游 `evaluate_retrieval.py` 的检索相关性判定用的是**纯子串匹配**：
一个块只要有任意 `key_facts` 出现在其正文里，就记为一个相关块。于是：

  · key_fact 写成**转述**而非原文 → 全语料 0 命中 → Recall 分母被兜成 1、
    该题所有块的 rel_flag=0 → HitRate/MRR/nDCG 全 0，看起来像"检索失败"，
    实际是**金标准写错了**。这类假阴性必须在上评估台之前排掉。
  · key_fact 太短（如"营业收入"）→ 命中成百上千块 → Recall 分母虚高，
    指标被稀释得没有区分度。

本脚本对每题逐条 key_fact 统计全语料命中块数，给出：
  · 命中 0 条的 fact —— 必须改写为原文表述（错误）；
  · 命中 > 200 条的 fact —— 建议加长（区分度不足）；
  · 每题的"相关块数（Recall 分母）"与命中页，便于人工核对页号是否正确。

用法
----
    python scripts/validate_ground_truth.py
    python scripts/validate_ground_truth.py --max-hits 200     # 自定义"过泛"阈值
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 金标准自检")
    ap.add_argument("--max-hits", type=int, default=200,
                    help="单条 fact 命中块数超过该值视为过泛（默认 200）")
    args = ap.parse_args()

    chunks_path = os.path.join(ROOT, "data", "chunks", "chunks.json")
    gt_path = os.path.join(ROOT, "data", "eval", "ground_truth.json")
    for p in (chunks_path, gt_path):
        if not os.path.isfile(p):
            raise SystemExit(f"缺少文件：{p}")

    chunks = json.load(open(chunks_path, encoding="utf-8"))
    gts = json.load(open(gt_path, encoding="utf-8"))
    print(f"语料块 {len(chunks)}；金标题 {len(gts)} 道\n")

    # 预建 (source, page_idx) → 块数，用于核对 page_hints
    texts = [(c.get("text") or "",
              c.get("source") or "",
              int(c.get("page_idx", -1))) for c in chunks]

    n_err = n_warn = 0
    for gt in gts:
        facts = list(gt.get("evidence_facts") or gt.get("key_facts") or [])
        print(f"--- id={gt['id']} [{gt.get('qtype','?')}] {gt['question'][:44]}")
        total_rel = 0
        for f in facts:
            n = sum(1 for t, _s, _p in texts if f in t)
            if n == 0:
                flag, n_err = "✗ 全语料 0 命中（必须改成原文表述）", n_err + 1
            elif n > args.max_hits:
                flag, n_warn = f"! 命中 {n} 块，过泛（区分度不足）", n_warn + 1
            else:
                flag = "ok"
            if n == 0 or n > args.max_hits:
                print(f"      {f!r:46s} n={n:<6d} {flag}")
        # 整题相关块数 = 命中任意 fact 的块
        for t, _s, _p in texts:
            if any(f in t for f in facts):
                total_rel += 1
        print(f"      相关块数（Recall 分母）= {total_rel}")

        # page_hints 核对
        # page_hints：优先用 page_hints_by_source（多跳题一题多源）
        ph_by_src = gt.get("page_hints_by_source") or {
            (gt.get("source") or ""): (gt.get("page_hints") or [])}
        for src, pages in ph_by_src.items():
            for ph in pages:
                n = sum(1 for t, s, p in texts
                        if s == src and p + 1 == ph and any(f in t for f in facts))
                mark = "ok" if n else "未命中（该页无 evidence_fact，或页号口径不符）"
                print(f"      page_hint 第{ph}页（{src[:24]}…）相关块 {n} {mark}")

    print(f"\n[汇总] 0 命中事实 {n_err} 条，过泛事实 {n_warn} 条")
    if n_err:
        print("        → 请修正 ground_truth.json 后再跑评估，否则指标会假性偏低。")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
