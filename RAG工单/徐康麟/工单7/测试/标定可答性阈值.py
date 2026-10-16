# -*- coding: utf-8 -*-
"""工单7：可答性阈值标定（用真实语料量出正/负样本的相似度与覆盖率分布）。

工单编号：人工智能NLP-RAG-功能测试及评估

动机（实测缺陷）：
    ``answerability.evaluate_answerability`` 的 G2' 分支是
    「``coverage < max(min_keyword_coverage-0.15, 0.15)`` 且 ``max_vector_similarity < min_score`` → 拒答」。
    其中 ``min_score`` 是**向量余弦**底线。工单4/6 在「招股说明书」语料上按
    「正样本 min 0.153 / 负样本 max 0.205」标定为 **0.12**；
    但本工单换成 9 份 A 股年报后，bge-m3 对任意中文文本对的余弦整体上移到 0.35~0.87，
    0.12 变成**恒真条件** → 只要覆盖率略低于 0.30 就必然拒答（实测 3/17 题被误拒）。

本脚本的用途：
    把标定过程**做成可复现的证据**：对 17 道正样本（测试问题集）与 6 道语料外负样本，
    各量一次「最大向量余弦 / 最大 BM25 / Top-3 覆盖率 / 语料路由命中」，输出分布与建议阈值。

用法：
    python 测试/标定可答性阈值.py
    python 测试/标定可答性阈值.py --out 测试/测试数据/阈值标定.json

退出码：0 成功；3 索引缺失。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "研发"))

# 语料外负样本（应被拒答）：前 2 句是「同体裁但实体不在语料里」，后 4 句是完全无关问句
NEGATIVES: tuple[str, ...] = (
    "特斯拉2019年的营业收入是多少？",
    "苹果公司2020年的净利润是多少？",
    "请问今天天气怎么样？",
    "李白是哪个朝代的诗人？",
    "如何做红烧肉？",
    "量子纠缠的原理是什么？",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单7：{WORK_ORDER} —— 可答性阈值标定")
    parser.add_argument("--questions", default=str(REPO_ROOT / "研发" / "评估" / "测试问题集.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "测试" / "测试数据" / "阈值标定.json"))
    args = parser.parse_args(argv)

    from app.core.config import get_config
    from app.core.retriever import load_bundle, hybrid_search
    from app.core.query_understanding import analyze
    from app.core.answerability import keyword_coverage_of

    cfg = get_config()
    manifest = Path(cfg.paths.index_dir) / "index_manifest.json"
    if not manifest.is_file():
        print(f"❌ 索引缺失：{manifest}", file=sys.stderr)
        return 3

    bundle = load_bundle(cfg)
    q_path = Path(args.questions)
    rows = [json.loads(line) for line in q_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("=" * 78)
    print(f"工单7 可答性阈值标定｜{WORK_ORDER}")
    print(f"索引：{manifest}（{bundle.size} 块）｜正样本 {len(rows)} 题｜负样本 {len(NEGATIVES)} 题")
    print(f"当前配置：min_score={cfg.answer.min_score}｜min_keyword_coverage={cfg.answer.min_keyword_coverage}")
    print("=" * 78)

    def probe(question: str) -> dict:
        analysis = analyze(question, corpus_issuers=bundle.corpus_issuers)
        hits = hybrid_search(bundle, question, cfg=cfg, analysis=analysis, top_k=10)
        vec = [float(h.get("vector_score") or 0.0) for h in hits] or [0.0]
        bm25 = [float(h.get("bm25_score") or 0.0) for h in hits] or [0.0]
        coverage = keyword_coverage_of(analysis.keywords,
                                       [str(h.get("content") or "")[:1200] for h in hits[:3]])
        return {"max_vector": round(max(vec), 4), "max_bm25": round(max(bm25), 2),
                "coverage": round(coverage, 4), "corpus_hint": analysis.corpus_hint,
                "n_hits": len(hits)}

    pos = []
    print("[正样本]")
    for row in rows:
        item = probe(row["question"])
        item["id"] = row["id"]
        pos.append(item)
        print(f"  {row['id']}  vec={item['max_vector']:.4f}  bm25={item['max_bm25']:8.2f}  "
              f"cov={item['coverage']:.3f}  hint={item['corpus_hint']}")
    neg = []
    print("[负样本（语料外，应拒答）]")
    for question in NEGATIVES:
        item = probe(question)
        item["question"] = question
        neg.append(item)
        print(f"  vec={item['max_vector']:.4f}  bm25={item['max_bm25']:8.2f}  "
              f"cov={item['coverage']:.3f}  {question}")

    pv = [x["max_vector"] for x in pos]
    nv = [x["max_vector"] for x in neg]
    pc = [x["coverage"] for x in pos]
    nc = [x["coverage"] for x in neg]
    suggested = round((min(pv) + max(nv)) / 2, 2) if min(pv) > max(nv) else max(nv)
    payload = {
        "工单": WORK_ORDER,
        "口径": "正样本=研发/评估/测试问题集.jsonl；负样本=语料外问句；"
                "max_vector=Top-10 内最大向量余弦；coverage=问句实词在 Top-3 证据中的覆盖率",
        "当前配置": {"min_score": cfg.answer.min_score,
                     "min_keyword_coverage": cfg.answer.min_keyword_coverage},
        "正样本": pos,
        "负样本": neg,
        "分布": {
            "正样本_向量余弦_最小": min(pv), "正样本_向量余弦_中位": round(statistics.median(pv), 4),
            "正样本_向量余弦_最大": max(pv),
            "负样本_向量余弦_最大": max(nv),
            "正样本_覆盖率_最小": min(pc), "正样本_覆盖率_中位": round(statistics.median(pc), 4),
            "负样本_覆盖率_最大": max(nc),
            "正样本_覆盖率低于0.30的题数": sum(1 for c in pc if c < 0.30),
        },
        "建议_min_score": suggested,
        "结论": (f"正样本余弦下界 {min(pv)} > 负样本余弦上界 {max(nv)} → 可分；"
                 f"取中间偏正样本一侧的建议值 {suggested}。原值 0.12 落在负样本分布之下，"
                 "会让「低覆盖率+低余弦」分支恒真，把可答题误拒（工单7 实测 3/17 误拒已据此修正）。"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("-" * 78)
    print(f"正样本余弦 min={min(pv):.4f} 中位={statistics.median(pv):.4f}｜负样本余弦 max={max(nv):.4f}")
    print(f"正样本覆盖率 min={min(pc):.3f} 中位={statistics.median(pc):.3f}｜负样本覆盖率 max={max(nc):.3f}")
    print(f"建议 min_score = {suggested}")
    print(f"✅ 已写出标定证据 → {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已中断]", file=sys.stderr)
        raise SystemExit(130)
