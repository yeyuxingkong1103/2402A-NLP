# -*- coding: utf-8 -*-
"""
工单15 —— 跨模态检索 6 题测试脚本（双模式）

模式 A（--mode ragflow）：对接真实 RAGFlow 服务，测 CN100347506C.pdf 的 6 个问题
模式 B（--mode local，默认）：在本交付项目的知识库上演示跨模态检索的
       「视觉引用识别 + 三路召回 RRF + 跨模态重排 + Prompt」，
       题目用《招股说明书2.pdf》的图像题（ID 5、6）作为等价验证。

A 模式前置：RAGFlow 已启动，环境变量 RAGFLOW_API_KEY / RAGFLOW_BASE / RAGFLOW_CHAT_ID
用法：
  python 工单15/code/test_cross_modal.py --mode local
  python 工单15/code/test_cross_modal.py --mode ragflow --tag before
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # 工单15
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# ---- 工单15 的 6 个测试问题（CN100347506C.pdf，第11页 图3）
QUESTIONS_15 = [
    {"id": 1, "question": "根据专利文本，本发明主要涉及哪种物料的分配装置？",
     "answer": "块状散料", "visual": False},
    {"id": 2, "question": "根据专利文本，本发明的分散装置包含以下哪个组件？",
     "answer": "链条", "visual": False},
    {"id": 3, "question": "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？",
     "answer": "位于编号12的部件之内", "visual": True},
    {"id": 4, "question": "在文件中第11页图3中，编号14的部件位于整个装置的哪个位置？",
     "answer": "顶部", "visual": True},
    {"id": 5, "question": "根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？",
     "answer": "部件13", "visual": True},
    {"id": 6, "question": "在文件中第11页图3的装置中，如果需要调整链条的位置，需要操作哪个部件?",
     "answer": "部件11", "visual": True},
]

# ---- 本地等价验证题（招股说明书2.pdf 的图像题，答案已核实）
QUESTIONS_LOCAL = [
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，"
                          "其中大客户销售部有几个销售处构成？",
     "answer": "销售部4个部门，大客户销售部6个销售处", "visual": True,
     "expected_pages": [39]},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与"
                          "增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
     "answer": "汽车电子最快(14.0%)，IC卡负增长(-2.0%)", "visual": True,
     "expected_pages": [72]},
]


def run_local(top_k: int = 5, tag: str = "local", baseline: bool = False):
    from src import config
    from src.build_kb import load_retriever
    sys.path.insert(0, str(HERE / "code"))
    from cross_modal_retrieval import (CrossModalRetriever, extract_visual_ref,
                                       answer as cm_answer)

    retriever = load_retriever("share12_struct")
    mode = "baseline" if baseline else "cross_modal"
    cm = CrossModalRetriever(retriever, pdf_path=config.SHARE_2, mode=mode)
    print(f"[模式] {mode}（{'优化前：纯文本混合检索' if baseline else '优化后：查询理解+三路召回+跨模态重排'}）")

    rows = []
    for q in QUESTIONS_LOCAL:
        t0 = time.perf_counter()
        ans, res = cm_answer(q["question"], cm, top_k=top_k)
        el = time.perf_counter() - t0
        ref = res["visual_ref"]
        hits = [c["page"] for c in res["candidates"]]
        top_hit = any(p in hits[:3] for p in q.get("expected_pages", []))
        # 答案要点命中率：把标准答案按逗号/顿号拆成要点逐条核对
        #   该要点含数字 → 核对数字是否出现；否则 → 核对前 4 个字是否出现
        import re as _re
        keys = [k.strip() for k in _re.split(r"[,，、]", q["answer"]) if k.strip()]
        hit_keys = []
        for k in keys:
            nums = _re.findall(r"\d+\.?\d*%?", k)
            if nums:
                ok = all(n in ans for n in nums)
            else:
                ok = k[:4] in ans
            if ok:
                hit_keys.append(k)
        rows.append({
            "id": q["id"], "question": q["question"], "expected": q["answer"],
            "answer": ans, "visual_ref": ref, "mode": mode,
            "enhanced_query": res["enhanced_query"][:200],
            "candidate_pages": hits, "top3_hit_target_page": top_hit,
            "要点命中率": round(len(hit_keys) / max(len(keys), 1), 2),
            "elapsed": round(el, 2),
        })
        print(f"Q{q['id']} top3命中目标页={top_hit} 要点命中={rows[-1]['要点命中率']:.0%} "
              f"{el:.1f}s :: {ans[:80]}")

    outdir = HERE / "产出物"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"跨模态检索结果_{tag}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    acc = sum(r["top3_hit_target_page"] for r in rows) / max(len(rows), 1)
    ks = sum(r["要点命中率"] for r in rows) / max(len(rows), 1)
    print(f"\nTop-3 命中含图页比例：{acc:.0%}｜平均要点命中率：{ks:.0%}")
    return rows


def run_ragflow(tag: str = "before"):
    import requests
    base = os.getenv("RAGFLOW_BASE", "http://127.0.0.1:9380")
    key = os.getenv("RAGFLOW_API_KEY", "")
    chat = os.getenv("RAGFLOW_CHAT_ID", "")
    if not key or not chat:
        print("请设置环境变量 RAGFLOW_API_KEY 与 RAGFLOW_CHAT_ID")
        sys.exit(1)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    rows = []
    for q in QUESTIONS_15:
        t0 = time.perf_counter()
        r = requests.post(f"{base}/api/v1/chats/{chat}/completions", headers=headers,
                          json={"question": q["question"], "stream": False}, timeout=180)
        el = time.perf_counter() - t0
        data = r.json().get("data", {}) if r.status_code == 200 else {}
        ans = data.get("answer", "")
        chunks = (data.get("reference") or {}).get("chunks") or []
        # 跨模态验收：图文题的 Top-3 必须命中第11页 / 含图3描述
        if q["visual"]:
            top3 = chunks[:3]
            hit = any(("11" in str(c.get("page_num", ""))) or ("图3" in c.get("content", ""))
                      for c in top3)
        else:
            hit = q["answer"] in ans
        rows.append({"id": q["id"], "question": q["question"], "expected": q["answer"],
                     "answer": ans, "top3_hit": hit, "elapsed": round(el, 2),
                     "top_chunks": [{"page": c.get("page_num"),
                                     "content": (c.get("content") or "")[:150]}
                                    for c in chunks[:3]]})
        print(f"Q{q['id']} {'✅' if hit else '❌'} {el:.1f}s :: {ans[:80]}")

    outdir = HERE / "产出物"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"6题测试结果_{tag}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n准确率 {sum(r['top3_hit'] for r in rows)}/{len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="local", choices=["local", "ragflow"])
    ap.add_argument("--tag", default="before")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--baseline", action="store_true",
                    help="优化前对照：纯文本混合检索，不启用跨模态通道")
    args = ap.parse_args()
    if args.mode == "local":
        run_local(args.top_k, args.tag, baseline=args.baseline)
    else:
        run_ragflow(args.tag)


if __name__ == "__main__":
    main()
