# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：检索精确度评估（工单硬性要求3：检索精度量化与展示）

对 16 道验收题**逐题**输出：
  1. 检索到的答案（文本）
  2. 检索精确度数值：Precision@k、Recall@k、Hit Rate、MRR、NDCG@k、Answer Accuracy
  3. 引用来源：页码 + block_type + 图像ID / 表格ID / 段落ID
  4. 过程观测：召回候选数、精排耗时、精排前后 Top-K 名次变化（硬性要求2）

相关性判定口径（可复现、不靠人工）
----------------------------------
金标每题给出 key_facts（必须出现的关键事实，如 "1,670"、"25.04%"）与可选的
regex_facts（避免"4"这种裸数字误匹配，如 `4\\s*个部门`）。
一个检索块被判为「相关」当且仅当它**至少命中 1 条关键事实**（或正则）。
  · Hit Rate@k：top-k 里是否至少有一个相关块；
  · Precision@k：top-k 中相关块占比；
  · Recall@k：top-k 命中数 ÷ **全语料**中相关块总数（分母由全量扫描得出）；
  · MRR：第一个相关块名次的倒数；
  · NDCG@k：二值相关性的归一化折损累计增益；
  · Answer Accuracy：最终答案覆盖 key_facts 的比例（事实命中率），
    并给出"全对"（全部事实命中）计数 —— 对应工单的"准确率 ≥90%"口径。

用法
----
    python evaluate_retrieval.py                          # 三策略全跑（默认）
    python evaluate_retrieval.py --strategies hybrid
    python evaluate_retrieval.py --no-answer               # 只算检索指标（不调 LLM，快）
    python evaluate_retrieval.py --rerank-algorithm tfidf   # 换重排算法对比
    python evaluate_retrieval.py --out data/eval/retrieval_eval.json \
        --report docs/04-检索精确度报告.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402  —— 必须最先导入


# ---------------------------------------------------------------------------
# 相关性判定
# ---------------------------------------------------------------------------
def _evidence_facts(gt: dict) -> list[str]:
    """检索相关性用的"证据事实"。

    与"答案必需事实"分开：
      · key_facts/regex_facts  = 答案必须包含的内容（判分用）；
      · evidence_facts         = 该题证据应出现在哪些块里（判检索相关性用）。
    例：id5 只问"几个部门/几个销售处"，答案无需列全名称，但**检索**必须找到
    含这些名称的组织图节点块；id6 答案只需行业名，检索却要找到含数值的图块。
    未单独给出 evidence_facts 时回退到 key_facts（绝大多数题二者一致）。
    """
    return list(gt.get("evidence_facts") or gt.get("key_facts") or [])


def _fact_hits(text: str, gt: dict) -> list[str]:
    """返回该文本命中的（检索相关性）证据事实列表。"""
    if not text:
        return []
    hits = [f for f in _evidence_facts(gt) if f in text]
    for pat in (gt.get("regex_facts") or []):
        try:
            if re.search(pat, text):
                hits.append(f"re:{pat}")
        except re.error:
            continue
    return hits


def corpus_stats(gt: dict, payloads: list[dict]) -> dict:
    """全语料的相关块统计（Recall 的分母）。"""
    n_rel = 0
    pages: set[int] = set()
    for p in payloads:
        if _fact_hits(p.get("text") or "", gt):
            n_rel += 1
            pages.add(int(p.get("page_idx", -1)))
    return {"n_relevant_in_corpus": n_rel, "relevant_pages": sorted(pages)}


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels: list[int]) -> float:
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0


def score_hits(hits: list[dict], gt: dict, k: int, corpus: dict) -> dict:
    """对一次检索结果计算全部指标。

    两套召回口径并存（都写进报告，避免"用宽松口径凑指标"）：
      · recall@k        —— 严格 IR 口径：相关**块**的召回率，分母是全语料相关块数。
                           同一事实常在几十个块里重复（表格行/表头/多处引用），
                           该值天然偏低，用于横向比较策略优劣；
      · fact_coverage@k —— **信息召回率**（工单"重要信息召回率 ≥95%"的口径）：
                           top-k 上下文并集覆盖了本题多少比例的关键事实。
                           它回答的是"重要信息有没有被检索到"，与业务目标一致。
    """
    top = hits[:k]
    rel_flags = [1 if _fact_hits(h.get("text") or "", gt) else 0 for h in top]
    n_rel_top = sum(rel_flags)
    n_rel_corpus = max(corpus["n_relevant_in_corpus"], 1)

    # 信息召回率：把 top-k 拼成一段上下文，看**证据事实**覆盖。
    # 只对 evidence_facts（纯子串）计数，**不套用答案侧的 regex_facts** ——
    # 正则是为"答案措辞"写的（如 id5 的 `4\s*个部门`），而检索到的组织图块写的是
    # "下级共4个"，拿答案正则去卡上下文会得出"召回 0"的假象（实测踩坑）。
    union_text = "\n".join((h.get("text") or "") for h in top)
    ev = _evidence_facts(gt)
    hit_ev = [f for f in ev if f in union_text]
    cov = {
        "coverage": round(len(hit_ev) / len(ev), 4) if ev else 0.0,
        "all_hit": bool(ev) and len(hit_ev) == len(ev),
        "facts_hit": hit_ev,
        "facts_missed": [f for f in ev if f not in hit_ev],
    }
    facts_total = len(ev)

    mrr = 0.0
    for i, r in enumerate(rel_flags, start=1):
        if r:
            mrr = 1.0 / i
            break
    return {
        "k": k,
        "n_hits": len(top),
        "n_relevant_in_topk": n_rel_top,
        "precision@k": round(n_rel_top / max(len(top), 1), 4),
        "recall@k": round(n_rel_top / n_rel_corpus, 4),
        "fact_coverage@k": cov["coverage"],
        "fact_all_covered@k": cov["all_hit"],
        "facts_total": facts_total,
        "facts_covered": cov["facts_hit"],
        "facts_missed": cov["facts_missed"],
        "hit_rate@k": 1.0 if n_rel_top > 0 else 0.0,
        "mrr": round(mrr, 4),
        "ndcg@k": round(ndcg_at_k(rel_flags), 4),
        "n_relevant_in_corpus": corpus["n_relevant_in_corpus"],
        "rel_flags": rel_flags,
    }


def answer_accuracy(answer: str, gt: dict) -> dict:
    """答案事实覆盖率（工单口径的"准确率"）。"""
    facts = list(gt.get("key_facts") or [])
    regexes = list(gt.get("regex_facts") or [])
    hit, miss = [], []
    for f in facts:
        (hit if f in (answer or "") else miss).append(f)
    for pat in regexes:
        try:
            ok = bool(re.search(pat, answer or ""))
        except re.error:
            ok = False
        (hit if ok else miss).append(f"re:{pat}")
    total = len(hit) + len(miss)
    return {
        "facts_total": total,
        "facts_hit": hit,
        "facts_missed": miss,
        "coverage": round(len(hit) / total, 4) if total else 0.0,
        "all_hit": (not miss) and total > 0,
    }


def cite_summary(citations: list[dict]) -> list[dict]:
    """引用来源：页码 + block_type + 图像ID/表格ID/段落ID（工单硬性要求3）。"""
    out = []
    for c in citations or []:
        out.append({
            "source": c.get("source", ""),
            "page": c.get("page_display") or (int(c.get("page_idx", 0)) + 1),
            "page_idx": c.get("page_idx", 0),
            "block_type": c.get("block_type", ""),
            "table_id": c.get("table_id", "") or "",
            "image_id": c.get("image_id", "") or "",
            "paragraph_id": c.get("parent_id") or c.get("chunk_id") or "",
            "org_path": " / ".join(c.get("org_path") or []) if c.get("org_path") else "",
        })
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def evaluate(strategies_list: list[str], k: int, with_answer: bool,
             rerank_algorithm: str, reranker_model: str, limit: int,
             questions_ids: list[int] | None) -> dict:
    from src import fulltext, rag, runtime, strategies

    gt_all = json.load(open(os.path.join("data", "eval", "ground_truth.json"),
                            encoding="utf-8"))
    gts = {g["id"]: g for g in gt_all}
    questions = json.load(open(os.path.join("data", "test_questions.json"),
                               encoding="utf-8"))
    if questions_ids:
        questions = [q for q in questions if q["id"] in questions_ids]
    if limit:
        questions = questions[:limit]

    print(f"[eval] 预热……", flush=True)
    rag.warmup()

    # 全语料（Recall 分母 / 相关性扫描）
    payloads = fulltext.build().payloads
    print(f"[eval] 语料 {len(payloads)} 块；题目 {len(questions)} 道；"
          f"策略 {strategies_list}", flush=True)

    corpus_cache = {qid: corpus_stats(gts[qid], payloads)
                    for qid in (q["id"] for q in questions) if qid in gts}

    rows: list[dict] = []
    t_all = time.time()
    for sidx, strat in enumerate(strategies_list):
        print(f"\n{'=' * 66}\n[eval] 策略 {strat}\n{'=' * 66}", flush=True)
        for qi, q in enumerate(questions, 1):
            gt = gts.get(q["id"])
            if not gt:
                print(f"  [{qi}/{len(questions)}] id={q['id']} 缺金标，跳过", flush=True)
                continue
            cfg = strategies.with_overrides(
                runtime.get(), strategy=strat,
                rerank_algorithm=rerank_algorithm, reranker_model=reranker_model)
            t0 = time.time()
            res = strategies.retrieve(q["question"], cfg)
            retrieval_ms = round((time.time() - t0) * 1000, 1)
            metrics = score_hits(res["hits"], gt, k, corpus_cache[q["id"]])

            row = {
                "id": q["id"],
                "strategy": strat,
                "question": q["question"],
                "metrics": metrics,
                "retrieval_ms": retrieval_ms,
                "retrieval_info": {
                    "arms": res["info"].get("arms"),
                    "recall_counts": res["info"].get("recall_counts"),
                    "recall_ms": res["info"].get("recall_ms"),
                    "fusion_ms": res["info"].get("fusion_ms"),
                    "rerank_ms": res["info"].get("rerank_ms"),
                    "n_candidates": res["info"].get("n_candidates"),
                    "rerank": {kk: vv for kk, vv in (res["info"].get("rerank") or {}).items()
                               if kk != "rank_changes"},
                    "rank_changes": (res["info"].get("rerank") or {}).get("rank_changes", []),
                },
                "hits": [{
                    "rank": i,
                    "source": h.get("source"),
                    "page": int(h.get("page_idx", 0)) + 1,
                    "block_type": h.get("block_type"),
                    "chunk_id": h.get("chunk_id"),
                    "table_id": h.get("table_id", ""),
                    "image_id": h.get("image_id", ""),
                    "org_node": h.get("org_node", ""),
                    "rerank_score": h.get("rerank_score"),
                    "fused_score": h.get("fused_score"),
                    "vector_score": h.get("vector_score"),
                    "fulltext_score": h.get("fulltext_score"),
                    "fact_hits": _fact_hits(h.get("text") or "", gt),
                    "text": (h.get("text") or "")[:400],
                } for i, h in enumerate(res["hits"], start=1)],
            }

            if with_answer:
                t1 = time.time()
                r = rag.ask(q["question"], use_cache=False)
                row["answer"] = r.answer
                row["answer_ms"] = round((time.time() - t1) * 1000, 1)
                row["answer_mode"] = r.mode
                row["answer_accuracy"] = answer_accuracy(r.answer, gt)
                row["citations"] = cite_summary(r.citations)
                row["timings"] = r.timings

            rows.append(row)
            m = metrics
            acc = row.get("answer_accuracy", {}).get("coverage", "-")
            print(f"  [{qi}/{len(questions)}] id={q['id']:>4} {strat:<8} "
                  f"hit={m['hit_rate@k']:.0f} P@{k}={m['precision@k']:.2f} "
                  f"R@{k}={m['recall@k']:.2f} MRR={m['mrr']:.2f} "
                  f"nDCG={m['ndcg@k']:.2f} 答案覆盖={acc} "
                  f"({retrieval_ms:.0f}ms)", flush=True)

    # ---- 汇总 ----
    agg: dict[str, dict] = {}
    for strat in strategies_list:
        rs = [r for r in rows if r["strategy"] == strat]
        if not rs:
            continue
        n = len(rs)
        agg[strat] = {
            "n": n,
            "hit_rate@k": round(sum(r["metrics"]["hit_rate@k"] for r in rs) / n, 4),
            "precision@k": round(sum(r["metrics"]["precision@k"] for r in rs) / n, 4),
            "recall@k": round(sum(r["metrics"]["recall@k"] for r in rs) / n, 4),
            "fact_coverage@k": round(sum(r["metrics"]["fact_coverage@k"] for r in rs) / n, 4),
            "fact_all_covered_rate": round(
                sum(1 for r in rs if r["metrics"]["fact_all_covered@k"]) / n, 4),
            "mrr": round(sum(r["metrics"]["mrr"] for r in rs) / n, 4),
            "ndcg@k": round(sum(r["metrics"]["ndcg@k"] for r in rs) / n, 4),
            "retrieval_ms_mean": round(sum(r["retrieval_ms"] for r in rs) / n, 1),
        }
        if with_answer:
            got = [r for r in rs if "answer_accuracy" in r]
            if got:
                agg[strat]["answer_accuracy"] = round(
                    sum(r["answer_accuracy"]["coverage"] for r in got) / len(got), 4)
                agg[strat]["answer_all_hit_rate"] = round(
                    sum(1 for r in got if r["answer_accuracy"]["all_hit"]) / len(got), 4)
                agg[strat]["answer_ms_mean"] = round(
                    sum(r["answer_ms"] for r in got) / len(got), 1)

    return {
        "workorder": "人工智能NLP-RAG-混合检索任务",
        "k": k,
        "strategies": strategies_list,
        "with_answer": with_answer,
        "rerank_algorithm": rerank_algorithm,
        "reranker_model": reranker_model,
        "n_questions": len({r["id"] for r in rows}),
        "aggregate": agg,
        "rows": rows,
        "elapsed_s": round(time.time() - t_all, 1),
    }


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------
def render_report(result: dict) -> str:
    k = result["k"]
    L: list[str] = []
    A = L.append
    A("# 工单06 检索精确度报告（16 题逐题：答案 + 精确度 + 引用来源）")
    A("")
    A(f"> **工单编号**：人工智能NLP-RAG-混合检索任务  ")
    A(f"> **评估脚本**：`evaluate_retrieval.py`  ")
    A(f"> **评测集**：`data/test_questions.json`（{result['n_questions']} 题）× "
      f"策略 {('、'.join(result['strategies']))}  ")
    A(f"> **相关性口径**：检索块命中金标关键事实即判相关；Recall 分母为全语料相关块数  ")
    A(f"> **精排算法**：{result['rerank_algorithm']}（模型 {result['reranker_model']}）  ")
    A(f"> **总耗时**：{result['elapsed_s']}s")
    A("")
    A("---")
    A("")
    A("## 1. 汇总指标（三策略对比）")
    A("")
    A(f"| 策略 | 题数 | **信息召回率**@{k} | 全召回率 | Hit Rate@{k} | Precision@{k} | "
      f"Recall@{k}(严格) | MRR | NDCG@{k} | 答案准确率 | 答案全对率 | 检索耗时均值 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    label = {"vector": "向量检索", "fulltext": "全文检索", "hybrid": "混合检索"}
    for strat, a in result["aggregate"].items():
        A(f"| {label.get(strat, strat)} | {a['n']} | **{a['fact_coverage@k']:.4f}** | "
          f"{a['fact_all_covered_rate']:.4f} | {a['hit_rate@k']:.4f} | "
          f"{a['precision@k']:.4f} | {a['recall@k']:.4f} | {a['mrr']:.4f} | "
          f"{a['ndcg@k']:.4f} | {a.get('answer_accuracy', '-')} | "
          f"{a.get('answer_all_hit_rate', '-')} | {a['retrieval_ms_mean']}ms |")
    A("")
    A("> 工单验收口径：**准确率 ≥ 90%、召回率 ≥ 95%**。")
    A(">")
    A("> · **信息召回率**（= 关键事实覆盖率）：top-5 上下文覆盖本题关键事实的比例均值 —— "
      "对应工单「确保重要信息的召回率」；")
    A("> · **全召回率**：关键事实被 top-5 全部覆盖的题目占比；")
    A("> · **Recall@k（严格）**：相关块召回率（分母为全语料相关块数）。同一事实在表格行/"
      "表头/多处引用里重复出现，该口径天然偏低，仅用于策略横向比较；")
    A("> · **答案准确率**：最终答案对关键事实的平均覆盖率（生成侧）；")
    A("> · **答案全对率**：关键事实全部出现在答案中的题目占比。")
    A("")
    A("---")
    A("")
    A("## 2. 逐题明细")
    A("")
    for strat in result["strategies"]:
        rows = [r for r in result["rows"] if r["strategy"] == strat]
        if not rows:
            continue
        A(f"### 2.{result['strategies'].index(strat) + 1} 策略：{label.get(strat, strat)}")
        A("")
        for r in rows:
            m = r["metrics"]
            A(f"#### 题 {r['id']}：{r['question']}")
            A("")
            A(f"- **信息召回率**@{k} = {m['fact_coverage@k']:.2f}"
              f"（覆盖 {len(m['facts_covered'])}/{m['facts_total']} 条关键事实"
              + (f"，缺：{'、'.join(m['facts_missed'])}" if m['facts_missed'] else "，全部覆盖")
              + "）")
            A(f"- **检索指标**：Hit Rate@{k} = {m['hit_rate@k']:.0f}，"
              f"Precision@{k} = {m['precision@k']:.2f}，Recall@{k} = {m['recall@k']:.2f}"
              f"（全语料相关块 {m['n_relevant_in_corpus']} 个），"
              f"MRR = {m['mrr']:.2f}，NDCG@{k} = {m['ndcg@k']:.2f}")
            info = r["retrieval_info"]
            rk = info.get("rerank") or {}
            A(f"- **过程观测**：召回通路 {'、'.join(info.get('arms') or [])}，"
              f"候选 {info.get('n_candidates')} 条，召回 {info.get('recall_ms')}ms"
              + (f"，融合 {info.get('fusion_ms')}ms" if info.get("fusion_ms") else "")
              + f"，精排 {info.get('rerank_ms')}ms"
              f"（算法 {rk.get('algorithm')}，候选 {rk.get('n_in')} → {rk.get('n_out')}，"
              f"名次变化 {rk.get('n_changed')} 条"
              + ("，**已降级**" if rk.get("degraded") else "") + "）")
            if r.get("answer") is not None:
                acc = r.get("answer_accuracy") or {}
                A(f"- **答案**（覆盖 {acc.get('coverage', 0):.0%}，"
                  f"{'全部命中' if acc.get('all_hit') else '缺：' + '、'.join(acc.get('facts_missed') or [])}）："
                  f"{r.get('answer')}")
                cits = r.get("citations") or []
                if cits:
                    bits = []
                    for c in cits[:5]:
                        part = (f"{c['source'].replace('.pdf', '')} 第{c['page']}页"
                                f"｜{c['block_type']}")
                        if c.get("table_id"):
                            part += f"｜表格ID:{c['table_id']}"
                        if c.get("image_id"):
                            part += f"｜图像ID:{c['image_id']}"
                        if c.get("org_path"):
                            part += f"｜组织路径:{c['org_path']}"
                        elif c.get("paragraph_id"):
                            part += f"｜段落ID:{c['paragraph_id']}"
                        bits.append(part)
                    A(f"- **引用来源**：{'；'.join(bits)}")
                t = r.get("timings") or {}
                if t:
                    A(f"- **耗时**：总计 {t.get('total_ms')}ms"
                      f"（检索 {t.get('retrieve_ms')}ms / 精排 {t.get('rerank_ms')}ms / "
                      f"生成 {t.get('gen_ms')}ms）")
            hits = r.get("hits") or []
            if hits:
                A("")
                A(f"| 名次 | 来源 | 页码 | block_type | 定位ID | 精排分 | 命中事实 |")
                A("| --- | --- | --- | --- | --- | --- | --- |")
                for h in hits:
                    loc = h.get("table_id") or h.get("image_id") or h.get("org_node") or h.get("chunk_id") or ""
                    score = h.get("rerank_score")
                    A(f"| {h['rank']} | {str(h.get('source', '')).replace('.pdf', '')} | "
                      f"{h.get('page')} | {h.get('block_type')} | {loc} | "
                      f"{('%.4f' % score) if isinstance(score, (int, float)) else '-'} | "
                      f"{'、'.join(h.get('fact_hits') or []) or '—'} |")
            A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单06 检索精确度评估（16 题）")
    ap.add_argument("--strategies", default="hybrid,vector,fulltext",
                    help="逗号分隔：hybrid,vector,fulltext")
    ap.add_argument("--top-k", type=int, default=5, help="评估的 k（默认 5）")
    ap.add_argument("--no-answer", action="store_true", help="跳过 LLM 答案生成（快）")
    ap.add_argument("--rerank-algorithm", default=None,
                    help="model | llm | tfidf | adaptive（默认取运行时设置）")
    ap.add_argument("--reranker-model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 题（冒烟）")
    ap.add_argument("--ids", default="", help="只评估指定题号，如 1,5,6")
    ap.add_argument("--out", default=os.path.join("data", "eval", "retrieval_eval.json"))
    ap.add_argument("--report", default=os.path.join("docs", "04-检索精确度报告.md"))
    args = ap.parse_args()

    from src import config, runtime

    strategies_list = [s.strip() for s in args.strategies.split(",") if s.strip()]
    cur = runtime.get()
    algo = args.rerank_algorithm or cur.rerank_algorithm
    model = args.reranker_model or cur.reranker_model
    ids = [int(x) for x in args.ids.split(",") if x.strip()] or None

    result = evaluate(strategies_list, args.top_k, not args.no_answer,
                      algo, model, args.limit, ids)
    result["embedding_model"] = cur.embedding_model

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(render_report(result))

    print(f"\n[eval] 完成：{args.out} / {args.report}")
    for strat, a in result["aggregate"].items():
        print(f"  {strat:<9} hit@{args.top_k}={a['hit_rate@k']:.3f} "
              f"P@{args.top_k}={a['precision@k']:.3f} R@{args.top_k}={a['recall@k']:.3f} "
              f"MRR={a['mrr']:.3f} nDCG={a['ndcg@k']:.3f} "
              f"答案准确率={a.get('answer_accuracy', '-')}")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
