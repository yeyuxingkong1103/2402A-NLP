# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 10 题 RAG 检索结果逐题输出（需求硬性要求3）

对 10 道测试题逐题产出：
  1. 问题原文 + 题型
  2. 检索到的 Top-K 文档片段（含原文内容）
  3. 引用来源：文件名 + 页码 + block_type（文本/表格/图像）+ 段落ID/表格ID/图像ID
  4. 最终答案（由 LLM 基于检索结果生成）
  5. 检索耗时（召回耗时 + Reranker 精排耗时 + LLM 生成耗时）
  6. 金标准对照：命中/遗漏的关键事实（供归因分析用）

检索逻辑**完全复用** 01~06 工单实现的 RAG 系统（`src/strategies.py` 调度
向量/全文/混合三策略 + `src/rerankers.py` 精排 + `src/rag.py` 生成），
本脚本只负责"调用 + 留痕 + 排版"，不重写任何检索逻辑。

用法
----
    python run_rag_questions.py                       # 默认 hybrid 策略
    python run_rag_questions.py --strategy vector
    python run_rag_questions.py --ids 3,10            # 只跑指定题
    python run_rag_questions.py --no-answer           # 只检索不生成（快）

输出
----
    data/eval/rag_results_<strategy>.json    机器可读全量留痕
    docs/04-RAG检索结果-<strategy>.md        人读逐题报告
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402  —— 必须最先导入

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_FRAG_CHARS = 700        # 报告中每个片段展示的原文上限（JSON 保留全文）


def _load_gt() -> dict:
    p = os.path.join(ROOT, "data", "eval", "ground_truth.json")
    return {g["id"]: g for g in json.load(open(p, encoding="utf-8"))}


def _load_questions(ids: list[int] | None) -> list[dict]:
    p = os.path.join(ROOT, "data", "test_questions.json")
    qs = json.load(open(p, encoding="utf-8"))
    return [q for q in qs if not ids or q["id"] in ids]


def _cite_of(hit: dict) -> dict:
    """命中块 → 引用来源（文件名 + 页码 + block_type + 块ID）。"""
    return {
        "source": hit.get("source"),
        "page": (int(hit["page_idx"]) + 1) if hit.get("page_idx") is not None else None,
        "block_type": hit.get("block_type"),
        "chunk_id": hit.get("chunk_id"),
        "parent_id": hit.get("parent_id"),
        "table_id": hit.get("table_id"),
        "image_id": hit.get("image_id"),
        "heading_path": hit.get("heading_path") or [],
    }


def _fact_check(text: str, gt: dict) -> dict:
    facts = list(gt.get("key_facts") or [])
    hit = [f for f in facts if f in (text or "")]
    return {"facts": facts, "hit": hit, "miss": [f for f in facts if f not in hit],
            "coverage": round(len(hit) / len(facts), 4) if facts else 0.0}


def run(strategy: str, ids: list[int] | None, with_answer: bool, top_k: int) -> dict:
    from src import rag, runtime

    gts = _load_gt()
    questions = _load_questions(ids)

    cur = runtime.get()
    print(f"[run] 策略={strategy} 嵌入={cur.embedding_model} "
          f"精排={cur.reranker_model}/{cur.rerank_algorithm} 题数={len(questions)}",
          flush=True)
    rag.warmup()

    # 关键：走 rag.retrieve（= Query理解 + 文档路由 + 召回 + 精排），
    # 与 rag.ask 生成答案时**同一条链路**。若改用 strategies.retrieve 裸检索，
    # 会跳过 doc_router，导致"检索留痕"与"答案实际看到的片段"不一致
    # （实测：id6 中国人寿题的裸检索被中信证券的同名表格挤占，而答案链路已路由过滤）。
    runtime.update(strategy=strategy)

    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        gt = gts.get(q["id"], {})

        t0 = time.time()
        hits, analysis, timings_rt = rag.retrieve(q["question"])
        retrieval_ms = round((time.time() - t0) * 1000, 1)
        info = analysis.get("retrieval_info") or {}
        route = analysis.get("doc_route") or {}
        rrk = analysis.get("rerank_info") or {}

        record: dict = {
            "id": q["id"],
            "qtype": gt.get("qtype"),
            "question": q["question"],
            "strategy": strategy,
            "retrieval_ms": retrieval_ms,
            "doc_route": route,
            "timings": {
                "analyze_ms": timings_rt.get("analyze_ms"),
                "retrieve_ms": timings_rt.get("retrieve_ms"),
                "recall_ms": info.get("recall_ms"),
                "fusion_ms": info.get("fusion_ms"),
                "rerank_ms": timings_rt.get("rerank_ms"),
                "retrieval_total_ms": retrieval_ms,
            },
            "retrieval_info": {k: v for k, v in info.items()
                               if k in ("strategy", "arms", "recall_counts", "fusion",
                                        "weights", "top_k_recall", "top_k_rerank")},
            "rerank_info": {k: v for k, v in rrk.items()
                            if k in ("n_in", "elapsed_ms", "degraded", "algorithm",
                                     "model", "rank_changes")},
            "hits": [
                {**_cite_of(h),
                 "rank": r,
                 "rerank_score": h.get("rerank_score"),
                 "fused_score": h.get("fused_score"),
                 "ranks": h.get("ranks"),
                 "text": h.get("text") or ""}
                for r, h in enumerate(hits[:top_k], start=1)
            ],
        }
        record["retrieval_fact_check"] = _fact_check(
            "\n".join(h["text"] for h in record["hits"]), gt)

        if with_answer:
            r0 = time.time()
            ans = rag.ask(q["question"], use_cache=False)
            record["answer"] = ans.answer
            record["answer_mode"] = ans.mode
            record["answer_model"] = ans.model
            record["citations"] = ans.citations
            record["answer_timings"] = ans.timings
            record["answer_wall_ms"] = round((time.time() - r0) * 1000, 1)
            record["answer_fact_check"] = _fact_check(ans.answer, gt)
            record["contexts"] = [
                {**_cite_of(c), "text": c.get("text") or ""} for c in ans.contexts
            ]
        record["ground_truth"] = {
            "reference": gt.get("reference"),
            "key_facts": gt.get("key_facts"),
            "unit": gt.get("unit"),
        }
        rows.append(record)

        a = record.get("answer_fact_check", {})
        print(f"  [{i}/{len(questions)}] id={q['id']:<3} 检索{retrieval_ms:>7.1f}ms "
              f"命中片段{len(record['hits'])} 检索事实覆盖={record['retrieval_fact_check']['coverage']:.2f}"
              + (f" 答案事实覆盖={a.get('coverage', 0):.2f}" if with_answer else ""),
              flush=True)

    return {"strategy": strategy, "top_k": top_k,
            "embedding_model": cur.embedding_model,
            "reranker_model": cur.reranker_model,
            "rerank_algorithm": cur.rerank_algorithm,
            "rows": rows}


def render(result: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# RAG 检索结果逐题输出（工单07）\n")
    A(f"> **工单编号**：人工智能NLP-RAG-功能测试及评估  ")
    A(f"> **检索策略**：`{result['strategy']}`　"
      f"**嵌入模型**：`{result['embedding_model']}`　"
      f"**Reranker**：`{result['reranker_model']}`（算法 `{result['rerank_algorithm']}`）  ")
    A(f"> **Top-K**：{result['top_k']}　**题型覆盖**：事实型/数值型/表格型/图像型/多跳推理型\n")
    A("检索逻辑复用 01~06 工单实现的 RAG 系统（`src/strategies.py` / `src/rerankers.py` / `src/rag.py`），"
      "本报告仅做调用留痕与排版。\n")
    A("---\n")

    for r in result["rows"]:
        A(f"## 问题 {r['id']}（{r.get('qtype') or '-'}）\n")
        A(f"**问题原文**：{r['question']}\n")

        t = r["timings"]
        A("**检索耗时**："
          f"召回 {t.get('recall_ms')} ms ＋ 融合 {t.get('fusion_ms')} ms ＋ "
          f"Reranker 精排 {t.get('rerank_ms')} ms ＝ 检索合计 {t.get('retrieval_total_ms')} ms")
        if r.get("answer_wall_ms") is not None:
            at = r.get("answer_timings") or {}
            gen = at.get("llm_ms") or at.get("generate_ms")
            A(f"；LLM 生成 {gen if gen is not None else r['answer_wall_ms']} ms"
              f"（端到端 {r['answer_wall_ms']} ms）")
        A("")

        ri = r.get("retrieval_info") or {}
        A(f"**检索配置**：arms={ri.get('arms')}　recall={ri.get('recall_counts')}　"
          f"fusion={ri.get('fusion')}　weights={ri.get('weights')}　"
          f"top_k_recall={ri.get('top_k_recall')} → top_k_rerank={ri.get('top_k_rerank')}")
        dr = r.get("doc_route") or {}
        if dr:
            A(f"**文档路由**：{dr.get('reason')}"
              + (f"　命中={dr.get('matched')}" if dr.get("matched") else "")
              + (f"　兜底放开={dr.get('fallback_opened')}" if dr.get("fallback_opened") else ""))
        rr = r.get("rerank_info") or {}
        if rr:
            A(f"**精排**：输入 {rr.get('n_in')} 条，用时 {rr.get('elapsed_ms')} ms，"
              f"降级={rr.get('degraded')}")
        A("")

        A(f"### 检索到的 Top-{len(r['hits'])} 片段\n")
        for h in r["hits"]:
            ids = []
            if h.get("chunk_id"):
                ids.append(f"chunk_id={h['chunk_id']}")
            if h.get("table_id"):
                ids.append(f"table_id={h['table_id']}")
            if h.get("image_id"):
                ids.append(f"image_id={h['image_id']}")
            if h.get("parent_id"):
                ids.append(f"parent_id={h['parent_id']}")
            A(f"**[{h['rank']}]** 来源 `{h['source']}` 第 **{h['page']}** 页　"
              f"block_type=`{h['block_type']}`　{'　'.join(ids)}")
            if h.get("heading_path"):
                A(f"　　章节：{' > '.join(h['heading_path'])}")
            A(f"　　精排分={h.get('rerank_score')}　融合分={h.get('fused_score')}　"
              f"各路名次={h.get('ranks')}")
            body = h["text"].replace("\n", "\n    ")
            A(f"\n    {body[:MAX_FRAG_CHARS]}"
              + ("　…（截断，全文见 JSON）" if len(h["text"]) > MAX_FRAG_CHARS else ""))
            A("")

        if r.get("answer") is not None:
            A("### 最终答案（LLM 基于检索结果生成）\n")
            A(f"> {r['answer']}\n")
            A(f"模式：`{r.get('answer_mode')}`　模型：`{r.get('answer_model')}`\n")

        A("### 金标准对照\n")
        rc = r["retrieval_fact_check"]
        A(f"- **检索**：关键事实覆盖 {rc['coverage']:.2f}"
          f"（命中 {rc['hit']}；遗漏 {rc['miss']}）")
        if r.get("answer_fact_check"):
            ac = r["answer_fact_check"]
            A(f"- **答案**：关键事实覆盖 {ac['coverage']:.2f}"
              f"（命中 {ac['hit']}；遗漏 {ac['miss']}）")
        gt = r.get("ground_truth") or {}
        if gt.get("unit"):
            A(f"- **单位口径**：{gt['unit']}")
        A(f"- **参考答案**：{gt.get('reference')}")
        A("\n---\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 · 10 题 RAG 检索结果逐题输出")
    ap.add_argument("--strategy", default="hybrid",
                    help="vector | fulltext | hybrid")
    ap.add_argument("--ids", default="", help="只跑指定题号，如 3,10")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-answer", action="store_true", help="只检索不生成")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()] or None
    result = run(args.strategy, ids, not args.no_answer, args.top_k)

    os.makedirs(os.path.join(ROOT, "data", "eval"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out_json = os.path.join(ROOT, "data", "eval", f"rag_results_{args.strategy}.json")
    out_md = os.path.join(ROOT, "docs", f"04-RAG检索结果-{args.strategy}.md")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(render(result))
    print(f"\n[run] 完成：{out_json}\n           {out_md}")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
