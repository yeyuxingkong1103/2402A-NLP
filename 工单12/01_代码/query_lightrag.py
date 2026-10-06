# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 关联工单：人工智能NLP-RAG-基于PDF文档的问答系统 | 人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 模块：lightrag_rag/query_lightrag —— LightRAG 检索问答（工单 12）
# 说明：加载已构建的 LightRAG 图存储/向量存储，用不同检索模式（naive / local / global / mix）
#       对问题做「检索 + 本地模型生成」；把**检索到的上下文块**与页码一并落盘，供与常规 RAG 对比。
# 用法（在 .venv_lightrag 下运行）：
#   python lightrag_rag\query_lightrag.py --mode mix --q "武汉兴图新科…注册资本是多少？"
#   python lightrag_rag\query_lightrag.py --eval --modes mix,naive,local
import argparse
import asyncio
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.stdout.reconfigure(encoding="utf-8")

Q1 = os.path.join(ROOT, "evaluation", "eval_questions.json")
Q2 = os.path.join(ROOT, "evaluation", "eval_questions_pdf2.json")
CHUNKS = os.path.join(ROOT, "data", "index", "chunks.jsonl")
PAGE_RE = re.compile(r"【(.+?)·第(\d+)页】")

_CHUNKS = {}


def load_questions():
    qs = []
    for p, key in ((Q2, "gold_page"), (Q1, "page")):
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for it in (d.get("questions") or d.get("items") or []):
            qs.append({"id": it["id"], "q": it["question"],
                       "gold": set(str(x) for x in (it.get(key) or [])),
                       "gold_answer": it.get("gold_answer") or "",
                       "src": os.path.basename(p)})
    return qs


def chunk_index():
    """(doc,page) → 该页若干块文本片段，用于把 LightRAG 返回的块映射回招股书页码。"""
    if _CHUNKS:
        return _CHUNKS
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("type") == "image":
                continue
            t = re.sub(r"\s+", "", c.get("text") or "")
            if len(t) >= 30:
                _CHUNKS[(c.get("doc"), str(c.get("page")))] = t
    return _CHUNKS


def map_page(content):
    """把 LightRAG 的块内容映射回 (doc, page)。"""
    if not content:
        return None, None
    m = PAGE_RE.search(content)
    if m:
        return m.group(1), m.group(2)
    t = re.sub(r"\s+", "", content)
    idx = chunk_index()
    for (doc, page), txt in idx.items():
        if t[:60] and (t[:60] in txt or txt[:60] in t):
            return doc, page
    return None, None


def norm_chunk(c):
    """LightRAG 返回的 chunk 元素 → {text, doc, page}。"""
    if isinstance(c, str):
        text = c
    else:
        text = c.get("content") or c.get("text") or ""
    doc, page = map_page(text)
    return {"text": text[:800], "doc": doc, "page": page}


async def ask(rag, q, mode, top_k, chunk_top_k, only_context=False, opt=False):
    from lightrag import QueryParam
    param = QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k,
                       only_need_context=only_context, enable_rerank=False)
    out = await rag.aquery_llm(q, param=param)
    data = (out or {}).get("data") or {}
    resp = (out or {}).get("llm_response") or {}
    ans = resp.get("content") if isinstance(resp, dict) else str(resp)
    chunks = [norm_chunk(c) for c in (data.get("chunks") or [])]
    if opt:
        chunks = graph_first(q, data.get("entities") or [], chunks)
    return {"answer": (ans or "").strip(), "chunks": chunks,
            "entities": data.get("entities") or [], "relationships": data.get("relationships") or []}


_CHUNK_TXT = {}


def chunk_text_of(cid):
    """chunk id -> 文本（用于把图谱实体的来源块换回页码）。"""
    if not _CHUNK_TXT:
        p = os.path.join(ROOT, "data", "lightrag_store", "kv_store_text_chunks.json")
        if os.path.exists(p):
            _CHUNK_TXT.update(json.load(open(p, encoding="utf-8")))
    v = _CHUNK_TXT.get(cid)
    if isinstance(v, dict):
        return v.get("content") or ""
    return v or ""


def graph_first(query, entities, chunks, max_extra=1):
    """LightRAG 检索优化（思路同工单 09「只补漏不抢位」，更轻）：
    ① 只取**问题里真正出现**的实体（题干命中）；
    ② 跳过 organization / concept 这类跨页泛实体（它们 source_id 遍布全书，补页等于噪声）；
    ③ 取「指标/数据/事件」类实体的**首个来源块所在页**，仅当该页原本未被向量块召回时，补 1 页并置于最前。
    这样对「单值事实题」能把实体所在页抬到最前，且不会把正确页挤下去。"""
    if not entities:
        return chunks
    qlow = query.lower()
    seen = {c["page"] for c in chunks if c["page"]}
    skip_types = {"organization", "concept", "person"}
    extra = []
    for e in entities:
        name = (e.get("entity_name") or "").strip()
        etype = (e.get("entity_type") or "").lower()
        if not name or name.lower() not in qlow or etype in skip_types:
            continue
        for cid in str(e.get("source_id") or "").split("<SEP>"):
            txt = chunk_text_of(cid.strip())
            doc, page = map_page(txt)
            if page and page not in seen:
                extra.append({"text": (txt or "")[:800], "doc": doc, "page": page, "from": "graph"})
                seen.add(page)
                break
        if len(extra) >= max_extra:
            break
    return extra + chunks


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mix")
    ap.add_argument("--q", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--modes", default="mix,naive,local")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--chunk-topk", type=int, default=8)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--opt", action="store_true",
                    help="开启优化：图谱实体命中页「只补漏不抢位」")
    a = ap.parse_args()

    sys.path.insert(0, HERE)
    from build_lightrag import build_rag
    # 检索/生成统一用本地 qwen2:7b（与常规 RAG 同模型，保证对比公平）；
    # 图谱构建阶段可用 LR_LLM=deepseek 加速，二者互不影响。
    rag = build_rag(os.environ.get("LR_QUERY_LLM", "ollama"))
    await rag.initialize_storages()

    if a.q and not a.eval:
        r = await ask(rag, a.q, a.mode, a.topk, a.chunk_topk, opt=a.opt)
        print("Q:", a.q, "\nmode:", a.mode)
        print("A:", r["answer"][:400])
        print("实体 %d 个 / 关系 %d 条 / 上下文块 %d 个" %
              (len(r["entities"]), len(r["relationships"]), len(r["chunks"])))
        for c in r["chunks"][:5]:
            print("   · p%s %s" % (c["page"], (c["text"] or "")[:80].replace("\n", " ")))
        if a.dump:
            print("ENTITY SAMPLE:", json.dumps(r["entities"][:2], ensure_ascii=False)[:600])
        await rag.finalize_storages()
        return

    os.makedirs(a.outdir, exist_ok=True)
    qs = load_questions()
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        rows = []
        t0 = time.time()
        for i, it in enumerate(qs, 1):
            t1 = time.time()
            r = await ask(rag, it["q"], mode, a.topk, a.chunk_topk, opt=a.opt)
            pages = [c["page"] for c in r["chunks"] if c["page"]]
            rank = next((k for k, p in enumerate(pages, 1) if p in it["gold"]), None)
            rows.append({"id": it["id"], "question": it["q"], "gold_pages": sorted(it["gold"]),
                         "gold_answer": it["gold_answer"], "top_pages": pages[:8], "hit_rank": rank,
                         "answer": r["answer"],
                         "contexts": [c["text"] for c in r["chunks"]],
                         "ctx_pages": [c["page"] for c in r["chunks"] if c["page"]],
                         "n_entities": len(r["entities"]), "n_relations": len(r["relationships"]),
                         "cost_ms": int((time.time() - t1) * 1000)})
            print("  [%s %s] id%-4s rank=%-4s pages=%s" %
                  (mode, "OK" if rank else "MISS", it["id"], rank, pages[:5]), flush=True)
        n = len(rows)
        h1 = sum(1 for r in rows if r["hit_rank"] == 1)
        h3 = sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 3)
        h5 = sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 5)
        mrr = sum((1.0 / r["hit_rank"]) if r["hit_rank"] else 0.0 for r in rows) / n
        summary = {"mode": mode + ("+opt" if a.opt else ""), "n": n, "hit@1": h1 / n, "hit@3": h3 / n,
                   "hit@5": h5 / n,
                   "mrr": mrr, "avg_ms": round(sum(r["cost_ms"] for r in rows) / n),
                   "chunk_topk": a.chunk_topk}
        out = {"summary": summary, "detail": rows}
        p = os.path.join(a.outdir, "lightrag_%s%s.json" % (mode, "_opt" if a.opt else ""))
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("== LightRAG[%s]: Hit@1=%.1f%% Hit@3=%.1f%% Hit@5=%.1f%% MRR=%.4f avg=%dms (%.1f min)" %
              (mode, 100 * summary["hit@1"], 100 * summary["hit@3"], 100 * summary["hit@5"],
               summary["mrr"], summary["avg_ms"], (time.time() - t0) / 60))
        print("saved ->", p)
    await rag.finalize_storages()
    print("LIGHTRAG_QUERY_DONE")


if __name__ == "__main__":
    asyncio.run(main())
