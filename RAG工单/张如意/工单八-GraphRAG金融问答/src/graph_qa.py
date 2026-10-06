# -*- coding: utf-8 -*-
"""
Graph RAG 问答：对 eval_question.md 的 10 个问题执行图谱检索 + 生成
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

工单要求：「问答系统基于 eval_question.md 中的 Question 进行检索，输出检索结果、
以及解析出来的知识图谱结构」——本脚本正是这条要求的实现：

    对每个问题：
      ① 局部检索 kg.local_search()  —— 实体链接 → 沿关系扩展 2 跳 → 子图（实体+关系）
      ② 全局检索 kg.global_search() —— 在社区摘要上做相关性排序（回答全局归纳题）
      ③ 可选向量兜底 retriever       —— 图谱信息不足时回退原文片段（与工单07 同通道，
                                        便于「比对 07 工单测试结果」时口径一致）
      ④ 融合生成 GraphRAG.answer(mode="hybrid")
      ⑤ 落盘：检索到的子图结构 + 答案 + 耗时

产物：
    results/graph_qa_results.json    逐题结构化结果（含子图、耗时、trace）
    results/graph_qa_results.md      人读版报告（演示视频与验收用）

运行：
    python 工单08-GraphRAG金融问答/src/graph_qa.py
    python .../graph_qa.py --mode hybrid --only Q01,Q03,Q04
    python .../graph_qa.py --no-retriever          # 纯图谱检索（不依赖向量库）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core.graph_rag import GraphRAG                     # noqa: E402

from build_graph import GRAPH_PATH                          # noqa: E402
from prepare_corpus import (                                # noqa: E402
    RESULTS_DIR, load_graph, load_questions,
)

JSON_PATH = RESULTS_DIR / "graph_qa_results.json"
MD_PATH = RESULTS_DIR / "graph_qa_results.md"


# ---------------------------------------------------------------------------
# 向量检索兜底（可选；索引不存在时自动降级为纯图谱检索）
# ---------------------------------------------------------------------------
def build_retriever(collection: str = "ccf_finance"):
    """
    尝试装载向量库 + BM25 索引。

    工单08 的图谱是从 CCF 年报构建的，因此这里默认使用同名 collection；
    若尚未为该语料建立索引（例如只跑了本工单的脚本），返回 None，
    GraphRAG 会自动只用图谱上下文作答——不会报错。
    """
    try:
        from rag_core.retriever import Retriever
        r = Retriever(collection)
        r.load_bm25()
        n = r.vs.count()
        if n == 0:
            print(f"[info] 向量库 {collection} 为空，本次仅用图谱检索")
            return None
        print(f"[info] 已装载向量兜底通道：{collection}（{n} 条向量）")
        return r
    except Exception as e:
        print(f"[info] 未启用向量兜底通道（{e}）——仅用图谱检索")
        return None


# ---------------------------------------------------------------------------
# 单题问答
# ---------------------------------------------------------------------------
def ask_one(rag: GraphRAG, kg, item: dict, top_k: int = 5) -> dict:
    """跑一个问题的完整 Graph RAG 链路，返回结构化结果。"""
    q = item["question"]
    t0 = time.perf_counter()
    res = rag.answer(q, top_k=top_k)                 # 含局部+全局（+向量）检索与生成
    elapsed = time.perf_counter() - t0

    # 单独再取一次子图结构，用于「输出解析出来的知识图谱结构」这一验收点
    local = kg.local_search(q)
    edges = [{
        "source": s, "target": t, "type": d.get("type", "相关"),
        "description": (d.get("description") or "")[:160],
    } for s, t, d in local["edges"][:60]]
    nodes = [{
        "name": n, "type": kg.entities[n].type,
        "description": (kg.entities[n].description or "")[:160],
        "degree": kg.entities[n].degree,
    } for n in sorted(local["nodes"],
                      key=lambda x: -kg.entities[x].degree)[:40] if n in kg.entities]

    glob = kg.global_search(q, top_k=3) if kg.community_summaries else {"communities": []}

    return {
        "id": item["id"],
        "question": q,
        "type": item.get("type", ""),
        "expected_points": item.get("ground_truth", ""),
        "answer": res["answer"],
        "latency_seconds": round(elapsed, 3),
        "trace": res.get("trace", {}),
        "subgraph": {
            "seeds": local["seeds"],
            "n_nodes": len(local["nodes"]),
            "n_edges": len(local["edges"]),
            "nodes": nodes,
            "edges": edges,
        },
        "communities": [
            {"id": cid, "summary": s[:400]} for cid, s in glob.get("communities", [])
        ],
        "n_contexts": len(res.get("contexts", [])),
        "contexts": [c[:1500] for c in res.get("contexts", [])],
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render_markdown(records: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# 工单08 Graph RAG 问答结果（eval_question.md 全 10 题）\n")
    lines.append(f"> 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答  \n"
                 f"> 检索模式：**{meta['mode']}**（局部子图 + 社区摘要全局检索"
                 f"{' + 向量兜底' if meta['retriever'] else ''}）  \n"
                 f"> 图谱规模：{meta['n_entities']} 实体 / {meta['n_relations']} 关系 / "
                 f"{meta['n_communities']} 社区  \n"
                 f"> 平均耗时：{meta['latency_avg']}s（图检索 + LLM 生成）  \n"
                 f"> 生成时间：{meta['generated_at']}\n")

    lines.append("## 总览\n")
    lines.append("| 编号 | 问题（截断） | 类型 | 种子实体数 | 子图规模 | 社区数 | 耗时(s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        sg = r["subgraph"]
        lines.append("| {} | {} | {} | {} | {}节点/{}关系 | {} | {} |".format(
            r["id"], (r["question"][:34] + "…") if len(r["question"]) > 34 else r["question"],
            r["type"], len(sg["seeds"]), sg["n_nodes"], sg["n_edges"],
            len(r["communities"]), r["latency_seconds"]))
    lines.append("")

    for r in records:
        sg = r["subgraph"]
        lines.append("---\n")
        lines.append(f"## {r['id']}　{r['question']}\n")
        lines.append(f"**类型**：{r['type']}　|　**耗时**：{r['latency_seconds']}s　|　"
                     f"**种子实体**：{'、'.join(sg['seeds']) or '（未命中实体，走全局检索）'}\n")
        lines.append("### 检索到的知识图谱子图\n")
        lines.append(f"- 子图规模：{sg['n_nodes']} 个实体、{sg['n_edges']} 条关系"
                     f"（局部检索 2 跳）")
        if r["communities"]:
            lines.append(f"- 全局检索命中社区："
                         f"{'、'.join('社区' + str(c['id']) for c in r['communities'])}")
        lines.append("")
        if sg["nodes"]:
            lines.append("| 实体 | 类型 | 度数 | 属性（图谱中累积的描述） |")
            lines.append("| --- | --- | --- | --- |")
            for n in sg["nodes"][:15]:
                lines.append(f"| {n['name']} | {n['type']} | {n['degree']} | "
                             f"{(n['description'] or '—')[:70]} |")
            lines.append("")
        if sg["edges"]:
            lines.append("<details><summary>展开关系列表（前 20 条）</summary>\n")
            lines.append("| 头实体 | 关系 | 尾实体 | 原文依据 |")
            lines.append("| --- | --- | --- | --- |")
            for e in sg["edges"][:20]:
                lines.append(f"| {e['source']} | {e['type']} | {e['target']} | "
                             f"{(e['description'] or '—')[:60]} |")
            lines.append("\n</details>\n")
        lines.append("### Graph RAG 生成的答案\n")
        lines.append("> " + (r["answer"] or "（空）").replace("\n", "\n> ") + "\n")
        if r["communities"]:
            lines.append("<details><summary>展开命中的社区摘要（全局检索上下文）</summary>\n")
            for c in r["communities"]:
                lines.append(f"**社区 {c['id']}**：{c['summary']}\n")
            lines.append("</details>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="工单08 Graph RAG 问答（eval_question.md）")
    ap.add_argument("--graph", type=Path, default=GRAPH_PATH, help="kg.json 路径")
    ap.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid"],
                    help="检索模式，默认 hybrid（局部子图 + 社区摘要）")
    ap.add_argument("--top-k", type=int, default=5, help="向量兜底/全局检索条数")
    ap.add_argument("--collection", default="ccf_finance", help="向量库 collection 名")
    ap.add_argument("--no-retriever", action="store_true", help="禁用向量兜底通道")
    ap.add_argument("--only", default=None, help="只跑指定题号，逗号分隔，如 Q01,Q04")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    args = ap.parse_args()

    questions = load_questions()
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        questions = [q for q in questions if q["id"].upper() in want]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        print("没有匹配的问题，请检查 --only 参数")
        return

    kg = load_graph(args.graph)
    print(f"载入图谱：{len(kg.entities)} 实体 / "
          f"{len(getattr(kg, 'relations_merged', []))} 关系 / {len(kg.communities)} 社区")

    # 社区摘要缺失时补一次（全局检索的前提；结果缓存在内存，不会每题重复调用）
    if not getattr(kg, "community_summaries", None) and args.mode in ("global", "hybrid"):
        print("图谱中缺少社区摘要，先补生成…")
        if not kg.communities:
            kg.detect_communities()
        kg.summarize_communities(top_n=20)

    retriever = None if args.no_retriever else build_retriever(args.collection)
    rag = GraphRAG(kg, retriever=retriever, mode=args.mode)

    records: list[dict] = []
    t_all = time.perf_counter()
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['id']} {item['question'][:38]}…")
        rec = ask_one(rag, kg, item, top_k=args.top_k)
        records.append(rec)
        print(f"      子图 {rec['subgraph']['n_nodes']} 节点 / "
              f"{rec['subgraph']['n_edges']} 关系，"
              f"命中社区 {len(rec['communities'])}，耗时 {rec['latency_seconds']}s")

    latencies = [r["latency_seconds"] for r in records]
    meta = {
        "work_order": "人工智能NLP-RAG-基于Graph RAG 实现金融问答",
        "graph": str(args.graph),
        "mode": args.mode,
        "profile": "optimized",
        "retriever": args.collection if retriever is not None else None,
        "n_entities": len(kg.entities),
        "n_relations": len(getattr(kg, "relations_merged", [])),
        "n_communities": len(kg.communities),
        "n_questions": len(records),
        "latency_avg": round(sum(latencies) / max(len(latencies), 1), 3),
        "latency_max": round(max(latencies), 3) if latencies else 0,
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps({"meta": meta, "results": records},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    MD_PATH.write_text(render_markdown(records, meta), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"问答完成：{len(records)} 题，平均耗时 {meta['latency_avg']}s")
    print(f"  结构化结果：{JSON_PATH}")
    print(f"  人读报告：  {MD_PATH}")
    print("  下一步：python 工单08-GraphRAG金融问答/src/compare_with_wo07.py")


if __name__ == "__main__":
    main()
