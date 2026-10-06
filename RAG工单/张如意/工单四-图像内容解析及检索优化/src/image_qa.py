# -*- coding: utf-8 -*-
"""
图像问题检索问答（id 5 / id 6 演示与自检）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

对两个「答案只在图里」的问题做完整问答，并**显式展示四件事**：
    1. 命中的图像 —— 图像文件路径 + 所在页码 + 来源（内嵌位图/矢量图整页渲染）
    2. 图像的语义描述 —— 多模态模型读出的层级树 / 结构化数据
    3. 生成的答案 —— RAG 严格依据检索上下文生成
    4. 检索精确度 —— Context Precision（RAGAS 口径）+ 要点命中率 + 图像页命中

两个问题的定位（详见 docs/图像问题定位分析.md）：
    id 5  《招股说明书2》第 39 页「公司组织结构图」——矢量绘制，文本层只有
          逐字竖排的节点名，层级关系完全丢失；
    id 6  《招股说明书2》第 72 页「2008 年中国 IC 市场应用结构与增长」——
          饼图（应用结构）+ 条形图（增长率），数值只在图里。

用法：
    python src/image_qa.py                     # 两个问题都跑
    python src/image_qa.py --qids 5            # 只跑 id 5
    python src/image_qa.py --show-chunks 8     # 展示更多检索片段
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_extractor import RESULTS_DIR, WORKDIR  # noqa: E402
from run_evaluation import GROUND_TRUTH, judge_keywords, safe_generate  # noqa: E402
from rag_core import config, evaluate  # noqa: E402

QA_JSON = RESULTS_DIR / "image_qa.json"
QA_MD = RESULTS_DIR / "image_qa.md"

IMAGE_QUESTIONS = config.QUESTIONS_IMAGE          # id 5 / id 6

# 评测元数据：每个问题的目标图页（用于「图像页命中」判定）
EXPECTED_PAGES: dict[int, set[int]] = {5: {39}, 6: {72, 310}}


# ---------------------------------------------------------------------------
# 图像语义描述查询
# ---------------------------------------------------------------------------
def _load_descriptions() -> dict:
    p = RESULTS_DIR / "image_descriptions.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"images": []}


def find_image_meta(doc: str, page: int) -> list[dict]:
    """从解析结果里取回该页图像的元信息（文件路径/描述/结构化数据）。"""
    out = []
    for r in _load_descriptions().get("images", []):
        if r.get("doc") == doc and int(r.get("page", -1)) == page:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# 单问回答
# ---------------------------------------------------------------------------
def answer_image_question(q: dict, retriever, top_k: int = 5,
                          show_chunks: int = 5) -> dict:
    """对一个图像问题做检索 + 生成 + 精度评估，返回完整痕迹。"""
    question = q["question"]
    qid = q["id"]
    gt = GROUND_TRUTH.get(qid, "")

    # ---- 1. 检索 ----
    t0 = time.perf_counter()
    res = retriever.retrieve(
        question, strategy="hybrid", top_k=top_k,
        recall_k=config.TOP_K_RECALL, reranker="cascade",
        fusion="rrf", alpha=config.HYBRID_ALPHA,
    )
    retrieve_s = time.perf_counter() - t0
    docs = res.docs
    contexts = [d.get("text", "") for d in docs]

    # ---- 2. 命中的图像 ----
    hit_images = []
    for d in docs:
        if d.get("type") != "image":
            continue
        metas = find_image_meta(str(d.get("doc", "")), int(d.get("page", 0)))
        hit_images.append({
            "doc": d.get("doc"), "page": d.get("page"),
            "score": round(float(d.get("final_score", d.get("score", 0) or 0)), 4),
            "chunk_id": d.get("chunk_id"),
            "image_files": [m["file"] for m in metas],
            "image_source": [m.get("source") for m in metas],
            "semantic_description": [m.get("vlm_description", "")
                                     for m in metas] or [(d.get("text") or "")[:500]],
            "chart_data": [m.get("chart_data", {}) for m in metas],
            "hierarchy_facts": [f for m in metas
                                for f in (m.get("hierarchy_facts") or [])],
            "text_snippet": (d.get("text") or "")[:400],
        })

    # ---- 3. 生成（模型不可用时给出占位答案，不中断演示）----
    t_gen = time.perf_counter()
    ctx = retriever.format_context(docs)
    gen = safe_generate(question, docs, ctx)
    total_s = retrieve_s + (time.perf_counter() - t_gen)

    # ---- 4. 检索精确度 ----
    try:
        ctx_precision = evaluate.context_precision(question, contexts, gt)
    except Exception as e:
        print(f"  [warn] Context Precision 计算失败：{e}")
        ctx_precision = float("nan")
    kw = judge_keywords(gen.answer, qid)
    pages = {int(d.get("page", 0)) for d in docs}
    expected = EXPECTED_PAGES.get(qid, set())
    page_hit = bool(pages & expected) if expected else None

    return {
        "id": qid,
        "question": question,
        "ground_truth": gt,
        "hit_images": hit_images,
        "answer": gen.answer,
        "citations": gen.citations,
        "precision": {
            "context_precision": None if ctx_precision != ctx_precision
            else round(ctx_precision, 4),
            "要点命中率": round(len(kw["命中"]) / max(kw.get("要点数", 1), 1), 4),
            "命中要点": kw["命中"],
            "漏答要点": kw["漏答"],
            "要点判定": kw["正确"],
            "图像页命中": page_hit,
            "期望图页": sorted(expected),
        },
        "retrieved_pages": sorted(pages),
        "retrieved_chunks": [
            {"rank": i, "chunk_id": d.get("chunk_id"), "doc": d.get("doc"),
             "page": d.get("page"), "type": d.get("type"),
             "score": round(float(d.get("final_score", d.get("score", 0) or 0)), 4),
             "snippet": (d.get("text") or "")[:200]}
            for i, d in enumerate(docs[:show_chunks], 1)
        ],
        "timings": {
            "retrieve_s": round(retrieve_s, 3),
            "total_s": round(total_s, 3),
            "detail": {k: round(v, 4) for k, v in res.timings.items()},
        },
    }


# ---------------------------------------------------------------------------
# 展示与落盘
# ---------------------------------------------------------------------------
def print_result(r: dict) -> None:
    """终端演示：命中的图像 → 语义描述 → 答案 → 检索精确度。"""
    line = "=" * 78
    print(f"\n{line}\n【问题 id {r['id']}】{r['question']}\n{line}")

    print(f"\n▌1. 命中的图像（{len(r['hit_images'])} 个图像块）")
    if not r["hit_images"]:
        print("   （无 —— 检索未命中任何图像块，说明图像未入库或排序靠后）")
    for im in r["hit_images"]:
        print(f"   · 《{im['doc']}》第{im['page']}页  相关度={im['score']}  "
              f"来源={im['image_source']}")
        for f in im["image_files"]:
            print(f"     图像文件：{f}")
            p = WORKDIR / f
            print(f"     绝对路径：{p.resolve()}"
                  f"{'' if p.exists() else '  [文件缺失]'}")
        if im["hierarchy_facts"]:
            print("     层级事实：")
            for x in im["hierarchy_facts"][:6]:
                print(f"       - {x}")

    print("\n▌2. 图像的语义描述（多模态模型输出，节选）")
    for im in r["hit_images"]:
        for d in im["semantic_description"]:
            print("   " + (d or "（空）").replace("\n", "\n   ")[:1200])
        for cd in im["chart_data"]:
            if cd:
                print("   结构化数据：" + json.dumps(cd, ensure_ascii=False)[:600])

    print("\n▌3. 生成的答案")
    print("   " + (r["answer"] or "（空）").replace("\n", "\n   "))

    pr = r["precision"]
    print("\n▌4. 检索精确度")
    print(f"   Context Precision（RAGAS 口径）：{pr['context_precision']}")
    print(f"   要点命中率：{pr['要点命中率'] * 100:.0f}%"
          f"（命中 {len(pr['命中要点'])} / 漏答 {len(pr['漏答要点'])}）")
    if pr["漏答要点"]:
        print(f"   漏答要点：{'、'.join(pr['漏答要点'])}")
    print(f"   图像页命中：{pr['图像页命中']}（期望图页 {pr['期望图页']}，"
          f"实际检索页码 {r['retrieved_pages']}）")
    print(f"   耗时：检索 {r['timings']['retrieve_s']}s / "
          f"端到端 {r['timings']['total_s']}s")

    print("\n▌附：Top 检索片段")
    for c in r["retrieved_chunks"]:
        print(f"   [{c['rank']}] 《{c['doc']}》第{c['page']}页 [{c['type']}] "
              f"score={c['score']}  {c['snippet'][:80]}…")


def save_results(results: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = sum(1 for r in results if r["precision"]["要点判定"])
    payload = {
        "workorder": "人工智能NLP-RAG-图像内容解析及检索优化",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_questions": len(results),
        "n_correct": ok,
        "accuracy": round(ok / max(len(results), 1), 4),
        "results": results,
    }
    QA_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    md = [
        "# 图像问题检索问答结果（id 5 / id 6）",
        "",
        "工单编号：人工智能NLP-RAG-图像内容解析及检索优化",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 答对：{ok}/{len(results)}",
        "",
    ]
    for r in results:
        pr = r["precision"]
        md += [
            f"## id {r['id']}：{r['question']}", "",
            f"**参考答案**：{r['ground_truth']}", "",
            f"**生成的答案**：", "", "```text", r["answer"], "```", "",
            f"**检索精确度**：Context Precision = {pr['context_precision']}；"
            f"要点命中率 = {pr['要点命中率'] * 100:.0f}%；"
            f"图像页命中 = {pr['图像页命中']}",
            "",
            "**命中的图像**", "",
        ]
        for im in r["hit_images"]:
            md += [f"- 《{im['doc']}》第{im['page']}页（相关度 {im['score']}，"
                   f"来源 {im['image_source']}）",
                   f"  - 图像文件：`{im['image_files']}`"]
            for x in im["hierarchy_facts"][:8]:
                md.append(f"  - 层级事实：{x}")
        md += ["", "**图像语义描述**", ""]
        for im in r["hit_images"]:
            for d in im["semantic_description"]:
                md += ["```text", (d or "")[:1500], "```"]
        if r["hit_images"] and any(im["chart_data"] for im in r["hit_images"]):
            md += ["**图表结构化数据**", ""]
            for im in r["hit_images"]:
                for cd in im["chart_data"]:
                    if cd:
                        md += ["```json",
                               json.dumps(cd, ensure_ascii=False, indent=2)[:1500],
                               "```"]
        md += ["**Top 检索片段**", "",
               "| # | 来源 | 页 | 类型 | 分数 | 片段 |",
               "|---|------|----|------|------|------|"]
        for c in r["retrieved_chunks"]:
            md.append(f"| {c['rank']} | {c['doc']} | {c['page']} | {c['type']} | "
                      f"{c['score']} | {c['snippet'][:60].replace('|', '／')} |")
        md += ["", "---", ""]
    QA_MD.write_text("\n".join(md), encoding="utf-8")
    return QA_JSON, QA_MD


def main() -> None:
    ap = argparse.ArgumentParser(description="图像问题（id5/id6）检索问答")
    ap.add_argument("--collection", default="wo04_multimodal")
    ap.add_argument("--qids", default="5,6", help="问题 id，逗号分隔")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--show-chunks", type=int, default=5)
    args = ap.parse_args()

    from build_multimodal_index import load_retriever

    qids = [int(x) for x in args.qids.split(",") if x.strip()]
    questions = [q for q in IMAGE_QUESTIONS if q["id"] in qids]
    if not questions:
        raise SystemExit(f"未找到问题 {qids}，可选 {[q['id'] for q in IMAGE_QUESTIONS]}")

    print(f"[检索器] 装载索引 {args.collection} …")
    retriever = load_retriever(args.collection)
    print(f"         向量库 {retriever.vs.count()} 条 / "
          f"BM25 {len(retriever.load_bm25().doc_ids)} 篇")

    results = []
    for q in questions:
        r = answer_image_question(q, retriever, top_k=args.top_k,
                                  show_chunks=args.show_chunks)
        results.append(r)
        print_result(r)

    jp, mp = save_results(results)
    print(f"\n[完成] 结果已保存：\n  {jp}\n  {mp}")


if __name__ == "__main__":
    main()
