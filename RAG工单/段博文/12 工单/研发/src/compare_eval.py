# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
检索对比 + RAGAS 评估模块（验收标准 2 核心）。

流程：
    1. 对工单指定的 16 个问题，分别用两条链路检索：
       - 传统 RAG：bge-m3 扁平向量检索 Top-K（rag_baseline）；
       - LightRAG：混合模式（hybrid）双层检索——局部（实体向量）+全局
         （关系/主题向量）+ 图扩展 + 源文本回溯（only_need_context=True）。
    2. 用同一个 DeepSeek 生成器基于各自检索到的上下文生成答案（temperature=0）。
    3. RAGAS 简化评估（4 指标，以参考答案关键词为基准）：
       - context_precision：检索上下文中相关内容的占比（按排名加权）；
       - context_recall：参考答案关键词被上下文覆盖的比例；
       - faithfulness：答案中的事实可归因于上下文的比例；
       - answer_relevancy：答案对参考答案关键词的覆盖比例。
    4. 输出逐题对比表 + 汇总指标对比 + JSON 结果文件。
"""

import asyncio
import json
import time
from datetime import datetime

import httpx

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TOP_K, CHUNK_TOP_K, RESULT_DIR,
)
from questions import QUESTIONS
import rag_baseline
from lightrag_build import build_rag
from logger import get_logger

logger = get_logger(__name__)


# ---------------- 答案生成（两链路共用） ----------------

def generate_answer(question: str, contexts: list[str]) -> str:
    """基于检索上下文生成答案（DeepSeek，temperature=0）。"""
    context = "\n\n".join(
        f"[资料 {i}]\n{c}" for i, c in enumerate(contexts, 1)
    )
    prompt = f"""基于以下参考资料回答问题，答案要准确、完整，包含关键数据。

{context}

问题：{question}

若资料不足以回答，请明确说明"根据提供的资料无法回答"。

答案："""
    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "你是一个严谨的招股说明书问答助手，擅长从财务文档中提取准确信息。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------- LightRAG 检索封装 ----------------

async def lightrag_retrieve(rag, question: str) -> tuple[list[str], float]:
    """LightRAG hybrid 模式检索，返回 (上下文列表, 耗时)。"""
    from lightrag import QueryParam

    t0 = time.time()
    ctx = await rag.aquery(
        question,
        param=QueryParam(mode="hybrid", only_need_context=True,
                         top_k=TOP_K, chunk_top_k=CHUNK_TOP_K),
    )
    elapsed = time.time() - t0
    # LightRAG 返回的上下文为分段字符串，按段落切分为"上下文单元"
    contexts = [p.strip() for p in str(ctx).split("\n----\n") if p.strip()]
    if len(contexts) == 1 and len(contexts[0]) > 4000:
        contexts = [p.strip() for p in contexts[0].split("\n\n") if p.strip()]
    return contexts, elapsed


# ---------------- RAGAS 简化评估 ----------------

def _hit_count(text: str, keywords: list[str]) -> int:
    """统计关键词命中数（大小写/千分位容错）。"""
    t = text.replace(",", "").replace("，", "").replace(" ", "")
    hit = 0
    for k in keywords:
        kk = k.replace(",", "").replace("，", "").replace(" ", "")
        if kk in t:
            hit += 1
    return hit


def ragas_evaluate_one(keywords, contexts, answer) -> dict:
    """对单个问题计算 RAGAS 简化指标。"""
    ctx_all = " ".join(contexts)

    # Context Precision：相关内容在上下文中的排名加权占比
    prec = 0.0
    rel_ranks = [i for i, c in enumerate(contexts, 1)
                 if _hit_count(c, keywords) > 0]
    if rel_ranks:
        prec = sum(1.0 / r for r in rel_ranks) / sum(1.0 / r for r in range(1, len(contexts) + 1))

    # Context Recall：关键词被上下文覆盖比例
    recall = _hit_count(ctx_all, keywords) / len(keywords) if keywords else 0.0

    # Faithfulness：答案命中关键词中可归因（同词出现在上下文）的比例
    ans_hits = [k for k in keywords if k.replace(",", "").replace("，", "") in
                answer.replace(",", "").replace("，", "").replace(" ", "")]
    faithful = 0.0
    if ans_hits:
        faithful = sum(1 for k in ans_hits if _hit_count(ctx_all, [k]) > 0) / len(ans_hits)

    # Answer Relevancy：答案对参考关键词的覆盖比例
    relevancy = len(ans_hits) / len(keywords) if keywords else 0.0

    return {
        "context_precision": round(prec, 4),
        "context_recall": round(recall, 4),
        "faithfulness": round(faithful, 4),
        "answer_relevancy": round(relevancy, 4),
    }


# ---------------- 主流程 ----------------

async def run_compare(limit: int | None = None) -> dict:
    """对 16 题执行双链路检索对比 + RAGAS 评估。"""
    qs = QUESTIONS[:limit] if limit else QUESTIONS

    rag = build_rag()
    await rag.initialize_storages()
    try:
        # 预热传统 RAG 索引
        rag_baseline.build_index()
        results = []

        for q in qs:
            print(f"\n[Q{q['id']}] {q['question'][:46]}...")
            row = {"id": q["id"], "type": q["type"],
                   "question": q["question"],
                   "expected_answer": q["expected_answer"],
                   "keywords": q["keywords"]}

            # ---- 传统 RAG ----
            t0 = time.time()
            rag_hits = rag_baseline.search(q["question"], top_k=CHUNK_TOP_K)
            t_rag_search = time.time() - t0
            rag_ctxs = [h["content"] for h in rag_hits]
            t0 = time.time()
            rag_answer = generate_answer(q["question"], rag_ctxs[:5])
            t_rag_gen = time.time() - t0
            row["rag"] = {
                "contexts": rag_ctxs,
                "answer": rag_answer,
                "search_time": round(t_rag_search, 2),
                "gen_time": round(t_rag_gen, 2),
                **ragas_evaluate_one(q["keywords"], rag_ctxs, rag_answer),
            }

            # ---- LightRAG ----
            lr_ctxs, t_lr_search = await lightrag_retrieve(rag, q["question"])
            t0 = time.time()
            lr_answer = generate_answer(q["question"], lr_ctxs[:8])
            t_lr_gen = time.time() - t0
            row["lightrag"] = {
                "contexts": lr_ctxs,
                "answer": lr_answer,
                "search_time": round(t_lr_search, 2),
                "gen_time": round(t_lr_gen, 2),
                **ragas_evaluate_one(q["keywords"], lr_ctxs, lr_answer),
            }

            results.append(row)
            print(f"  传统RAG: P={row['rag']['context_precision']:.2f} "
                  f"R={row['rag']['context_recall']:.2f} A={row['rag']['answer_relevancy']:.2f} "
                  f"检索{t_rag_search:.1f}s | LightRAG: P={row['lightrag']['context_precision']:.2f} "
                  f"R={row['lightrag']['context_recall']:.2f} A={row['lightrag']['answer_relevancy']:.2f} "
                  f"检索{t_lr_search:.1f}s")
    finally:
        await rag.finalize_storages()

    # ---- 汇总 ----
    n = len(results)
    summary = {"n": n}
    for side in ("rag", "lightrag"):
        summary[side] = {
            m: round(sum(r[side][m] for r in results) / n, 4)
            for m in ("context_precision", "context_recall",
                      "faithfulness", "answer_relevancy")
        }
        summary[side]["overall"] = round(
            sum(summary[side][m] for m in
                ("context_precision", "context_recall", "faithfulness",
                 "answer_relevancy")) / 4, 4)
        summary[side]["avg_search_time"] = round(
            sum(r[side]["search_time"] for r in results) / n, 2)

    out = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": results,
    }
    (RESULT_DIR / "compare_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*64}\nRAGAS 指标对比（{n} 题）\n{'='*64}")
    print(f"{'指标':<22}{'传统RAG':>10}{'LightRAG':>10}")
    for m in ("context_precision", "context_recall", "faithfulness",
              "answer_relevancy", "overall"):
        print(f"{m:<22}{summary['rag'][m]:>10.4f}{summary['lightrag'][m]:>10.4f}")
    print(f"{'平均检索耗时(s)':<20}{summary['rag']['avg_search_time']:>10.2f}"
          f"{summary['lightrag']['avg_search_time']:>10.2f}")
    print("结果已保存 compare_results.json")
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG vs LightRAG 对比评估")
    parser.add_argument("--limit", type=int, default=None, help="只测前 N 题")
    args = parser.parse_args()
    asyncio.run(run_compare(limit=args.limit))


if __name__ == "__main__":
    main()
