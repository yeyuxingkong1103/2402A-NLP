# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
脚本：RAGAS 评估（RAG 检索问答 vs 纯 LLM）

工单"演示/验收标准"要求：
  1、覆盖给定的 10 道必测问题；
  2、把"基于 PDF 的返回结果"与"只使用 LLM 返回的答案"做对比；
  3、选择 RAG 评估体系（RAGAS）进行评估并返回评估结果；
  4、指标包含 准确性 / 忠实度 / 上下文召回率 / 响应时间。

本脚本产出四类指标：

  A. 客观指标（不依赖任何裁判模型，最可信）
     · 关键事实命中率：标准答案里的关键数字/术语是否出现在回答中
     · 拒答率：回答是否承认"无法确定"
     · 响应时间：端到端耗时、首字耗时（工单要求 ≤3 秒）

  B. RAGAS 指标（仅 RAG 侧，因为纯 LLM 没有检索上下文）
     · faithfulness        忠实度：答案是否都能被检索上下文支撑
     · answer_relevancy    答案相关性
     · context_precision   上下文精确率
     · context_recall      上下文召回率（需要标准答案）

  C. LLM 裁判准确性（0~1）：把回答与标准答案一起交给裁判模型打分

  D. 对比结论：把上述结果汇总成一张表 + 一段结论

用法：
    python evaluate.py                  # 完整评估（RAG + 纯 LLM + RAGAS）
    python evaluate.py --no-ragas       # 跳过 RAGAS（省时）
    python evaluate.py --limit 3        # 只跑前 3 题（调试）
    python evaluate.py --reuse          # 复用 data/eval/answers.json，不重新问答
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import argparse
import json
import os
import re
import statistics
import sys
import time

from src import config, embedder, llm, rag, vector_store

ANSWERS_FILE = os.path.join(config.EVAL_DIR, "answers.json")
RESULTS_FILE = os.path.join(config.EVAL_DIR, "eval_results.json")
REPORT_FILE = os.path.join(config.EVAL_DIR, "评估报告.md")
GT_FILE = os.path.join(config.EVAL_DIR, "ground_truth.json")


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72, flush=True)


def _norm(text: str) -> str:
    """比对用归一化：去掉空白、千分位逗号，统一全角括号。"""
    t = re.sub(r"\s+", "", text or "")
    t = t.replace(",", "").replace("，", "")
    return t.replace("（", "(").replace("）", ")")


# ---------------------------------------------------------------------------
# 第一阶段：跑问答，收集答案 / 上下文 / 耗时
# ---------------------------------------------------------------------------
def collect(questions: list[dict], reuse: bool = False) -> dict:
    if reuse and os.path.isfile(ANSWERS_FILE):
        print(f"复用已有问答结果：{ANSWERS_FILE}")
        with open(ANSWERS_FILE, encoding="utf-8") as fh:
            return json.load(fh)

    rag.warmup()  # 先把模型加载与 BM25 构建的冷启动成本付掉，避免污染第一条的耗时

    records = []
    for i, q in enumerate(questions, start=1):
        qid, qtext = q["id"], q["question"]
        print(f"\n[{i}/{len(questions)}] {qid} {qtext}", flush=True)

        t0 = time.time()
        r = rag.ask(qtext)
        rag_wall = time.time() - t0
        print(f"  RAG  {r.timings['total_s']}s -> {r.answer[:60]!r}")

        pure = rag.ask_pure_llm(qtext)
        print(f"  纯LLM {pure.timings['total_s']}s -> {pure.answer[:60]!r}")

        records.append({
            "id": qid, "question": qtext,
            "rag": {
                "answer": r.answer,
                "contexts": [(c.get("raw_text") or c.get("text", "")) for c in r.contexts],
                "context_pages": [int(c.get("page_idx", 0)) + 1 for c in r.contexts],
                "citations": r.citations,
                "timings": r.timings,
                "wall_s": round(rag_wall, 3),
                "analysis": r.analysis,
            },
            "llm": {
                "answer": pure.answer,
                "timings": pure.timings,
            },
        })

    os.makedirs(config.EVAL_DIR, exist_ok=True)
    with open(ANSWERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)
    print(f"\n问答结果已保存 -> {ANSWERS_FILE}")
    return records


# ---------------------------------------------------------------------------
# 第二阶段 A：客观指标
# ---------------------------------------------------------------------------
def objective_metrics(records: list[dict], gt: dict[int, dict]) -> dict:
    rows = []
    for rec in records:
        g = gt.get(rec["id"], {})
        facts = g.get("key_facts", [])
        row = {"id": rec["id"], "question": rec["question"]}
        for side in ("rag", "llm"):
            ans = _norm(rec[side]["answer"])
            hit = sum(1 for f in facts if _norm(f) in ans)
            row[f"{side}_key_hit"] = round(hit / len(facts), 3) if facts else None
            row[f"{side}_refusal"] = bool(re.search(
                r"无法确定|无法回答|未找到|没有找到|未提及|无法提供|不知道", ans))
            row[f"{side}_chars"] = len(rec[side]["answer"])
            t = rec[side]["timings"]
            row[f"{side}_total_s"] = t.get("total_s")
            row[f"{side}_ttft_s"] = t.get("ttft_s")
        rows.append(row)

    def _avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    summary = {}
    for side in ("rag", "llm"):
        summary[side] = {
            "关键事实命中率": _avg(f"{side}_key_hit"),
            "拒答率": round(statistics.mean([1 if r[f"{side}_refusal"] else 0
                                            for r in rows]), 3) if rows else None,
            "平均响应时间_s": _avg(f"{side}_total_s"),
            "响应时间中位数_s": round(statistics.median(
                [r[f"{side}_total_s"] for r in rows if r.get(f"{side}_total_s")]), 3),
            "≤3秒占比": round(statistics.mean(
                [1 if (r.get(f"{side}_total_s") or 99) <= 3 else 0 for r in rows]), 3),
            "平均答案长度": round(statistics.mean(
                [r[f"{side}_chars"] for r in rows]), 1),
        }
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# 第三阶段 B：RAGAS
# ---------------------------------------------------------------------------
def _bge_embeddings():
    """把本地 bge-m3 包装成 RAGAS 需要的 LangChain Embeddings（全离线）。

    【踩坑记录】不能用 `class X(Embeddings, _LocalBGE)` 这种写法：
    MRO 里 Embeddings 排在前面，它的抽象方法先被找到，实例化时会报
    "Can't instantiate abstract class ... without an implementation"。
    直接继承 Embeddings 并实现两个方法即可。
    """
    from langchain_core.embeddings import Embeddings

    class _BgeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embedder.embed_texts(list(texts))

        def embed_query(self, text):
            return embedder.embed_query(text)

    return _BgeEmbeddings()


def ragas_metrics(records: list[dict], gt: dict[int, dict]) -> dict:
    """跑 RAGAS。任何失败都返回 {"error": ...}，不阻断整份评估。"""
    try:
        from langchain_ollama import ChatOllama
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)

        judge = ChatOllama(model=llm.resolve_model(), base_url=config.OLLAMA_HOST,
                           temperature=0.0, num_ctx=config.OLLAMA_NUM_CTX,
                           num_predict=768, keep_alive="30m")

        ragas_llm = LangchainLLMWrapper(judge)
        ragas_emb = LangchainEmbeddingsWrapper(_bge_embeddings())

        # 降低 answer_relevancy 的采样次数：3B 裁判模型下 3 次采样太慢，收益也有限
        answer_relevancy.strictness = 1

        samples = []
        for rec in records:
            g = gt.get(rec["id"], {})
            samples.append(SingleTurnSample(
                user_input=rec["question"],
                response=rec["rag"]["answer"],
                retrieved_contexts=rec["rag"]["contexts"] or [""],
                reference=g.get("reference", ""),
            ))
        dataset = EvaluationDataset(samples=samples)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm, embeddings=ragas_emb, raise_exceptions=False,
        )
        df = result.to_pandas()
        per_q = []
        for rec, (_, row) in zip(records, df.iterrows()):
            per_q.append({
                "id": rec["id"],
                "faithfulness": _f(row.get("faithfulness")),
                "answer_relevancy": _f(row.get("answer_relevancy")),
                "context_precision": _f(row.get("context_precision")),
                "context_recall": _f(row.get("context_recall")),
            })
        summary = {}
        for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            vals = [r[k] for r in per_q if r[k] is not None]
            summary[k] = round(statistics.mean(vals), 3) if vals else None
            summary[k + "_有效样本"] = len(vals)
        return {"summary": summary, "per_question": per_q}
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


def _f(v):
    try:
        f = float(v)
        return None if f != f else round(f, 3)  # NaN -> None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 第二阶段 C：LLM 裁判准确性
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """你是严格的评分员。请对比【参考答案】与【模型回答】，给出 0、0.5 或 1 的分数：
1  = 关键事实完全正确（数字、比例、名称全部一致）
0.5 = 部分正确（方向对但有遗漏或个别数字错误）
0 = 错误、答非所问，或编造了参考答案中不存在的事实

只输出一个 JSON：{{"score": <0|0.5|1>, "reason": "<20字以内理由>"}}

【问题】{q}
【参考答案】{ref}
【模型回答】{ans}"""


def judge_accuracy(records: list[dict], gt: dict[int, dict]) -> dict:
    """用同一台本机模型做裁判，分别给 RAG 与纯 LLM 的回答打分。"""
    out = {"rag": [], "llm": []}
    for rec in records:
        g = gt.get(rec["id"], {})
        for side in ("rag", "llm"):
            prompt = JUDGE_PROMPT.format(q=rec["question"], ref=g.get("reference", ""),
                                         ans=rec[side]["answer"][:1500])
            try:
                obj = rag.llm_json("你只输出 JSON。", prompt, num_predict=120)
                score = float(obj.get("score", 0))
            except Exception:
                score = None
            out[side].append({"id": rec["id"], "score": score})
    summary = {}
    for side in ("rag", "llm"):
        vals = [r["score"] for r in out[side] if r["score"] is not None]
        summary[side] = {
            "平均准确性": round(statistics.mean(vals), 3) if vals else None,
            "有效样本": len(vals),
            "满分题数": sum(1 for v in vals if v >= 0.99),
        }
    out["summary"] = summary
    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(obj: dict, ras: dict, jud: dict, records: list[dict],
                 gt: dict[int, dict]) -> None:
    s = obj["summary"]
    L: list[str] = []
    L.append("# RAG 问答系统评估报告\n")
    L.append(f"- 工单编号：{config.WORKORDER_ID}")
    L.append(f"- 知识库：{config.SOURCE_NAME}（{vector_store.count()} 个向量片段）")
    L.append(f"- Embedding：bge-m3（{config.EMBEDDING_MODEL_PATH}）")
    L.append(f"- 生成模型：{llm.resolve_model()}（Ollama 本机）")
    L.append(f"- 题目数量：{len(records)}（工单给定的必测问题）\n")

    L.append("## 一、总体对比（RAG vs 纯 LLM）\n")
    L.append("| 指标 | RAG 检索问答 | 纯 LLM（无检索） |")
    L.append("| --- | --- | --- |")
    L.append(f"| 关键事实命中率 | {s['rag']['关键事实命中率']} | {s['llm']['关键事实命中率']} |")
    L.append(f"| LLM 裁判准确性（0~1） | {jud['summary']['rag']['平均准确性']} | "
             f"{jud['summary']['llm']['平均准确性']} |")
    L.append(f"| 裁判满分题数 | {jud['summary']['rag']['满分题数']}/{len(records)} | "
             f"{jud['summary']['llm']['满分题数']}/{len(records)} |")
    L.append(f"| 平均响应时间（秒） | {s['rag']['平均响应时间_s']} | {s['llm']['平均响应时间_s']} |")
    L.append(f"| 响应时间中位数（秒） | {s['rag']['响应时间中位数_s']} | {s['llm']['响应时间中位数_s']} |")
    L.append(f"| ≤3 秒占比 | {s['rag']['≤3秒占比']} | {s['llm']['≤3秒占比']} |")
    L.append(f"| 拒答率 | {s['rag']['拒答率']} | {s['llm']['拒答率']} |")
    L.append(f"| 平均答案长度（字） | {s['rag']['平均答案长度']} | {s['llm']['平均答案长度']} |\n")

    L.append("## 二、RAGAS 指标（仅 RAG 侧）\n")
    if ras.get("skipped"):
        L.append("> 本次评估跳过了 RAGAS（使用 `--no-ragas` 运行），"
                 "如需该项指标请重新执行 `python evaluate.py`。\n")
    elif "error" in ras:
        L.append(f"> RAGAS 执行失败：`{ras['error']}`\n")
    else:
        rs = ras["summary"]
        L.append("| 指标 | 均值 | 有效样本 |")
        L.append("| --- | --- | --- |")
        for k, label in (("faithfulness", "忠实度"), ("answer_relevancy", "答案相关性"),
                         ("context_precision", "上下文精确率"), ("context_recall", "上下文召回率")):
            L.append(f"| {label}（{k}） | {rs.get(k)} | {rs.get(k + '_有效样本')} |")
        L.append("")

    L.append("## 三、逐题明细\n")
    L.append("| ID | 关键事实命中(RAG/LLM) | 准确性(RAG/LLM) | RAG耗时(s) | LLM耗时(s) | 引用数(核验通过) |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    jr = {r["id"]: r["score"] for r in jud["rag"]}
    jl = {r["id"]: r["score"] for r in jud["llm"]}
    for r in obj["rows"]:
        rec = next(x for x in records if x["id"] == r["id"])
        cits = rec["rag"]["citations"]
        good = sum(1 for c in cits if c.get("verified"))
        L.append(f"| {r['id']} | {r['rag_key_hit']} / {r['llm_key_hit']} | "
                 f"{jr.get(r['id'])} / {jl.get(r['id'])} | "
                 f"{r['rag_total_s']} | {r['llm_total_s']} | {len(cits)}({good}) |")
    L.append("")

    L.append("## 四、逐题问答原文\n")
    for rec in records:
        g = gt.get(rec["id"], {})
        L.append(f"### [{rec['id']}] {rec['question']}\n")
        L.append(f"**标准答案**：{g.get('reference', '（未提供）')}\n")
        L.append(f"**RAG 回答**：{rec['rag']['answer']}\n")
        pages = "、".join(f"[{c['index']}]第{c['page']}页"
                          for c in rec["rag"]["citations"]) or "无"
        L.append(f"*引用出处*：{pages}\n")
        L.append(f"**纯 LLM 回答**：{rec['llm']['answer']}\n")
        L.append("---\n")

    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"评估报告 -> {REPORT_FILE}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="RAGAS 评估（RAG vs 纯 LLM）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse", action="store_true", help="复用 answers.json")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    with open(config.EVAL_QUESTIONS_FILE, encoding="utf-8") as fh:
        questions = json.load(fh)
    if args.limit:
        questions = questions[:args.limit]
    with open(GT_FILE, encoding="utf-8") as fh:
        gt = {g["id"]: g for g in json.load(fh)}

    _banner(f"第 1 步 / 4：跑问答（{len(questions)} 题，RAG + 纯 LLM）")
    records = collect(questions, reuse=args.reuse)

    _banner("第 2 步 / 4：客观指标（关键事实命中率 / 拒答率 / 响应时间）")
    obj = objective_metrics(records, gt)
    for side, label in (("rag", "RAG"), ("llm", "纯 LLM")):
        s = obj["summary"][side]
        print(f"  {label:5s} 命中率={s['关键事实命中率']} 拒答率={s['拒答率']} "
              f"平均耗时={s['平均响应时间_s']}s ≤3s占比={s['≤3秒占比']}")

    _banner("第 3 步 / 4：RAGAS 指标（忠实度/相关性/上下文精确率/召回率）")
    if args.no_ragas:
        ras = {"skipped": True}
        print("  已跳过（--no-ragas）")
    else:
        ras = ragas_metrics(records, gt)
        print("  " + json.dumps(ras.get("summary", ras), ensure_ascii=False))

    _banner("第 4 步 / 4：LLM 裁判准确性")
    if args.no_judge:
        jud = {"summary": {"rag": {"平均准确性": None, "满分题数": 0, "有效样本": 0},
                           "llm": {"平均准确性": None, "满分题数": 0, "有效样本": 0}},
               "rag": [], "llm": []}
        print("  已跳过（--no-judge）")
    else:
        jud = judge_accuracy(records, gt)
        print(f"  RAG 准确性={jud['summary']['rag']['平均准确性']} "
              f"纯LLM 准确性={jud['summary']['llm']['平均准确性']}")

    result = {"workorder": config.WORKORDER_ID, "n_questions": len(records),
              "objective": obj, "ragas": ras, "judge": jud,
              "config": {"embedding": config.EMBEDDING_MODEL_PATH,
                         "llm": config.OLLAMA_MODEL,
                         "collection": config.COLLECTION_NAME,
                         "points": vector_store.count(),
                         "vector_db": config.QDRANT_MODE}}
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"结果 JSON -> {RESULTS_FILE}")

    write_report(obj, ras, jud, records, gt)
    _banner("评估完成")
    return 0


if __name__ == "__main__":
    sys.exit(bootstrap.run_with_large_stack(main))
