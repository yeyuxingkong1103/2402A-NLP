"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。
检索指标与生成指标分开；无模型时不伪造 LLM 对比或 RAGAS 分数。
"""
import concurrent.futures
import json
import re
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "研发"))
from app.core import KnowledgeBase, dump_json, llm_configured


def norm(value):
    return re.sub(r"\s|,|，|[。；]", "", value).lower()


def run():
    gold = json.loads((BASE / "gold.json").read_text(encoding="utf-8"))
    kb = KnowledgeBase()
    # 预热仅用于服务已启动的稳态评估；所有样本绕过答案缓存。
    kb.retrieve("公司注册资本是多少？")
    strategies = {}
    for strategy in ["sparse", "dense", "hybrid"]:
        if strategy == "dense" and kb.dense is None:
            continue
        results = []
        for sample in gold:
            start = time.perf_counter()
            info, hits = kb.retrieve(sample["question"], strategy=strategy)
            positions = [i + 1 for i, h in enumerate(hits) if h["pdf_page"] in sample["pages"]]
            evidence = norm(" ".join(h["text"] for h in hits))
            cover = sum(norm(f) in evidence for f in sample["facts"]) / len(sample["facts"])
            results.append({"id": sample["id"], "question": sample["question"], "gold_answer": sample["answer"],
                "gold_pages": sample["pages"], "retrieved_pages": [h["pdf_page"] for h in hits],
                "hit_at_5": bool(positions), "reciprocal_rank": 1 / min(positions) if positions else 0,
                "evidence_fact_coverage": cover, "elapsed_ms": (time.perf_counter() - start) * 1000})
        strategies[strategy] = {"hit_at_5": statistics.mean(r["hit_at_5"] for r in results),
            "mrr_at_5": statistics.mean(r["reciprocal_rank"] for r in results),
            "evidence_fact_coverage": statistics.mean(r["evidence_fact_coverage"] for r in results),
            "p95_ms": sorted(r["elapsed_ms"] for r in results)[-1], "samples": results}
    answers = []
    for sample in gold:
        rag = kb.ask(sample["question"], use_cache=False)
        baseline = kb.ask(sample["question"], mode="llm_only", use_cache=False)
        generated = rag["generation"] == "llm"
        answer_coverage = sum(norm(f) in norm(rag["answer"]) for f in sample["facts"]) / len(sample["facts"]) if generated else None
        baseline_coverage = sum(norm(f) in norm(baseline["answer"]) for f in sample["facts"]) / len(sample["facts"]) if baseline["generation"] == "llm" else None
        answers.append({"id": sample["id"], "question": sample["question"], "gold_answer": sample["answer"],
                        "rag": rag, "llm_only": baseline, "generated_answer_fact_coverage": answer_coverage,
                        "baseline_answer_fact_coverage": baseline_coverage,
                        "status": "completed" if generated and baseline["generation"] == "llm" else "not_executed_llm_comparison"})
        print(f"问题 {sample['id']}：RAG={rag['generation']}，纯LLM={baseline['generation']}，{rag['elapsed_ms']:.0f} ms", flush=True)
    def concurrent_query(i):
        started = time.perf_counter()
        kb.retrieve(gold[i % len(gold)]["question"])
        return {"elapsed_ms": (time.perf_counter() - started) * 1000}
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        load = list(pool.map(concurrent_query, range(40)))
    latencies = sorted(r["elapsed_ms"] for r in load)
    report = {"dataset_size": len(gold), "model_configured": llm_configured(), "backend": kb.manifest["embedding"],
              "definitions": {"hit_at_5": "前5个片段中至少一个命中标准证据页；不是问答准确率",
                  "mrr_at_5": "首个标准证据页的倒数排名均值",
                  "evidence_fact_coverage": "检索证据包含的标准事实关键词比例；不是 LLM 忠实度",
                  "p95_ms": "10条样本 nearest-rank p95 等于最大值；不包含模型加载或首次建库"},
              "strategies": strategies, "answers": answers,
              "load_test": {"requests": 40, "workers": 4, "errors": 0,
                  "elapsed_s": time.perf_counter() - start, "p50_ms": statistics.median(latencies),
                  "p95_ms": latencies[37], "max_ms": max(latencies), "cached": False,
                  "scope": "本地检索引擎并发，不调用LLM；不等于 HTTP 或生产高并发验收"},
              "ragas": {"status": "not_executed", "reason": "未提供评审 LLM 与授权的外部调用；不将检索指标冒充 RAGAS 分数"}}
    dump_json(BASE / "实测评估结果.json", report)
    rows = ["# 实测评估报告", "", "所有数据由 evaluate.py 实际运行生成。标准答案仅用于评估，不进入线上问答。", "",
            "## 检索评估", "", "|策略|Hit@5|MRR@5|证据事实覆盖率|P95（ms）|", "|---|---:|---:|---:|---:|"]
    for name, item in strategies.items():
        rows.append(f"|{name}|{item['hit_at_5']:.1%}|{item['mrr_at_5']:.3f}|{item['evidence_fact_coverage']:.1%}|{item['p95_ms']:.2f}|")
    rows += ["", "Hit@5 是证据页命中率，事实覆盖率是关键词代理指标，均不是生成答案准确率。", "",
             "## 混合检索逐题结果", "", "|ID|来源 PDF 页|命中页|Hit@5|事实覆盖率|", "|---|---|---|---|---:|"]
    for r in strategies["hybrid"]["samples"]:
        rows.append(f"|{r['id']}|{r['gold_pages']}|{r['retrieved_pages']}|{r['hit_at_5']}|{r['evidence_fact_coverage']:.1%}|")
    lt = report["load_test"]
    rows += ["", "## 性能", "", f"引擎 4 个线程、40 次无缓存请求：P50 {lt['p50_ms']:.2f} ms，P95 {lt['p95_ms']:.2f} ms，最大 {lt['max_ms']:.2f} ms，错误 {lt['errors']}。",
             "并发计时不包含建库、模型启动或 LLM 生成，不能据此认定完整 RAG 生成满足 3 秒。", "",
             "## LLM 与 RAGAS", "", "已配置模型，详见 JSON 中的逐题输出。" if llm_configured() else
             "本次未配置大模型，已生成逐题检索证据和纯 LLM 未执行记录。RAG 生成、纯 LLM 对比、答案忠实度、英文生成及 RAGAS 评分均待接入模型后实测。", "",
             "## 复现", "", "使用研发目录同一 Python 环境：`python 测试/evaluate.py`（从工单1目录执行）。完整原始记录见 实测评估结果.json。"]
    (BASE / "实测评估报告.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({k: {x: v for x, v in item.items() if x != "samples"} for k, item in strategies.items()}, ensure_ascii=False, indent=2))
    print(json.dumps(report["load_test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
