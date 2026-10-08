# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
scripts/evaluate_v6.py —— 工单六 检索策略对比评估脚本（新增文件）

三种检索策略 × 16 题对比：
  vector  ：bge-m3 向量召回 + LLM 交叉编码重排
  fulltext：倒排索引全文检索（AND + 多字段 TF-IDF）+ TF-IDF 重排
  hybrid  ：向量+全文 RRF 投票融合 + LLM 重排（默认）
指标：
  accuracy   答案期望关键词全命中率（≥90% 目标）
  recall     检索上下文对期望关键词的覆盖率（≥95% 目标）
  latency_ms 单轮平均响应（≤3000ms 目标）
输出：docs/eval_v6_results.json + 控制台 Markdown 对比表
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()                                   # 工单六：显式加载 .env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK_ORDER = "人工智能NLP-RAG-混合检索任务"
OUT_JSON = Path("docs/eval_v6_results.json")

# 工单六：16 题与期望关键词（复用工单四题集，覆盖文本/表格/图像/英文）
QUESTIONS = [
    (1, "招股说明书1", "本次发行的发行股数是多少？", ["股", "万股"]),
    (2, "招股说明书1", "本次发行的募集资金总额是多少？", ["万元", "亿元", "募集"]),
    (3, "招股说明书1", "公司的主要关联方有哪些？", ["有限公司", "关联"]),
    (4, "招股说明书1", "公司前五大股东的持股比例是多少？", ["%", "持股", "股东"]),
    (5, "招股说明书1", "报告期内，公司来自军用领域的收入分别是多少？", ["万元", "军用"]),
    (6, "招股说明书1", "报告期内公司营业收入分别是多少？", ["万元", "营业收入"]),
    (7, "招股说明书1", "前五大客户占营业收入的比例是多少？", ["%", "客户"]),
    (8, "招股说明书2", "武汉力源信息本次发行的发行股数是多少？", ["万股", "股"]),
    (9, "招股说明书2", "力源信息的募集资金投向哪些项目？", ["项目", "募集"]),
    (10, "招股说明书2", "力源信息报告期内主营业务收入构成是什么？", ["%", "收入"]),
    (11, "招股说明书1", "How many shares are issued in this offering?", ["股", "万"]),
    (12, "招股说明书1", "What is the total amount of funds raised?", ["万元", "亿"]),
    (13, "招股说明书1", "Who are the main related parties of the company?", ["有限公司"]),
    (14, "招股说明书1", "What is the shareholding ratio of top five shareholders?",
     ["%", "股东"]),
    (105, "招股说明书2",
     "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
     ["4", "销售部", "销售处"]),
    (106, "招股说明书2",
     "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
     ["汽车", "IC卡"]),
]

# 工单六：三套策略（与 PRESETS 对应，fulltext 用 tfidf 重排器避免模型依赖差异）
STRATEGIES = {
    "vector": {"mode": "vector", "reranker": "llm"},
    "fulltext": {"mode": "fulltext", "reranker": "tfidf", "match": "and"},
    "hybrid": {"mode": "hybrid", "fusion": "rrf", "reranker": "llm"},
}


def context_text(r: dict) -> str:
    parts = [c.get("content", "") for c in r.get("retrieved_text_chunks", [])]
    parts += [c.get("content", c.get("table_text", ""))
              for c in r.get("retrieved_tables", [])]
    parts += [f"{im.get('caption','')}{im.get('ocr_text','')}{im.get('vqa_text','')}"
              for im in r.get("retrieved_images", [])]
    return "\n".join(parts)


def run_strategy(engine, name: str, cfg: dict) -> list:
    rows = []
    for qid, doc, question, kws in QUESTIONS:
        t0 = time.perf_counter()
        try:
            r = engine.ask(question, doc_id=doc, use_image=True,
                           retrieval_config=cfg)
            latency = (time.perf_counter() - t0) * 1000
            answer = r.get("answer", "")
            hits = [k for k in kws if k in answer]
            ctx = context_text(r)
            ctx_hits = [k for k in kws if k in ctx]
            row = {"id": qid, "question": question, "mode": name,
                   "accuracy": 1.0 if len(hits) == len(kws) else 0.0,
                   "recall": round(len(ctx_hits) / len(kws), 3),
                   "hit_keywords": hits, "ctx_hits": ctx_hits,
                   "latency_ms": round(latency, 1),
                   "answer": answer[:200],
                   "vector_hits": r.get("retrieval", {}).get("vector_hits", 0),
                   "fulltext_hits": r.get("retrieval", {}).get("fulltext_hits", 0)}
        except Exception as e:
            row = {"id": qid, "question": question, "mode": name,
                   "accuracy": 0.0, "recall": 0.0, "hit_keywords": [],
                   "ctx_hits": [], "latency_ms": 0.0, "answer": f"ERROR: {e}",
                   "vector_hits": 0, "fulltext_hits": 0}
        rows.append(row)
        print(f"[{name}] id{qid} acc={row['accuracy']} "
              f"recall={row['recall']} lat={row['latency_ms']}ms")
    return rows


def summarize(rows: list) -> dict:
    n = max(1, len(rows))
    return {
        "accuracy": round(sum(r["accuracy"] for r in rows) / n, 3),
        "recall": round(sum(r["recall"] for r in rows) / n, 3),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        "latency_ok_rate": round(
            sum(1 for r in rows if r["latency_ms"] <= 3000) / n, 3),
    }


def main() -> None:
    print(f"=== 工单六检索策略对比评估（{WORK_ORDER}） ===")
    from src.rag_engine_v6 import RAGEngineV6
    engine = RAGEngineV6(top_k=5)

    print("--- 预热中（模型 + 全文索引 + 真实查询）---")
    try:
        engine.warmup()
        engine.ask("公司主营业务是什么", doc_id="招股说明书1")
        print("--- 预热完成 ---")
    except Exception as e:
        print(f"预热警告: {e}")

    all_rows, summary = [], {}
    for name, cfg in STRATEGIES.items():
        all_rows += run_strategy(engine, name, cfg)
        summary[name] = summarize([r for r in all_rows if r["mode"] == name])

    result = {"work_order": WORK_ORDER, "ts": time.strftime("%F %T"),
              "questions": len(QUESTIONS), "strategies": STRATEGIES,
              "rows": all_rows, "summary": summary}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print("\n| 策略 | 准确率(≥90%) | 召回率(≥95%) | 平均响应ms(≤3000) | 达标率 |")
    print("|---|---|---|---|---|")
    for name in STRATEGIES:
        s = summary[name]
        ok = "✓" if (s["accuracy"] >= 0.9 and s["recall"] >= 0.95
                     and s["avg_latency_ms"] <= 3000) else "✗"
        print(f"| {name} | {s['accuracy']} | {s['recall']} | "
              f"{s['avg_latency_ms']} | {s['latency_ok_rate']} {ok} |")
    print(f"\n结果已写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
