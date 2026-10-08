# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
scripts/evaluate.py —— 工单二性能/准确率评估脚本

用法（项目根目录）：
  python scripts/evaluate.py --optimized   # 优化链路：10 题逐个计时 + 平均
  python scripts/evaluate.py --baseline    # 基线链路（工单一 ask_rag）对比

流程：预热（加载模型/索引，不计入计时）→ 10 题顺序请求 → 每题记录 latency_ms 与
cache_hit → 写入 MySQL qa_logs → 输出明细表与平均值（验收标准：平均 ≤ 3000ms）。
第二轮重复运行时语义缓存命中，延迟显著下降（对比数据用于优化报告）。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根入 path
from dotenv import load_dotenv

load_dotenv()
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

# 工单二 10 条核心测试问题（与 docs/02_优化前后对比方案.md 问题集一致，人工智能NLP-RAG-基于PDF文档的问答系统优化）
TEST_QUESTIONS = [
    {"id": "Q1", "q": "武汉兴图新科电子股份有限公司的注册资本是多少？", "type": "zh-fact"},
    {"id": "Q2", "q": "公司的法定代表人是谁？", "type": "zh-fact"},
    {"id": "Q3", "q": "军用领域收入占营业收入的比例大概是多少？", "type": "zh-table"},
    {"id": "Q4", "q": "公司主要客户有哪些？客户集中度如何？", "type": "zh-summary"},
    {"id": "Q5", "q": "公司面临哪些主要风险因素？", "type": "zh-summary"},
    {"id": "Q6", "q": "本次募集资金的主要用途是什么？", "type": "zh-summary"},
    {"id": "Q7", "q": "报告期内公司的研发投入情况如何？", "type": "zh-fact"},
    {"id": "Q8", "q": "What is the registered capital of Wuhan Xingtu Xinke?", "type": "en-fact"},
    {"id": "Q9", "q": "Who is the legal representative of the company?", "type": "en-fact"},
    {"id": "Q10", "q": "What are the main risk factors of the company?", "type": "en-summary"},
]
TARGET_AVG_MS = 3000  # 工单验收：平均响应 ≤ 3 秒


def warmup(engine):
    """预热：加载检索索引/模型/建立 LLM 连接（不计入计时，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    t0 = time.time()
    from src.rag_engine import get_hybrid_retriever
    get_hybrid_retriever()
    engine.ask("预热：公司全称是什么？", use_cache=False)
    print(f"[预热完成 {(time.time() - t0):.1f}s（不计入指标）]")


def run_eval(use_optimized: bool, use_cache: bool = True):
    """执行评估：优化链路或基线链路（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    from src.rag_engine import RAGEngine
    engine = RAGEngine(top_k=5)
    rows = []
    for item in TEST_QUESTIONS:
        t0 = time.time()
        if use_optimized:
            r = engine.ask_optimized(item["q"], use_cache=use_cache)
        else:
            r = engine.ask_rag(item["q"])
        wall_ms = (time.time() - t0) * 1000
        rows.append({"id": item["id"], "type": item["type"], "q": item["q"],
                     "latency_ms": r.get("latency_ms", round(wall_ms, 1)),
                     "wall_ms": round(wall_ms, 1),
                     "cache_hit": r.get("cache_hit", False),
                     "lang": r.get("lang", "zh"),
                     "answer_preview": (r.get("answer") or "")[:60].replace("\n", " ")})
        print(f"{item['id']} [{item['type']:>10}] {r.get('latency_ms'):>8}ms "
              f"cache={'Y' if r.get('cache_hit') else '-'} | {rows[-1]['answer_preview']}")
    avg = sum(x["latency_ms"] for x in rows) / len(rows)
    p95 = sorted(x["latency_ms"] for x in rows)[int(0.95 * len(rows)) - 1]
    n_cache = sum(1 for x in rows if x["cache_hit"])
    result = {"mode": "optimized" if use_optimized else "baseline", "use_cache": use_cache,
              "n": len(rows), "avg_latency_ms": round(avg, 1), "p95_latency_ms": round(p95, 1),
              "cache_hits": n_cache, "target_avg_ms": TARGET_AVG_MS,
              "pass": avg <= TARGET_AVG_MS, "rows": rows}
    # 写入 MySQL qa_logs（工单要求：每次请求耗时入库，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    try:
        from src.db import get_session
        from src.models import QALog
        with get_session() as sess:
            for x in rows:
                sess.add(QALog(question=f"[eval-{result['mode']}] {x['q']}",
                               rag_answer=x["answer_preview"],
                               latency_ms=x["latency_ms"],
                               retrieved_chunks={"cache_hit": x["cache_hit"], "type": x["type"]}))
            sess.commit()
        result["qa_logs_written"] = len(rows)
    except Exception as e:
        logger.warning(f"qa_logs 写入失败（不影响评估）: {e}")
        result["qa_logs_written"] = 0
    return result


def print_summary(result):
    mode = result["mode"]
    print("=" * 76)
    print(f"工单二评估结果 [{mode}]（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    print(f"  问题数量   : {result['n']}")
    print(f"  平均响应   : {result['avg_latency_ms']} ms（目标 ≤ {result['target_avg_ms']} ms）"
          f" → {'✅ 达标' if result['pass'] else '❌ 未达标'}")
    print(f"  P95 响应   : {result['p95_latency_ms']} ms")
    print(f"  缓存命中   : {result['cache_hits']}/{result['n']}")
    print(f"  qa_logs    : 写入 {result.get('qa_logs_written', 0)} 条")
    out = f"data/optimized/eval_{mode}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  明细已保存 : {out}")
    print("=" * 76)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="工单二评估脚本（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--optimized", action="store_true", help="评估工单二优化链路")
    g.add_argument("--baseline", action="store_true", help="评估工单一基线链路（对比用）")
    ap.add_argument("--no-cache", action="store_true", help="禁用语义缓存（冷查询计时）")
    args = ap.parse_args()
    use_opt = not args.baseline
    engine_ref = None
    if use_opt:
        # 先构建引擎用于预热
        from src.rag_engine import RAGEngine
        engine_ref = RAGEngine(top_k=5)
        warmup(engine_ref._optimized)
    result = run_eval(use_opt, use_cache=not args.no_cache)
    print_summary(result)
