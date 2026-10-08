# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
scripts/evaluate_optimized.py —— 工单二 Step 7 优化前后对比评估

三链路 × 10 题 × 六指标：
  链路：baseline（工单一 ask_rag，Milvus 固定分块检索）/ optimized（工单二混合检索+重排
        +父子块+多语言）/ pure_llm（ask_llm 无检索对照）
  指标：
    1. accuracy        准确率：答案命中参考锚点关键词（客观判定）
    2. latency         响应时间：链路返回 latency_ms
    3. faithfulness    忠实度：LLM-as-Judge 判定答案是否完全基于参考信息（0-1）
    4. relevance       答案相关性：Judge 判定答案是否切题（0-1）
    5. ctx_precision   上下文精度：top_k 检索块中含锚点块占比
    6. ctx_recall      上下文召回：检索块文本含锚点比例（块覆盖）
  产出：data/optimized/compare_eval.json + docs/screenshots/ 对比图（--charts）
用法：
  python scripts/evaluate_optimized.py            # 三链路评估
  python scripts/evaluate_optimized.py --charts   # 仅根据已有 JSON 重新生成图表
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

OUT_JSON = "data/optimized/compare_eval.json"

# ---------- 工单二 10 条核心问题 + 参考锚点（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
# anchors：答案正确性锚点（命中任一即算对）；ctx_anchors：检索上下文应包含的关键词
QUESTIONS = [
    {"id": "Q1", "q": "武汉兴图新科电子股份有限公司的注册资本是多少？", "type": "zh-fact",
     "anchors": ["5,520", "5520", "55.20", "55.2"], "ctx_anchors": ["5,520", "注册资本"]},
    {"id": "Q2", "q": "公司的法定代表人是谁？", "type": "zh-fact",
     "anchors": ["程家明"], "ctx_anchors": ["程家明", "法定代表人"]},
    {"id": "Q3", "q": "军用领域收入占营业收入的比例大概是多少？", "type": "zh-table",
     "anchors": ["97", "82.10", "82"], "ctx_anchors": ["军品收入", "占营业收入"]},
    {"id": "Q4", "q": "公司主要客户有哪些？客户集中度如何？", "type": "zh-summary",
     "anchors": ["军方", "电子科技", "军工", "客户集中"], "ctx_anchors": ["客户"]},
    {"id": "Q5", "q": "公司面临哪些主要风险因素？", "type": "zh-summary",
     "anchors": ["军品", "客户集中", "税收优惠", "应收账款"], "ctx_anchors": ["风险"]},
    {"id": "Q6", "q": "本次募集资金的主要用途是什么？", "type": "zh-summary",
     "anchors": ["特种", "产业化", "研发", "补充流动资金"], "ctx_anchors": ["募集资金"]},
    {"id": "Q7", "q": "报告期内公司的研发投入情况如何？", "type": "zh-fact",
     "anchors": ["研发费用", "研发投入", "占营业收入"], "ctx_anchors": ["研发"]},
    {"id": "Q8", "q": "What is the registered capital of Wuhan Xingtu Xinke?", "type": "en-fact",
     "anchors": ["5,520", "5520", "55.20", "55.2"], "ctx_anchors": ["注册资本", "5,520"]},
    {"id": "Q9", "q": "Who is the legal representative of the company?", "type": "en-fact",
     "anchors": ["程家明", "Cheng Jiaming"], "ctx_anchors": ["程家明", "法定代表人"]},
    {"id": "Q10", "q": "What are the main risk factors of the company?", "type": "en-summary",
     "anchors": ["military", "customer", "concentration", "军"], "ctx_anchors": ["风险"]},
]


_NEGATION = re.compile(r"未提及|未明确|无法确定|无法判断|没有提到|does not (explicitly )?(identify|mention|specify)|not explicitly|cannot determine|not identified", re.I)


def _contains(text: str, anchors) -> bool:
    return any(a.lower() in (text or "").lower() for a in anchors)


def _answer_correct(answer: str, anchors) -> bool:
    """工单二锚点判定（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    命中锚点且答案未以否定口径作答（'未提及'/'无法确定'类回答不算正确）"""
    if _NEGATION.search(answer or ""):
        return False
    return _contains(answer, anchors)


def llm_judge(question: str, answer: str, refs: str = "") -> dict:
    """LLM-as-Judge（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    忠实度 faithfulness（答案是否完全基于参考信息）与相关性 relevance（是否切题），0-1 分"""
    try:
        from src.llm_client import chat
        sys_p = ("你是严格的 RAG 评估专家。仅输出 JSON：{\"faithfulness\": 0-1, \"relevance\": 0-1}。"
                 "faithfulness：答案是否完全由参考信息支撑（无参考时按常识一致性）；"
                 "relevance：答案是否直接回答了问题。只输出 JSON。")
        user = f"问题：{question}\n\n参考信息：{refs[:1500] or '（无）'}\n\n待评答案：{answer[:800]}"
        r = chat(messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                 temperature=0.0, max_tokens=100)
        m = re.search(r"\{[^}]*\}", r.get("content", ""), re.S)
        if m:
            j = json.loads(m.group())
            return {"faithfulness": float(j.get("faithfulness", 0)),
                    "relevance": float(j.get("relevance", 0))}
    except Exception as e:
        logger.warning(f"Judge 失败: {e}")
    return {"faithfulness": 0.0, "relevance": 0.0}


def eval_chain(engine, chain: str) -> dict:
    """评估一条链路 10 题（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    rows = []
    for item in QUESTIONS:
        t0 = time.time()
        if chain == "optimized":
            r = engine.ask_optimized(item["q"], use_cache=False)  # 冷跑公平对比（缓存收益单独展示）
            refs_texts = [x.get("preview", "") for x in r.get("references", [])]
        elif chain == "baseline":
            r = engine.ask_rag(item["q"])
            refs_texts = [x.get("preview", "") for x in r.get("references", [])]
        else:  # pure_llm
            r = engine.ask_llm(item["q"])
            refs_texts = []
        wall = (time.time() - t0) * 1000
        answer = r.get("answer", "") or ""
        latency = r.get("latency_ms", round(wall, 1))
        # 客观指标（否定口径不计正确，人工智能NLP-RAG-基于PDF文档的问答系统优化）
        acc = _answer_correct(answer, item["anchors"])
        ctx_hit = sum(1 for t in refs_texts if _contains(t, item["ctx_anchors"]))
        ctx_precision = round(ctx_hit / len(refs_texts), 3) if refs_texts else 0.0
        ctx_recall = 1.0 if ctx_hit > 0 else 0.0
        j = llm_judge(item["q"], answer, " ".join(refs_texts))
        rows.append({"id": item["id"], "type": item["type"], "question": item["q"],
                     "answer": answer[:200], "latency_ms": latency, "accuracy": int(acc),
                     "ctx_precision": ctx_precision, "ctx_recall": ctx_recall,
                     "faithfulness": j["faithfulness"], "relevance": j["relevance"],
                     "n_refs": len(refs_texts)})
        print(f"[{chain:>9}] {item['id']} {latency:>7.0f}ms acc={'Y' if acc else 'N'} "
              f"faith={j['faithfulness']:.1f} rel={j['relevance']:.1f} "
              f"ctxP={ctx_precision:.2f} ctxR={ctx_recall:.0f} | {answer[:40].replace(chr(10), ' ')}")
    n = len(rows)
    summary = {
        "chain": chain, "n": n,
        "accuracy": round(sum(x["accuracy"] for x in rows) / n, 3),
        "avg_latency_ms": round(sum(x["latency_ms"] for x in rows) / n, 1),
        "p95_latency_ms": round(sorted(x["latency_ms"] for x in rows)[int(0.95 * n) - 1], 1),
        "faithfulness": round(sum(x["faithfulness"] for x in rows) / n, 3),
        "relevance": round(sum(x["relevance"] for x in rows) / n, 3),
        "ctx_precision": round(sum(x["ctx_precision"] for x in rows) / n, 3),
        "ctx_recall": round(sum(x["ctx_recall"] for x in rows) / n, 3),
    }
    return {"summary": summary, "rows": rows}


def warmup(engine):
    """预热检索索引与 LLM 连接（不计入指标，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    from src.rag_engine import get_hybrid_retriever
    get_hybrid_retriever()
    engine.ask_optimized("预热：公司全称是什么？", use_cache=False)
    print("[预热完成]")


def main():
    ap = argparse.ArgumentParser(description="工单二 Step 7 对比评估（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    ap.add_argument("--charts", action="store_true", help="仅根据已有 JSON 重新生成图表")
    args = ap.parse_args()

    if args.charts:
        from scripts.gen_charts import gen_all
        data = json.load(open(OUT_JSON, encoding="utf-8"))
        gen_all(data)
        return

    from src.rag_engine import RAGEngine
    engine = RAGEngine(top_k=5)
    warmup(engine)
    result = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "work_order": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
              "questions": [q["id"] for q in QUESTIONS], "chains": {}}
    for chain in ("baseline", "optimized", "pure_llm"):
        print("=" * 76)
        result["chains"][chain] = eval_chain(engine, chain)
    # 优化提升幅度（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    b, o = result["chains"]["baseline"]["summary"], result["chains"]["optimized"]["summary"]
    result["improvement"] = {
        "accuracy_delta": round(o["accuracy"] - b["accuracy"], 3),
        "accuracy_lift_pct": round((o["accuracy"] - b["accuracy"]) / max(b["accuracy"], 1e-9) * 100, 1),
        "latency_delta_ms": round(o["avg_latency_ms"] - b["avg_latency_ms"], 1),
        "latency_lift_pct": round((b["avg_latency_ms"] - o["avg_latency_ms"]) / max(b["avg_latency_ms"], 1e-9) * 100, 1),
        "ctx_recall_delta": round(o["ctx_recall"] - b["ctx_recall"], 3),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("=" * 76)
    for c in ("baseline", "optimized", "pure_llm"):
        s = result["chains"][c]["summary"]
        print(f"[{c:>9}] 准确率={s['accuracy']:.0%} 平均延迟={s['avg_latency_ms']}ms "
              f"忠实度={s['faithfulness']} 相关性={s['relevance']} "
              f"上下文P={s['ctx_precision']} 上下文R={s['ctx_recall']}")
    print(f"提升: 准确率 +{result['improvement']['accuracy_lift_pct']}% | "
          f"延迟 -{result['improvement']['latency_lift_pct']}%")
    print(f"明细: {OUT_JSON}")
    from scripts.gen_charts import gen_all
    gen_all(result)


if __name__ == "__main__":
    main()
