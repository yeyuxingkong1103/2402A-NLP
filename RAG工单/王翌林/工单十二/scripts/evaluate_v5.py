# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
scripts/evaluate_v5.py —— 工单五 多轮对话评估脚本（新增文件）

跑通工单五验收的 5 轮连续对话，验证：
  1. 指代消解正确性（他/这个公司 → 兴图新科；那X呢 → 切换实体复用问法）
  2. 准确率（每轮期望关键词命中）
  3. 响应时间 ≤ 3s
输出：docs/eval_v5_results.json + 控制台 Markdown 表
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()                                   # 工单五：显式加载 .env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK_ORDER = "人工智能NLP-RAG-Query 理解优化任务"
OUT_JSON = Path("docs/eval_v5_results.json")

# ---------------- 工单五：5 轮验收对话 ----------------
DIALOG = [
    {
        "turn": 1,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "expect_keywords": ["万元", "军用"],
        "expect_entity": "武汉兴图新科电子股份有限公司",
        "expect_doc": "招股说明书1",
        "expect_strategy": "direct",
    },
    {
        "turn": 2,
        "question": "他参与的哪个工程荣获了国家科技进步一等奖？",
        "expect_keywords": ["科技进步", "工程"],
        "expect_entity": "武汉兴图新科电子股份有限公司",
        "expect_doc": "招股说明书1",
        "expect_strategy": "pronoun_resolution",
    },
    {
        "turn": 3,
        "question": "这个公司的法定代表人是谁？",
        "expect_keywords": ["法定代表人"],
        "expect_entity": "武汉兴图新科电子股份有限公司",
        "expect_doc": "招股说明书1",
        "expect_strategy": "pronoun_resolution",
    },
    {
        "turn": 4,
        "question": "那武汉力源信息技术股份有限公司呢？",
        "expect_keywords": ["法定代表人"],
        "expect_entity": "武汉力源信息技术股份有限公司",
        "expect_doc": "招股说明书2",
        "expect_strategy": "switch_entity_reuse_intent",
    },
    {
        "turn": 5,
        "question": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
        "expect_keywords": ["销售处", "销售部"],
        "expect_entity": "武汉力源信息技术股份有限公司",
        "expect_doc": "招股说明书2",
        "expect_strategy": "direct",
    },
]


def score_turn(answer: str, expect_keywords: list) -> dict:
    """工单五：关键词命中评分"""
    hits = [k for k in expect_keywords if k in answer]
    return {
        "hit_keywords": hits,
        "accuracy": 1.0 if len(hits) == len(expect_keywords) else 0.0,
    }


def main() -> None:
    print(f"=== 工单五多轮对话评估（{WORK_ORDER}） ===")
    from src.conversation_engine import ConversationEngine
    engine = ConversationEngine()

    # 工单五：预热（避免冷启动污染首题延迟）
    # 仅加载模型不够——首问的 Milvus 连接 + LLM 首连仍有数百~数千毫秒开销，
    # 故额外执行一次真实 RAG 查询，把所有懒加载路径走完。
    print("--- 预热中（模型加载 + 真实 RAG 查询）---")
    try:
        engine._get_rag().warmup()
        engine._get_rag().ask("公司主营业务是什么", doc_id="招股说明书1")
        print("--- 预热完成 ---")
    except Exception as e:
        print(f"预热警告: {e}")

    rows = []
    session_id = None
    for turn in DIALOG:
        t0 = time.perf_counter()
        try:
            r = engine.chat(turn["question"], session_id=session_id,
                            use_image=True)
            session_id = r["session_id"]
            latency = (time.perf_counter() - t0) * 1000
            s = score_turn(r.get("answer", ""), turn["expect_keywords"])
            # 工单五：指代消解正确性校验
            entity_ok = r.get("entity") == turn["expect_entity"]
            doc_ok = r.get("doc_id") == turn["expect_doc"]
            strategy_ok = (r.get("coref_strategy") == turn["expect_strategy"]
                           or (turn["expect_strategy"] == "pronoun_resolution"
                               and r.get("coref_strategy") in
                               ("pronoun_resolution", "inherit_entity")))
            coref_ok = entity_ok and doc_ok and strategy_ok
        except Exception as e:
            r = {"answer": f"ERROR: {e}", "session_id": session_id,
                 "entity": None, "doc_id": None, "coref_strategy": "error",
                 "resolved_query": turn["question"]}
            latency = (time.perf_counter() - t0) * 1000
            s = {"hit_keywords": [], "accuracy": 0.0}
            entity_ok = doc_ok = strategy_ok = coref_ok = False

        row = {
            "turn": turn["turn"],
            "question": turn["question"],
            "resolved_query": r.get("resolved_query", ""),
            "entity": r.get("entity"),
            "doc_id": r.get("doc_id"),
            "strategy": r.get("coref_strategy"),
            "answer": r.get("answer", "")[:300],
            "accuracy": s["accuracy"],
            "hit_keywords": s["hit_keywords"],
            "entity_ok": entity_ok,
            "doc_ok": doc_ok,
            "strategy_ok": strategy_ok,
            "coref_ok": coref_ok,
            "latency_ms": round(latency, 1),
            "latency_ok": latency <= 3000,
        }
        rows.append(row)
        print(f"[turn {row['turn']}] acc={row['accuracy']} "
              f"coref={'OK' if coref_ok else 'FAIL'} "
              f"lat={row['latency_ms']}ms "
              f"entity={row['entity']} strategy={row['strategy']}")

    # 工单五：汇总
    n = len(rows)
    acc = sum(r["accuracy"] for r in rows) / n
    coref_rate = sum(1 for r in rows if r["coref_ok"]) / n
    lat_ok_rate = sum(1 for r in rows if r["latency_ok"]) / n
    avg_lat = sum(r["latency_ms"] for r in rows) / n

    result = {
        "work_order": WORK_ORDER,
        "ts": time.strftime("%F %T"),
        "dialog_turns": n,
        "rows": rows,
        "summary": {
            "accuracy": round(acc, 3),
            "coref_resolution_rate": round(coref_rate, 3),
            "avg_latency_ms": round(avg_lat, 1),
            "latency_ok_rate": round(lat_ok_rate, 3),
            "all_pass": acc >= 0.9 and coref_rate >= 0.9 and avg_lat <= 3000,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print("\n| 轮次 | 问题(摘要) | 准确率 | 消解策略 | 实体正确 | 耗时ms | ≤3s |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['turn']} | {r['question'][:25]}... | {r['accuracy']} | "
              f"{r['strategy']} | {'✓' if r['coref_ok'] else '✗'} | "
              f"{r['latency_ms']} | {'✓' if r['latency_ok'] else '✗'} |")
    print(f"\n总准确率: {acc:.1%} ｜ 消解正确率: {coref_rate:.1%} ｜ "
          f"平均耗时: {avg_lat:.0f}ms ｜ 达标: {'✓' if result['summary']['all_pass'] else '✗'}")
    print(f"结果已写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
