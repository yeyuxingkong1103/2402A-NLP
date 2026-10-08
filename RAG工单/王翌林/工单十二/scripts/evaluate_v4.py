# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
scripts/evaluate_v4.py —— 工单四 对比评估脚本（新增文件）

三模式对比（16 题 = 工单三 T01-T14 + 工单四 IMG-05/06）：
  v3（无图像，ask_rag） vs v4（有图像，ask） vs pure_llm（基线）
指标（启发式，见 metrics 定义）：
  accuracy(关键词全命中) / faithfulness(答案-上下文支撑率) /
  relevance(答案-问题相关性) / ctx_precision(上下文精度) / ctx_recall(上下文召回) /
  latency；按题型分 text/table/image 三类统计。
输出：docs/eval_v4_results.json + 控制台 Markdown 对比表。
"""
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv  # 工单四：显式加载 .env，否则 DEEPSEEK_API_KEY 缺失

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()  # 工单四：从项目根目录加载 .env（教训：新脚本必须显式加载）
WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
OUT_JSON = Path("docs/eval_v4_results.json")

# ---------------- 工单四：16 题与期望关键词 ----------------
QUESTIONS = [
    (1, "招股说明书1", "text", "本次发行的发行股数是多少？",
     ["股", "万股"]),
    (2, "招股说明书1", "table", "本次发行的募集资金总额是多少？",
     ["万元", "亿元", "募集"]),
    (3, "招股说明书1", "text", "公司的主要关联方有哪些？",
     ["有限公司", "关联"]),
    (4, "招股说明书1", "table", "公司前五大股东的持股比例是多少？",
     ["%", "持股", "股东"]),
    (5, "招股说明书1", "table", "报告期内，公司来自军用领域的收入分别是多少？",
     ["万元", "军用"]),
    (6, "招股说明书1", "table", "报告期内公司营业收入分别是多少？",
     ["万元", "营业收入"]),
    (7, "招股说明书1", "table", "前五大客户占营业收入的比例是多少？",
     ["%", "客户"]),
    (8, "招股说明书2", "table", "武汉力源信息本次发行的发行股数是多少？",
     ["万股", "股"]),
    (9, "招股说明书2", "text", "力源信息的募集资金投向哪些项目？",
     ["项目", "募集"]),
    (10, "招股说明书2", "table", "力源信息报告期内主营业务收入构成是什么？",
     ["%", "收入"]),
    (11, "招股说明书1", "text", "How many shares are issued in this offering?",
     ["股", "万"]),
    (12, "招股说明书1", "table", "What is the total amount of funds raised?",
     ["万元", "亿"]),
    (13, "招股说明书1", "text", "Who are the main related parties of the company?",
     ["有限公司"]),
    (14, "招股说明书1", "table", "What is the shareholding ratio of top five shareholders?",
     ["%", "股东"]),
    # 工单四：图像题 id 使用 105/106，避免与 T05/T06 冲突污染金标锚定
    (105, "招股说明书2", "image", "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
     ["4", "销售部", "销售处"]),
    (106, "招股说明书2", "image", "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
     ["汽车", "IC卡"]),
]

# 工单四：图像题的金标要点（faithfulness/ctx 指标锚点）
GOLD_FACTS = {105: ["电话及网络销售部", "渠道销售部", "珠海", "深圳", "北京", "武汉", "广州", "成都"],
              106: ["14.0", "-2.0", "汽车", "IC卡"]}


def _terms(text: str) -> set:
    """工单四：中文 2-gram + 英文词切分（轻量词项化）"""
    en = set(w.lower() for w in re.findall(r"[A-Za-z]{2,}", text))
    zh = set(text[i:i + 2] for i in range(len(text) - 1)
             if re.match(r"[\u4e00-\u9fa5]", text[i])
             and re.match(r"[\u4e00-\u9fa5]", text[i + 1]))
    return en | zh


def score_answer(qid: int, qtype: str, question: str, answer: str,
                 expect_kws: list, contexts: str) -> dict:
    """工单四：启发式指标计算（准确率/忠实度/相关性/上下文精度召回）"""
    ans, ctx, q = _terms(answer), _terms(contexts), _terms(question)
    kws_hit = [k for k in expect_kws if k in answer]
    gold = GOLD_FACTS.get(qid, [])
    # 工单四：忠实度——答案词项被上下文支撑的比例（图像题锚定金标事实）
    anchor = _terms(" ".join(gold)) & ans if gold else ans
    support = {t for t in (anchor or ans) if t in ctx}
    faithfulness = len(support) / max(1, len(anchor or ans))
    # 工单四：相关性——答案与问题词项重合
    relevance = len(ans & q) / max(1, min(len(q), 30))
    # 工单四：上下文精度——期望关键词出现在检索上下文的比例；召回——金标事实落入上下文
    ctx_precision = (sum(1 for k in expect_kws if k in contexts)
                     / max(1, len(expect_kws)))
    ctx_recall = (sum(1 for g in gold if g in contexts) / len(gold)) if gold else \
        (1.0 if kws_hit else (0.5 if any(k in contexts for k in expect_kws) else 0.0))
    return {"kws_hit": kws_hit,
            "accuracy": 1.0 if len(kws_hit) == len(expect_kws) else 0.0,
            "faithfulness": round(faithfulness, 3),
            "relevance": round(relevance, 3),
            "ctx_precision": round(ctx_precision, 3),
            "ctx_recall": round(ctx_recall, 3)}


def collect_context(r: dict) -> str:
    """工单四：汇总三路检索上下文（用于忠实度/上下文指标）"""
    parts = [c.get("content", "") for c in r.get("retrieved_text_chunks", [])]
    parts += [c.get("content", c.get("table_text", "")) for c in
              r.get("retrieved_tables", [])]
    parts += [f"{im.get('caption','')}{im.get('ocr_text','')}{im.get('vqa_text','')}"
              for im in r.get("retrieved_images", [])]
    return "\n".join(parts)


def run_mode(engine, mode: str) -> list:
    """工单四：单模式跑全部 16 题"""
    rows = []
    for qid, doc, qtype, question, kws in QUESTIONS:
        # 工单四：perf_counter 单调时钟计时，规避 WSL2 墙钟跳变导致负耗时
        t0 = time.perf_counter()
        try:
            if mode == "pure_llm":
                r = engine["llm"](question)
            elif mode == "v3":
                r = engine["v3"].ask_rag(question, doc_id=doc)
            else:
                r = engine["v4"].ask(question, doc_id=doc, use_image=True)
            latency = (time.perf_counter() - t0) * 1000
            ctx = collect_context(r)
            s = score_answer(qid, qtype, question, r.get("answer", ""), kws, ctx)
            s.update({"id": qid, "doc": doc, "qtype": qtype,
                      "question": question, "mode": mode,
                      "latency_ms": round(latency, 1),
                      "answer": r.get("answer", "")[:200],
                      "n_images": len(r.get("retrieved_images", []))})
        except Exception as e:                      # 工单四：单题失败容错
            s = {"id": qid, "doc": doc, "qtype": qtype, "question": question,
                 "mode": mode, "answer": f"ERROR: {e}", "accuracy": 0.0,
                 "faithfulness": 0.0, "relevance": 0.0, "ctx_precision": 0.0,
                 "ctx_recall": 0.0,
                 "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                 "kws_hit": [], "n_images": 0}
        rows.append(s)
        print(f"[{mode}] id{qid} acc={s['accuracy']} lat={s['latency_ms']}ms")
    return rows


def summarize(rows: list) -> dict:
    """工单四：模式级汇总（整体+分题型）"""
    def agg(rs, keys=("accuracy", "faithfulness", "relevance",
                      "ctx_precision", "ctx_recall")):
        n = max(1, len(rs))
        return {k: round(sum(r[k] for r in rs) / n, 3) for k in keys} | {
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rs) / n, 1)}

    out = {"all": agg(rows)}
    for qt in ("text", "table", "image"):
        sub = [r for r in rows if r["qtype"] == qt]
        if sub:
            out[qt] = agg(sub)
    return out


def main() -> None:
    """工单四：三模式对比评估主流程"""
    print(f"=== 工单四对比评估（{WORK_ORDER}） ===")
    from src.rag_engine_v3 import RAGEngineV3
    from src.rag_engine_v4 import RAGEngineV4
    v3 = RAGEngineV3(top_k=5)
    v4 = RAGEngineV4(v3_engine=v3, top_k=5)
    engines = {"v3": v3, "v4": v4, "llm": v3.ask_llm}

    # 工单四：预热（加载 bge-m3/reranker + LLM 首连），避免冷启动污染首题延迟
    print("--- 预热中（模型懒加载 + LLM 首连）---")
    try:
        v3.ask_llm("你好")
        v3.ask_rag("公司主营业务是什么", doc_id="招股说明书1")
        v4.ask("公司主营业务是什么", doc_id="招股说明书1")
        print("--- 预热完成 ---")
    except Exception as e:                      # 工单四：预热失败不阻塞评估
        print(f"预热警告: {e}")

    all_rows = []
    for mode, runner in (("pure_llm", run_mode), ("v3", run_mode), ("v4", run_mode)):
        all_rows += runner(engines, mode)

    result = {"work_order": WORK_ORDER, "ts": time.strftime("%F %T"),
              "note": "图像通道：Qwen2-VL-2B-Instruct 生成 caption 与模板化 VQA，"
                      "PaddleOCR 提取图内文字，bge-m3 文本向量 + 关键词加权检索"
                      "（Chinese-CLIP 模型缺失时自动跳过图像向量通道）",
              "questions": len(QUESTIONS), "rows": all_rows,
              "summary": {m: summarize([r for r in all_rows if r["mode"] == m])
                          for m in ("pure_llm", "v3", "v4")}}
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    # 工单四：控制台 Markdown 对比表（准确率提升百分比）
    sm = result["summary"]
    print("\n| 指标 | pure_llm | 工单三(v3) | 工单四(v4) | v4 vs v3 |")
    print("|---|---|---|---|---|")
    for k, name in (("accuracy", "准确率"), ("faithfulness", "忠实度"),
                    ("relevance", "答案相关性"), ("ctx_precision", "上下文精度"),
                    ("ctx_recall", "上下文召回")):
        v3v, v4v = sm["v3"]["all"][k], sm["v4"]["all"][k]
        lift = f"+{(v4v - v3v) / max(v3v, 0.01) * 100:.0f}%" if v3v else "—"
        print(f"| {name} | {sm['pure_llm']['all'][k]} | {v3v} | {v4v} | {lift} |")
    print(f"| 平均响应ms | {sm['pure_llm']['all']['avg_latency_ms']} | "
          f"{sm['v3']['all']['avg_latency_ms']} | {sm['v4']['all']['avg_latency_ms']} | — |")
    for qt, name in (("table", "表格准确率"), ("image", "图像准确率")):
        if qt in sm["v4"]:
            print(f"| {name}(v3/v4) | — | {sm['v3'].get(qt,{}).get('accuracy','—')} | "
                  f"{sm['v4'][qt]['accuracy']} | — |")
    print(f"\n结果已写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
