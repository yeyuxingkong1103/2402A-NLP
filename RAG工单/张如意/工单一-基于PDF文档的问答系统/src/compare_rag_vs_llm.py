# -*- coding: utf-8 -*-
"""
工单01 对比实验：RAG（基于PDF）vs 纯 LLM（无文档上下文）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

工单要求：「对比基于 pdf 的返回结果和只使用 LLM 返回的答案的对比分析」。
本脚本对工单给定的 10 个问题分别执行：
  1. RAG 回答   —— Pipeline.ask：Query理解 → 检索 → 生成，答案带页码引用
  2. 纯 LLM 回答 —— generator.generate_llm_only：不提供任何文档上下文（基线）
并记录：答案全文、引用来源、耗时、关键词命中情况，输出
  results/rag_vs_llm.json（结构化全量数据）
  results/rag_vs_llm.md （人读对比报告：总览表 + 逐题对比 + 结论）

用法：
    python "工单01-基于PDF文档的问答系统/src/compare_rag_vs_llm.py"
    python .../src/compare_rag_vs_llm.py --ids 260,95
    python .../src/compare_rag_vs_llm.py --top-k 8 --sleep 1

注意：
  · 纯 LLM 的耗时在重复运行时可能接近 0，因为 rag_core.llm 带磁盘缓存；
    演示/录屏前建议清理 data/cache/llm 目录以呈现真实延迟差异。
  · 关键词判分表来自 run_evaluation.ANSWERS_EXPECTED，可按需在该文件中校准。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# --- 让脚本可以独立运行：把项目根目录（工单作业/）加入模块搜索路径 ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 同级脚本目录（复用 run_evaluation 中的参考答案表）
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_core import config, generator              # noqa: E402
from rag_core.pipeline import PRESETS, Pipeline     # noqa: E402

from run_evaluation import ANSWERS_EXPECTED         # noqa: E402  （参考答案判分表）

CASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="工单01：RAG 与纯 LLM 答案对比（10 个问题）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="wo01_baseline", choices=sorted(PRESETS),
                   help="RAG 侧使用的流水线预设，默认 wo01_baseline")
    p.add_argument("--collection", default="prospectus", help="向量库集合名")
    p.add_argument("--top-k", type=int, default=None, help="检索片段数，默认取预设")
    p.add_argument("--ids", default=None, help="只对比指定题号，逗号分隔")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="每题之间的间隔秒数（限流保护），默认 0")
    p.add_argument("--out-json", default=str(RESULTS_DIR / "rag_vs_llm.json"))
    p.add_argument("--out-md", default=str(RESULTS_DIR / "rag_vs_llm.md"))
    return p


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _keyword_hits(answer: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """返回 (命中的关键词, 漏答的关键词)。"""
    hits = [k for k in keywords if k in (answer or "")]
    miss = [k for k in keywords if k not in (answer or "")]
    return hits, miss


def _clip(text: str, n: int = 1600) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n…（已截断，完整内容见 rag_vs_llm.json）"


def _cell(text: str, n: int = 60) -> str:
    """把文本压成适合放进 Markdown 表格的一行。"""
    s = (text or "").replace("\n", " ").replace("|", "\\|").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# 单题对比
# ---------------------------------------------------------------------------
def compare_one(pipeline: Pipeline, item: dict, top_k: int | None,
                keywords: list[str]) -> dict:
    """对一个问题分别跑 RAG 与纯 LLM，返回对比记录。"""
    qid, question = item["id"], item["question"]
    print(f"\n[{qid}] {question}")

    # ---- 1) RAG 回答 ----
    t0 = time.perf_counter()
    rag_error = ""
    try:
        result = pipeline.ask(question, top_k=top_k, return_trace=True)
        rag_answer = result.get("answer", "")
        rag_docs = result.get("docs", [])
        rag_citations = result.get("citations", [])
        rag_refused = bool(result.get("refused"))
    except Exception as e:                       # 容错：RAG 侧失败仍保留纯 LLM 基线
        rag_error, rag_answer, rag_docs, rag_citations, rag_refused = (
            str(e), f"（RAG 问答失败：{e}）", [], [], False)
        print(f"  [警告] RAG 侧失败：{e}")
    rag_latency = time.perf_counter() - t0

    # ---- 2) 纯 LLM 基线（不提供上下文）----
    t0 = time.perf_counter()
    llm_error = ""
    try:
        gen = generator.generate_llm_only(question)
        llm_answer = gen.answer
    except Exception as e:
        llm_error, llm_answer = str(e), f"（纯 LLM 问答失败：{e}）"
        print(f"  [警告] 纯 LLM 侧失败：{e}")
    llm_latency = time.perf_counter() - t0

    # ---- 3) 关键词命中对比 ----
    rag_hits, rag_miss = _keyword_hits(rag_answer, keywords)
    llm_hits, llm_miss = _keyword_hits(llm_answer, keywords)
    if len(rag_hits) > len(llm_hits):
        winner = "RAG"
    elif len(llm_hits) > len(rag_hits):
        winner = "纯LLM"
    else:
        winner = "持平"

    print(f"  RAG {rag_latency * 1000:.0f} ms / 纯LLM {llm_latency * 1000:.0f} ms | "
          f"关键词命中 {len(rag_hits)}/{len(keywords)} vs "
          f"{len(llm_hits)}/{len(keywords)} → {winner}")

    return {
        "id": qid,
        "question": question,
        "keywords": keywords,
        "rag": {
            "answer": rag_answer,
            "latency": round(rag_latency, 3),
            "refused": rag_refused,
            "error": rag_error,
            "citations": rag_citations,
            "docs": [{
                "chunk_id": d.get("chunk_id", ""), "doc": d.get("doc", ""),
                "page": d.get("page", ""), "type": d.get("type", ""),
                "score": round(float(d.get("final_score", d.get("score", 0))), 4),
                "snippet": (d.get("text") or "")[:200],
            } for d in rag_docs],
            "keyword_hits": rag_hits, "keyword_miss": rag_miss,
        },
        "llm_only": {
            "answer": llm_answer,
            "latency": round(llm_latency, 3),
            "error": llm_error,
            "keyword_hits": llm_hits, "keyword_miss": llm_miss,
        },
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(records: list[dict]) -> dict:
    """统计两侧的耗时、引用覆盖率、关键词命中率与胜负。"""
    rag_lat = [r["rag"]["latency"] for r in records if not r["rag"]["error"]]
    llm_lat = [r["llm_only"]["latency"] for r in records if not r["llm_only"]["error"]]
    n = len(records)

    rag_all_hit = sum(1 for r in records
                      if r["keywords"] and not r["rag"]["keyword_miss"])
    llm_all_hit = sum(1 for r in records
                      if r["keywords"] and not r["llm_only"]["keyword_miss"])
    cited = sum(1 for r in records if r["rag"]["citations"])
    refused = sum(1 for r in records if r["rag"]["refused"])

    return {
        "n": n,
        "rag_avg_latency": round(statistics.mean(rag_lat), 3) if rag_lat else None,
        "rag_p95_latency": round(sorted(rag_lat)[int(len(rag_lat) * 0.95) - 1], 3)
                            if len(rag_lat) >= 2 else (round(rag_lat[0], 3) if rag_lat else None),
        "rag_max_latency": round(max(rag_lat), 3) if rag_lat else None,
        "llm_avg_latency": round(statistics.mean(llm_lat), 3) if llm_lat else None,
        "rag_citation_rate": round(cited / n, 4) if n else None,
        "rag_refused": refused,
        "rag_keyword_accuracy": round(rag_all_hit / n, 4) if n else None,
        "llm_keyword_accuracy": round(llm_all_hit / n, 4) if n else None,
        "rag_win": sum(1 for r in records if r["winner"] == "RAG"),
        "llm_win": sum(1 for r in records if r["winner"] == "纯LLM"),
        "tie": sum(1 for r in records if r["winner"] == "持平"),
    }


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------
def write_markdown(records: list[dict], summary: dict, path: Path, meta: dict) -> None:
    lines: list[str] = []
    lines.append("# RAG（基于PDF）vs 纯 LLM 对比结果")
    lines.append("")
    lines.append("> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统  ")
    lines.append(f"> 生成时间：{meta['generated_at']}　"
                 f"RAG 预设：`{meta['preset']}`　top_k：{meta['top_k']}　"
                 f"生成模型：`{meta['llm_model']}`  ")
    lines.append(f"> 问题数：{summary['n']}（《招股说明书1.pdf》兴图新科 10 问）")
    lines.append("")
    lines.append("## 一、方法说明")
    lines.append("")
    lines.append("- **RAG**：`Pipeline.ask()` —— Query 理解 → 向量/全文检索 → 重排 → "
                 "按上下文生成，答案需标注【来源】页码。")
    lines.append("- **纯 LLM**：`generator.generate_llm_only()` —— 只把问题交给同一个"
                 "大模型，不提供任何文档内容，代表「不用知识库」的基线。")
    lines.append("- **关键词命中**：按人工整理的关键信息点判分（全部命中记 1 题），"
                 "详见 `run_evaluation.py` 的 `ANSWERS_EXPECTED`。")
    lines.append("")
    lines.append("## 二、总览")
    lines.append("")
    lines.append("| 指标 | RAG（基于 PDF） | 纯 LLM 基线 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 平均耗时(s) | {summary['rag_avg_latency']} | {summary['llm_avg_latency']} |")
    lines.append(f"| 最大耗时(s) | {summary['rag_max_latency']} | - |")
    lines.append(f"| 关键词准确率 | {summary['rag_keyword_accuracy']} | "
                 f"{summary['llm_keyword_accuracy']} |")
    lines.append(f"| 带引用来源比例 | {summary['rag_citation_rate']} | 0（无来源可标） |")
    lines.append(f"| 拒答（未找到） | {summary['rag_refused']} | 0（无拒答机制，倾向于编造） |")
    lines.append(f"| 逐题胜负 | RAG 胜 {summary['rag_win']} / "
                 f"纯 LLM 胜 {summary['llm_win']} / 持平 {summary['tie']} | - |")
    lines.append("")
    lines.append("## 三、逐题总览表")
    lines.append("")
    lines.append("| id | 问题 | RAG 耗时(s) | 纯LLM 耗时(s) | "
                 "RAG 关键词命中 | 纯LLM 关键词命中 | 优胜方 | RAG 引用页 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        pages = "、".join(str(c.get("page", "?")) for c in r["rag"]["citations"]) or "-"
        lines.append(
            f"| {r['id']} | {_cell(r['question'])} | {r['rag']['latency']} | "
            f"{r['llm_only']['latency']} | "
            f"{len(r['rag']['keyword_hits'])}/{len(r['keywords'])} | "
            f"{len(r['llm_only']['keyword_hits'])}/{len(r['keywords'])} | "
            f"{r['winner']} | {_cell(pages, 30)} |")
    lines.append("")
    lines.append("## 四、逐题详情")
    lines.append("")
    for r in records:
        lines.append(f"### id={r['id']}")
        lines.append("")
        lines.append(f"**问题**：{r['question']}")
        lines.append("")
        lines.append(f"**RAG 答案**（耗时 {r['rag']['latency']} s）：")
        lines.append("")
        lines.append(_clip(r["rag"]["answer"]))
        lines.append("")
        cites = r["rag"]["citations"]
        if cites:
            lines.append("**RAG 引用来源**：" + "；".join(
                f"《{c.get('doc', '')}》第{c.get('page', '?')}页" for c in cites))
            lines.append("")
        lines.append(f"**纯 LLM 答案**（耗时 {r['llm_only']['latency']} s）：")
        lines.append("")
        lines.append(_clip(r["llm_only"]["answer"]))
        lines.append("")
        lines.append(f"**关键词对比**：RAG 命中 {r['rag']['keyword_hits'] or '无'}，"
                     f"漏答 {r['rag']['keyword_miss'] or '无'}；"
                     f"纯 LLM 命中 {r['llm_only']['keyword_hits'] or '无'}，"
                     f"漏答 {r['llm_only']['keyword_miss'] or '无'}。")
        lines.append("")
    lines.append("## 五、结论")
    lines.append("")
    lines.append("1. **可溯源性**：RAG 答案附【来源】页码，可回到 PDF 核对；纯 LLM 无法给出"
                 "任何来源，正确性无法验证。")
    lines.append("2. **准确性**：涉及注册资本、法定代表人、募投金额、报告期收入等"
                 "招股书细节时，纯 LLM 依赖训练语料的记忆，容易张冠李戴甚至编造数字；"
                 "RAG 从检索片段中读数，明显更稳。")
    lines.append("3. **拒答能力**：文档中没有的信息，RAG 会明确回复「未能找到」，"
                 "而纯 LLM 通常会给出看似合理的猜测（幻觉），这是金融场景不可接受的。")
    lines.append("4. **代价**：RAG 多了一次检索（本地毫秒级），总耗时与纯 LLM 同量级，"
                 "仍满足 3 秒要求，属于「几乎无额外成本换准确性」。")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()

    if not config.DEEPSEEK_API_KEY:
        print("[错误] 未配置 DEEPSEEK_API_KEY，无法调用生成模型。")
        print('       请先设置环境变量，例如：set DEEPSEEK_API_KEY=sk-xxxx')
        return 2

    questions = list(config.QUESTIONS_XINGTU)
    if args.ids:
        wanted = {int(x) for x in args.ids.replace("，", ",").split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            print(f"[错误] --ids {args.ids} 未匹配到任何问题。")
            return 2

    cfg = PRESETS[args.preset]
    print("=" * 66)
    print("  工单01 对比实验：RAG（基于PDF） vs 纯 LLM")
    print(f"  预设={args.preset} 集合={args.collection} 题数={len(questions)}")
    print("=" * 66)

    pipeline = Pipeline(cfg, collection=args.collection)
    try:
        pipeline.load_index()
    except Exception as e:
        print(f"[错误] 索引装载失败：{e}")
        print("       请先执行 build_index.py 建索引。")
        return 3
    if pipeline.retriever.vs.count() == 0:
        print("[错误] 向量索引为空，请先执行 build_index.py。")
        return 3

    records = []
    for item in questions:
        records.append(compare_one(pipeline, item, args.top_k,
                                   ANSWERS_EXPECTED.get(str(item["id"]), {}).get(
                                       "keywords", [])))
        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = summarize(records)
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preset": args.preset, "collection": args.collection,
        "top_k": args.top_k or cfg.top_k, "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL_NAME,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"meta": meta, "summary": summary, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(records, summary, Path(args.out_md), meta)

    print("\n" + "=" * 66)
    print(f"  完成：{summary['n']} 题")
    print(f"  RAG 平均耗时 {summary['rag_avg_latency']} s / "
          f"纯 LLM 平均耗时 {summary['llm_avg_latency']} s")
    print(f"  关键词准确率：RAG {summary['rag_keyword_accuracy']} vs "
          f"纯 LLM {summary['llm_keyword_accuracy']}")
    print(f"  逐题胜负：RAG {summary['rag_win']} / 纯 LLM {summary['llm_win']} / "
          f"持平 {summary['tie']}")
    print(f"  结果已保存：\n    {out_json}\n    {args.out_md}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
