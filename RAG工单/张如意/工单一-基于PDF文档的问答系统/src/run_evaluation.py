# -*- coding: utf-8 -*-
"""
工单01 RAG 评估脚本：对工单 10 个问题跑 RAGAS 四大指标 + 关键词准确率
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

流程：
  1. 装载索引与流水线，对 10 个问题逐一执行 RAG 问答（记录答案/引用/耗时）
  2. 调用 rag_core.evaluate 计算 RAGAS 四大指标：
       Faithfulness 忠实度 / Answer Relevancy 答案相关性 /
       Context Precision 上下文精度 / Context Recall 上下文召回
  3. 计算关键词准确率（evaluate.keyword_accuracy，全部关键信息点命中才算答对）
  4. 结果保存为 results/evaluation.json 与 results/evaluation.md

用法：
    python "工单01-基于PDF文档的问答系统/src/run_evaluation.py"
    python .../src/run_evaluation.py --ids 260,95 --top-k 8
    python .../src/run_evaluation.py --metrics faithfulness,answer_relevancy
    python .../src/run_evaluation.py --answers my_answers.json    # 覆盖参考答案
    python .../src/run_evaluation.py --ragas                      # 已装官方 ragas 时走官方实现
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# --- 让脚本可以独立运行：把项目根目录（工单作业/）加入模块搜索路径 ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config, evaluate                 # noqa: E402
from rag_core.pipeline import PRESETS, Pipeline       # noqa: E402

CASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAGAS_METRICS = ["faithfulness", "answer_relevancy",
                 "context_precision", "context_recall"]
LATENCY_BUDGET_S = 3.0


# ---------------------------------------------------------------------------
# 参考答案（人工整理初稿；可用 --answers 指定 JSON 覆盖后重跑）
# ---------------------------------------------------------------------------
# 说明：
#   1. ground_truth 供 Context Recall / Answer Correctness 使用；
#      keywords 供「关键词准确率」自动判分使用（全部命中才算答对）。
#   2. 下表由人工从《招股说明书1.pdf》整理成初稿，若与实际文档口径不一致，
#      请查看 evaluation.md 的「关键词命中明细」定位漏答项后修正，或用
#      --answers answers.json 覆盖（格式见下方 load_answers 的文档字符串）。
#   3. 数字统一写常见口径（如 "5,520" 可同时匹配 "5,520万元"/"5,520.00万元"）。
ANSWERS_EXPECTED: dict[str, dict] = {
    "260": {
        "ground_truth": (
            "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别为"
            "2016年度、2017年度、2018年度及2019年1-6月的对应金额（单位：万元），"
            "军用领域收入是公司主营业务收入的主要来源，具体金额见招股说明书"
            "「主营业务收入构成」相关表格。"),
        "keywords": ["2016", "2017", "2018"],
    },
    "95": {
        "ground_truth": (
            "武汉兴图新科电子股份有限公司参与制定了相关技术标准"
            "（国家军用标准），体现其在视频指挥/指挥信息系统领域的技术积累。"),
        "keywords": ["标准"],
    },
    "33": {
        "ground_truth": (
            "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占"
            "主营业务收入的比重分别为2016年度、2017年度、2018年度及"
            "2019年1-6月各期比例（%），军用领域收入占比整体保持在较高水平。"),
        "keywords": ["2016", "2017", "2018"],
    },
    "34": {
        "ground_truth": (
            "电子信息行业的上游主要涉及电子元器件、芯片（集成电路）、"
            "印制电路板（PCB）、结构件、计算机及外设等企业。"),
        "keywords": ["上游"],
    },
    "957": {
        "ground_truth": (
            "武汉兴图新科电子股份有限公司已经成为军用视频指挥领域"
            "（国防信息化领域）的重要供应商。"),
        "keywords": ["军用"],
    },
    "793": {
        "ground_truth": (
            "电子信息行业的下游主要包括国防、政府、能源、交通、金融、"
            "教育、医疗等行业。"),
        "keywords": ["下游"],
    },
    "795": {
        "ground_truth": (
            "武汉兴图新科电子股份有限公司参与的相关工程（视频指挥/"
            "指挥信息系统方向）荣获国家科技进步一等奖。"),
        "keywords": ["一等奖"],
    },
    "543": {
        "ground_truth": "武汉兴图新科电子股份有限公司的注册资本为5,520万元。",
        "keywords": ["5,520"],
    },
    "531": {
        "ground_truth": "武汉兴图新科电子股份有限公司的法定代表人是程家明。",
        "keywords": ["程家明"],
    },
    "207": {
        "ground_truth": (
            "武汉兴图新科电子股份有限公司计划使用本次发行募集资金中的"
            "一部分用于补充流动资金，具体金额见招股说明书「募集资金运用」章节。"),
        "keywords": ["补充流动资金"],
    },
}


def load_answers(path: str | None) -> dict:
    """
    读取参考答案表。

    覆盖文件格式（JSON，UTF-8）：
        {
          "543": {"ground_truth": "……", "keywords": ["5,520"]},
          "531": {"keywords": ["程家明"]}
        }
    未在文件中出现的题号沿用内置初稿。
    """
    answers = {k: dict(v) for k, v in ANSWERS_EXPECTED.items()}
    if not path:
        return answers
    p = Path(path)
    if not p.exists():
        print(f"[警告] 参考答案文件不存在：{p}，改用内置初稿。")
        return answers
    data = json.loads(p.read_text(encoding="utf-8"))
    for qid, item in data.items():
        answers.setdefault(str(qid), {})
        answers[str(qid)].update(item)
    print(f"[信息] 已从 {p} 载入 {len(data)} 条参考答案覆盖。")
    return answers


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="工单01：RAGAS 四大指标 + 关键词准确率评估",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="wo01_baseline", choices=sorted(PRESETS),
                   help="被评估的流水线预设，默认 wo01_baseline")
    p.add_argument("--collection", default="prospectus", help="向量库集合名")
    p.add_argument("--top-k", type=int, default=None, help="检索片段数，默认取预设")
    p.add_argument("--ids", default=None,
                   help="只评估指定题号，逗号分隔，如 260,95,33")
    p.add_argument("--metrics", default=",".join(RAGAS_METRICS),
                   help="评估指标子集，默认四大指标；可加 answer_correctness")
    p.add_argument("--answers", default=None, help="参考答案覆盖文件（JSON）")
    p.add_argument("--ragas", action="store_true",
                   help="已安装官方 ragas 包时改走官方实现")
    p.add_argument("--out-json", default=str(RESULTS_DIR / "evaluation.json"))
    p.add_argument("--out-md", default=str(RESULTS_DIR / "evaluation.md"))
    p.add_argument("--debug", action="store_true", help="打印异常堆栈")
    return p


# ---------------------------------------------------------------------------
# 采集 RAG 回答
# ---------------------------------------------------------------------------
def collect_records(pipeline: Pipeline, questions: list[dict], top_k: int | None,
                    answers: dict) -> list[evaluate.EvalRecord]:
    """对每个问题跑一次 RAG，组装 EvalRecord（失败不中断，记入错误提示）。"""
    records: list[evaluate.EvalRecord] = []
    for i, item in enumerate(questions, 1):
        qid, question = item["id"], item["question"]
        gt = answers.get(str(qid), {})
        print(f"\n[{i}/{len(questions)}] 检索并生成 id={qid} …")
        t0 = time.perf_counter()
        try:
            result = pipeline.ask(question, top_k=top_k, return_trace=True)
            answer = result.get("answer", "")
            docs = result.get("docs", [])
        except Exception as e:                   # 容错：单题失败不影响整轮评估
            answer, docs = f"（本题问答失败：{e}）", []
            print(f"  [警告] id={qid} 问答失败：{e}")
        latency = time.perf_counter() - t0

        rec = evaluate.EvalRecord(
            qid=qid, question=question, answer=answer,
            ground_truth=gt.get("ground_truth", ""),
            contexts=[d.get("text", "") for d in docs],
            reference_doc=(docs[0].get("doc", "") if docs else ""),
            retrieved_docs=[d.get("doc", "") for d in docs],
            retrieved_pages=[d.get("page", 0) for d in docs],
            latency=latency,
        )
        print(f"  答案 {len(answer)} 字，命中片段 {len(docs)} 条，"
              f"耗时 {latency * 1000:.0f} ms")
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------
def _clip(text: str, n: int = 1600) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f"\n…（已截断，完整内容见 JSON）"


def write_markdown(records: list[evaluate.EvalRecord], summary: dict,
                   keyword_result: dict, path: Path, meta: dict) -> None:
    lines: list[str] = []
    lines.append("# 工单01 问答系统评估报告")
    lines.append("")
    lines.append("> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统  ")
    lines.append(f"> 生成时间：{meta['generated_at']}  ")
    lines.append(f"> 流水线预设：`{meta['preset']}`　集合：`{meta['collection']}`　"
                 f"top_k：{meta['top_k']}　生成模型：`{meta['llm_model']}`  ")
    lines.append(f"> 样本数：{summary.get('n', 0)}　评估器：`{summary.get('evaluator', 'builtin')}`")
    lines.append("")

    lines.append("## 一、评估口径")
    lines.append("")
    lines.append("| 指标 | 含义 |")
    lines.append("| --- | --- |")
    lines.append("| Faithfulness 忠实度 | 答案中的论断能被检索上下文支撑的比例，抗幻觉核心指标 |")
    lines.append("| Answer Relevancy 答案相关性 | 由答案反推问题与原问题的平均语义相似度，衡量是否切题 |")
    lines.append("| Context Precision 上下文精度 | 有用片段是否排在检索结果前列 |")
    lines.append("| Context Recall 上下文召回 | 参考答案的信息有多少能从检索上下文中找到 |")
    lines.append("| 关键词准确率 | 预设关键信息点全部命中才算答对（人工判分口径的自动化近似） |")
    lines.append("")

    lines.append("## 二、总体结果")
    lines.append("")
    lines.append("```text")
    lines.append(evaluate.format_summary(summary))
    lines.append("```")
    lines.append("")
    acc = summary.get("keyword_accuracy")
    if acc is not None:
        lines.append(f"关键词准确率：**{acc * 100:.1f}%** "
                     f"（{summary.get('keyword_correct', 0)}/"
                     f"{summary.get('keyword_total', 0)} 题全部命中）")
    lat = summary.get("latency_avg")
    if lat is not None:
        ok = "满足" if lat <= LATENCY_BUDGET_S else "超出"
        lines.append("")
        lines.append(f"响应时间：平均 **{lat:.3f} s**、最大 {summary.get('latency_max', 0):.3f} s，"
                     f"{ok}工单「不超过 3 秒」要求"
                     f"（注：该耗时含 LLM 生成，首次运行无缓存时更有代表性）。")
    lines.append("")

    lines.append("## 三、关键词准确率明细")
    lines.append("")
    lines.append("| 题号 | 是否答对 | 命中关键词 | 漏答关键词 |")
    lines.append("| --- | --- | --- | --- |")
    for d in keyword_result.get("details", []):
        lines.append(f"| {d['id']} | {'是' if d['正确'] else '否'} | "
                     f"{'、'.join(d['命中']) or '-'} | {'、'.join(d['漏答']) or '-'} |")
    lines.append("")
    lines.append("> 提示：若某题「漏答」的关键词确实出现在正确答案中，说明该题答错；"
                 "若关键词本身与文档口径不符，请用 `--answers` 覆盖参考答案表后重跑。")
    lines.append("")

    lines.append("## 四、逐题明细")
    lines.append("")
    for r in records:
        lines.append(f"### id={r.qid}")
        lines.append("")
        lines.append(f"**问题**：{r.question}")
        lines.append("")
        lines.append(f"**RAG 答案**：")
        lines.append("")
        lines.append(_clip(r.answer))
        lines.append("")
        lines.append(f"**参考答案（初稿）**：{r.ground_truth}")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        for k, v in r.metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif v is not None:
                lines.append(f"| {k} | {v} |")
        lines.append(f"| latency(s) | {r.latency:.3f} |")
        lines.append("")
        pages = "、".join(str(p) for p in r.retrieved_pages) or "-"
        lines.append(f"检索页码：{pages}")
        lines.append("")

    lines.append("## 五、结论")
    lines.append("")
    lines.append("- 忠实度与上下文召回反映「检索是否找全、生成是否照文档说话」，"
                 "是工单01 验收「基于 PDF 内容返回答案」的直接证据。")
    lines.append("- 关键词准确率对应工单要求的「问答准确率」，逐题命中情况见第三节。")
    lines.append("- 与纯 LLM 的对比见 `rag_vs_llm.md`：RAG 答案带页码引用、"
                 "可用检索上下文复核，纯 LLM 在招股书细节数字上易出现编造。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()

    if not config.DEEPSEEK_API_KEY:
        print("[错误] 未配置 DEEPSEEK_API_KEY，无法调用生成模型与 LLM 判分。")
        print('       请先设置环境变量，例如：set DEEPSEEK_API_KEY=sk-xxxx')
        return 2

    questions = list(config.QUESTIONS_XINGTU)
    if args.ids:
        wanted = {int(x) for x in args.ids.replace("，", ",").split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            print(f"[错误] --ids {args.ids} 未匹配到任何问题。")
            return 2

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in
               ("faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "answer_correctness")]
    if unknown:
        print(f"[错误] 未知指标：{unknown}")
        return 2

    cfg = PRESETS[args.preset]
    print("=" * 66)
    print("  工单01 评估：RAGAS 四大指标 + 关键词准确率")
    print(f"  预设={args.preset} 指标={metrics} 题数={len(questions)}")
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

    answers = load_answers(args.answers)
    records = collect_records(pipeline, questions, args.top_k, answers)

    print("\n开始计算评估指标（每题都会调用 LLM 判分，请稍候）…")
    try:
        summary = evaluate.evaluate_records(
            records, metrics=metrics, use_ragas=args.ragas, verbose=True)
    except Exception as e:                       # 容错：依赖缺失时降级重试
        print(f"[警告] 指标计算中断：{e}")
        if args.debug:
            traceback.print_exc()
        fallback = [m for m in metrics
                    if m not in ("answer_relevancy", "answer_correctness")]
        if fallback and fallback != metrics:
            print(f"[降级] 跳过依赖向量模型的指标，改用：{fallback}")
            summary = evaluate.evaluate_records(
                records, metrics=fallback, use_ragas=False, verbose=True)
            summary["degraded_from"] = metrics
        else:
            print("       请检查网络、DEEPSEEK_API_KEY 与 sentence-transformers 安装。")
            return 4

    # 关键词判分使用（可能被 --answers 覆盖过的）答案表，而非仅内置初稿
    keywords_map = {qid: item.get("keywords", []) for qid, item in answers.items()}
    keyword_result = evaluate.keyword_accuracy(records, keywords_map)
    summary["keyword_accuracy"] = round(keyword_result["accuracy"], 4)
    summary["keyword_correct"] = keyword_result["correct"]
    summary["keyword_total"] = keyword_result["total"]
    summary["preset"] = args.preset
    summary["collection"] = args.collection
    summary["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_json = Path(args.out_json)
    evaluate.save_report(records, summary, out_json)
    write_markdown(records, summary, keyword_result, Path(args.out_md), {
        "generated_at": summary["generated_at"],
        "preset": args.preset, "collection": args.collection,
        "top_k": args.top_k or cfg.top_k, "llm_model": config.LLM_MODEL,
    })

    print("\n" + evaluate.format_summary(summary))
    print(f"\n关键词准确率：{keyword_result['accuracy'] * 100:.1f}%"
          f"（{keyword_result['correct']}/{keyword_result['total']}）")
    if summary.get("latency_avg"):
        verdict = "满足" if summary["latency_avg"] <= LATENCY_BUDGET_S else "超出"
        print(f"平均响应时间：{summary['latency_avg']:.3f} s（{verdict} 3 秒要求）")
    print(f"\n结果已保存：\n  {out_json}\n  {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
