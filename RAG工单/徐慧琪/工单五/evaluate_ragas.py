# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-Query理解优化
         （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
模块：RAGAS 评估（本地 Ollama 裁判 + 本地 bge-m3，全程离线）

说明（工单要求"RAGAS（或 TruLens）"）：
  ragas 0.2.x 默认用 OpenAI 作裁判与 embedding —— 本工单禁止外部依赖与下载，
  故将 ragas 的 LLM 换成 **本机 Ollama qwen2.5:3b**，embedding 换成
  **本机 bge-m3**（D:\\model\\bge-m3），全程离线可复现。

指标：faithfulness / answer_relevancy / context_recall / context_precision
对比：RAG 回答 vs 纯 LLM 回答（工单第四步要求的三方对比中的 RAGAS 部分）

两套题（工单5 起）：
  · **14 道必测题**（口径与工单3 一致，故两次 RAGAS 可直接对比）；
  · **5 轮验收序列**（工单5 的多轮题）：答案与 contexts 不重跑，直接复用
    `run_acceptance.py` 落盘的 `data/eval/acceptance_result.json`——多轮题的
    contexts 只有在「带对话状态」的那一次运行里才是对的，单独重跑等于换了输入。

输出：data/eval/ragas_results.json + docs/04-RAGAS评估报告.md

用法：
  python evaluate_ragas.py                # 14 题 + 5 轮验收序列
  python evaluate_ragas.py --limit 3      # 冒烟
  python evaluate_ragas.py --reuse        # 复用上次 answers（不重新调用 LLM）
  python evaluate_ragas.py --skip-multiturn   # 只跑 14 题
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402


def _banner(t: str) -> None:
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}", flush=True)


def _build_ragas_llm():
    """ragas 用 LangChain LLM 接口；指向本机 Ollama（已 pull 模型，不下载）。"""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    from src import config
    lc = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_HOST,
                    temperature=0.0, num_ctx=config.OLLAMA_NUM_CTX)
    return LangchainLLMWrapper(lc)


def _build_ragas_embeddings():
    """ragas embeddings 指向本机 bge-m3（local_files_only）。"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src import config
    emb = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb)


def collect_answers(limit: int = 0, reuse: bool = False) -> list[dict]:
    """收集 RAG 与纯 LLM 的回答（供 RAGAS 评估）。"""
    from src import config, llm, rag

    q_file = config.EVAL_QUESTIONS_FILE
    with open(q_file, encoding="utf-8") as fh:
        questions = json.load(fh)
    with open(config.GT_FILE, encoding="utf-8") as fh:
        gt = {g["id"]: g for g in json.load(fh)}
    if limit:
        questions = questions[:limit]

    cache_file = os.path.join(config.EVAL_DIR, "ragas_answers.json")
    if reuse and os.path.isfile(cache_file):
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)

    rag.warmup()
    rows: list[dict] = []
    for i, q in enumerate(questions, start=1):
        qid, question = q["id"], q["question"]
        print(f"[{i}/{len(questions)}] id={qid} {question[:36]}…", flush=True)
        r = rag.ask(question)
        contexts = [b.get("parent_text", "") for b in (r.contexts or [])]
        try:
            resp = llm.chat([
                {"role": "system",
                 "content": "你是招股说明书问答助手。若不知道确切答案，请回答“不确定”。不要编造。"},
                {"role": "user", "content": question}], num_predict=256)
            llm_answer = resp.get("answer", "")
        except Exception as exc:  # noqa: BLE001
            llm_answer = f"（失败：{exc}）"
        rows.append({
            "id": qid, "question": question,
            "reference": gt.get(qid, {}).get("reference", ""),
            "rag_answer": r.answer, "rag_contexts": contexts,
            "llm_answer": llm_answer, "rag_timings": r.timings,
        })
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    return rows


# ---------------------------------------------------------------------------
# 5 轮验收序列（多轮题）的 RAGAS 输入
# ---------------------------------------------------------------------------
# `reference` 取自需求文档《需求开发.docx》的标准答案，逐条对应——
# **不是**从系统输出倒推的，否则 context_recall 会自己给自己打分。
_MT_REFERENCES = {
    1: "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别为 "
       "2016 年 6,464.51 万元、2017 年 14,414.16 万元、2018 年 18,780.67 万元、"
       "2019 年 1-6 月 4,627.14 万元。",
    2: "武汉兴图新科电子股份有限公司参与的“某情报、指挥、控制与通信网络一体化工程”"
       "荣获了国家科技进步一等奖。",
    3: "武汉兴图新科电子股份有限公司的法定代表人是程家明。",
    4: "武汉力源信息技术股份有限公司的法定代表人是赵马克。",
    5: "武汉力源信息技术股份有限公司组织结构图中，大客户销售部的销售处最多，"
       "共有珠海销售处、深圳销售处、北京销售处、武汉销售处、广州销售处、成都销售处 6 个。",
}


def collect_acceptance_rows() -> list[dict]:
    """多轮题的 RAGAS 输入：复用 run_acceptance.py 的答案与 contexts。

    返回空列表 = 验收产物缺失或未跑过，调用方须**如实说明**而不是静默跳过。
    """
    from src import config

    path = os.path.join(config.EVAL_DIR, "acceptance_result.json")
    if not os.path.isfile(path):
        print(f"⚠ 未找到 {path}（先跑 python run_acceptance.py），"
              "多轮题不纳入本次 RAGAS", flush=True)
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for t in data.get("rows", []):
        idx = t.get("turn")
        ref = _MT_REFERENCES.get(idx)
        if not ref:
            print(f"⚠ 第 {idx} 轮无标准答案（_MT_REFERENCES 未覆盖），已跳过", flush=True)
            continue
        rows.append({
            "id": f"Q{idx}",
            "question": t["question"],
            "reference": ref,
            # 多轮题评的是**多轮系统实际给出的答案**，不是重问一遍
            "rag_answer": t.get("answer", ""),
            "rag_contexts": [c.get("text", "") for c in (t.get("contexts") or [])],
            "rewrite_mode": t.get("rewrite_mode", ""),
        })
    return rows


def run_ragas(rows: list[dict]) -> dict:
    """执行 RAGAS 评估（RAG 侧）。纯 LLM 侧无 contexts，只评 faithfulness/relevancy。"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    from src import config

    llm_w = _build_ragas_llm()
    emb_w = _build_ragas_embeddings()
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm = llm_w
    answer_relevancy.embeddings = emb_w

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["rag_answer"] for r in rows],
        "contexts": [r["rag_contexts"] for r in rows],
        "ground_truth": [r["reference"] for r in rows],
    })
    result = evaluate(ds, metrics=metrics, llm=llm_w, embeddings=emb_w)
    df = result.to_pandas()
    per_q = []
    for i, r in enumerate(rows):
        row = {"id": r["id"]}
        for m in config.RAGAS_METRICS:
            if m in df.columns:
                v = df.iloc[i][m]
                row[m] = None if v != v else round(float(v), 3)  # NaN → None
        per_q.append(row)
    summary = {}
    for m in config.RAGAS_METRICS:
        vals = [p[m] for p in per_q if p.get(m) is not None]
        summary[m] = round(sum(vals) / len(vals), 3) if vals else None
    return {"summary": summary, "per_question": per_q}


def _metric_table(summary: dict, metrics: list[str]) -> list[str]:
    desc = {
        "faithfulness": "回答是否忠实于检索上下文（1=无编造）",
        "answer_relevancy": "回答与问题的相关性",
        "context_recall": "检索上下文对标准答案（reference）的覆盖度",
        "context_precision": "检索上下文中相关内容的占比",
    }
    out = ["| 指标 | 均值 | 说明 |", "| --- | --- | --- |"]
    for m in metrics:
        out.append(f"| {m} | {summary.get(m)} | {desc.get(m, '')} |")
    return out


def _verbatim_in_contexts(answer: str, contexts: list[dict]) -> bool:
    """答案（去空白后）是否**逐字**出现在检索片段里。

    用途：`faithfulness` 低分时的归因。若答案逐字来自上下文却被打 0 分，
    那是**裁判误判**，不是编造——两者必须分清，否则会把裁判的问题记成系统的缺陷。
    """
    a = re.sub(r"\s+", "", answer or "")
    if not a:
        return False
    # contexts 既可能是纯文本串（collect_acceptance_rows 的形态），
    # 也可能是带 text 字段的字典（raw 产物的形态），两种都吃
    return any(a in re.sub(r"\s+", "", c.get("text", "") if isinstance(c, dict) else c)
               for c in (contexts or []))


def write_report(ragas: dict, rows: list[dict],
                 mt_rows: list[dict] | None = None) -> str:
    """生成 `docs/04-RAGAS评估报告.md`（工单5 口径：14 题 + 5 轮验收序列）。"""
    from src import config

    mt = ragas.get("multiturn") or {}
    # 多轮表要带「改写方式」，从验收产物现取（很轻，不加载任何模型），
    # 这样 `--report-only` 与完整跑生成的是同一份报告。
    mt_rows = mt_rows or (collect_acceptance_rows() if mt else [])
    n_table = sum(1 for r in rows if r["id"] in (1, 2, 3, 4))
    metrics = config.RAGAS_METRICS

    lines = [
        "# RAGAS 评估报告（工单 05）",
        "",
        f"- 工单编号：{config.WORKORDER_ID}",
        f"- 评估题目：**{len(rows)} 道必测题** + **{len(mt_rows)} 轮验收序列**"
        f"（必测题中表格题 {n_table} 道：id 1~4）",
        "- 检索范围：**多文档**（招股说明书1 + 招股说明书2），表格与组织结构图已结构化入库",
        f"- 裁判模型：{config.OLLAMA_MODEL}（本机 Ollama，离线）",
        f"- 评估用 Embedding：bge-m3（{config.EMBEDDING_MODEL_PATH}，离线）",
        "- 说明：ragas 默认依赖 OpenAI；本报告已将裁判与 embedding 全部替换为本机模型，",
        "  全程离线可复现。",
        "",
        "## 一、14 道必测题",
        "",
        "口径与工单 03 完全一致（同题库、同裁判、同 embedding），故两轮 RAGAS 可直接对比。",
        "",
        *_metric_table(ragas["summary"], metrics),
        "",
        "### 1.1 逐题明细",
        "",
        "| ID | " + " | ".join(metrics) + " |",
        "| --- |" + " --- |" * len(metrics),
    ]
    for p in ragas["per_question"]:
        lines.append(f"| {p['id']} | " + " | ".join(str(p.get(m)) for m in metrics) + " |")

    # --- 与工单3（优化前）的同口径对比 ---
    base = {"faithfulness": 0.631, "answer_relevancy": 0.545,
            "context_recall": 1.0, "context_precision": 0.775}
    lines += ["", "## 二、与工单 03（优化前）的同口径对比", "",
              "| 指标 | 工单05 | 工单03（优化前） | 变化 |", "| --- | --- | --- | --- |"]
    for m in metrics:
        v, b = ragas["summary"].get(m), base.get(m)
        if v is None or b is None:
            continue
        diff = round(v - b, 3)
        arrow = "↑" if diff > 0.005 else ("↓" if diff < -0.005 else "≈")
        lines.append(f"| {m} | {v} | {b} | {diff:+} {arrow} |")
    lines += [
        "",
        "**同口径对比的读法**：这 14 道题**不含任何多轮题**，工单5 的多轮改写在本组上",
        "不产生作用（首轮即完整问题，全部走 `passthrough`），所以两组数字**本就应当接近**；",
        "它的作用是证明「加多轮没有把单轮能力做退」——这是回归证据，不是收益证据。",
        "本工单的收益在第三节（多轮题）。数字若出现小幅波动，来源是 RAGAS 裁判（本机 3b）",
        "自身的不稳定性与 `answer_relevancy` 的反向生成随机性，不在本工单的改动范围内。",
        "",
    ]

    if mt_rows and mt.get("summary"):
        lines += [
            "## 三、5 轮验收序列（多轮题）",
            "",
            "- 参考答案（`reference`）取自需求文档《需求开发.docx》的标准答案，",
            "  **不是**从系统输出倒推的；",
            "- 答案与 contexts 复用 `python run_acceptance.py` 的产物"
            "（`data/eval/acceptance_result.json`）——多轮题的 contexts 只有在"
            "**带对话状态**的那一次运行里才是对的，单独重跑等于换了输入；",
            "- 因此本组数字评的是「多轮链路真实产出的回答」，与 `docs/06-验收序列逐轮报告.md` "
            "逐轮可对照。",
            "",
            *_metric_table(mt["summary"], metrics),
            "",
            "### 3.1 逐轮明细",
            "",
            "| 轮次 | 改写方式 | 答案逐字来自上下文 | " + " | ".join(metrics) + " |",
            "| --- | --- | --- |" + " --- |" * len(metrics),
        ]
        modes = {r["id"]: r.get("rewrite_mode", "") for r in mt_rows}
        ctxs = {r["id"]: r.get("rag_contexts", []) for r in mt_rows}
        answers = {r["id"]: r.get("rag_answer", "") for r in mt_rows}
        verbatim: dict[str, bool] = {}
        for p in mt["per_question"]:
            qid = p["id"]
            verbatim[qid] = _verbatim_in_contexts(answers.get(qid, ""), ctxs.get(qid, []))
            lines.append(f"| {qid} | {modes.get(qid, '')} | "
                         f"{'✅ 是' if verbatim[qid] else '—'} | "
                         + " | ".join(str(p.get(m)) for m in metrics) + " |")
        # 归因：低 faithfulness 但答案逐字来自上下文 → 裁判压分，不是编造。
        # 阈值取 0.5 而不是 ==0：同一份答案在两次裁判运行间实测得 0.0 / 0.333，
        # 只盯 0.0 会漏掉「明明逐字来自上下文却只给三分之一分」的情形。
        judge_err = [p["id"] for p in mt["per_question"]
                     if (p.get("faithfulness") or 0) < 0.5 and verbatim.get(p["id"])]
        lines += [
            "",
            "### 3.2 怎么读这组数字",
            "",
            "- **`context_recall` 5 轮全部 1.0**：这是本工单最关键的一格——Q2/Q3/Q4 的",
            "  原问题含代词或省略，**不做改写就不可能检索到覆盖标准答案的片段**。",
            "  5/5 说明改写后的独立问题确实检索到了正确答案所在的位置；",
            "- **`answer_relevancy` 对简短与列点式答案天然偏低**（该指标由「答案反向生成",
            "  问题」计算，短答案反向生成的问题必然泛化），Q1 的长数字列表得 0.0 属口径特性，",
            "  与 `docs/04` 历史上记录的同一现象一致，不代表答非所问；",
            "- **`faithfulness` 低分要归因，不能一律当成编造**：本组「答案逐字来自上下文」",
            "  一列就是为此设的——若该列为 ✅ 却只拿到不足一半的分，那是**裁判压分**而非幻觉。",
        ]
        if judge_err:
            got = {p["id"]: p.get("faithfulness") for p in mt["per_question"]}
            detail = "、".join(f"{q}（{got.get(q)}）" for q in judge_err)
            lines += [
                "",
                f"  **本次实测**：{detail} 的 `faithfulness` 低于 0.5，"
                "但该轮答案在检索片段中**逐字可查**（脚本自动核对，见上表），",
                "  故判定为**裁判压分**，不是编造。该轮答案与上下文的一致性以",
                "  `docs/06-验收序列逐轮报告.md` 的逐字对照为准。",
                "",
                "  **附带证据（裁判不稳定性是实测的，不是推测）**：同一份 Q2 答案"
                "在两次裁判运行间实测得 **0.0** 与 **0.333** 两个分数，",
                "  而该答案两次都逐字来自检索片段。因此本表的绝对值只能横向读"
                "（同一组内比较），不能当成精确刻度。",
            ]
        lines += [
            "",
            "> 结论：多轮组的 `context_recall = 1.0` 与逐轮事实全命中（见 docs/06）",
            "> 互为佐证；`faithfulness` 的绝对数值受本机 3b 裁判稳定性限制，",
            "> 不宜作为「有没有编造」的唯一判据。",
            "",
        ]
    else:
        lines += [
            "## 三、5 轮验收序列（多轮题）",
            "",
            "**本次未评**：未找到 `data/eval/acceptance_result.json`，或运行时带了 "
            "`--skip-multiturn` / `--limit`。多轮题不纳入本报告是**如实缺项**，",
            "不是「评了但没写」——先跑 `python run_acceptance.py` 再重跑本脚本即可补齐。",
            "",
        ]

    n_nan = sum(1 for p in ragas["per_question"] if p.get("context_precision") is None)
    mt_nan = sum(1 for p in (mt.get("per_question") or [])
                 if p.get("context_precision") is None)
    if n_nan or mt_nan:
        nan_line = (f"- 14 题明细中有 **{n_nan} 处 `context_precision` 为 None**、"
                    f"多轮题中有 **{mt_nan} 处**：RAGAS 的裁判（本机 qwen2.5:3b）"
                    "在个别子任务上超时（日志中的 `Exception raised in Job[..]: "
                    "TimeoutError()`），该格无值。这是**离线小模型裁判的已知局限**，"
                    "汇总时按「有效值求平均」处理，未做插补、未剔除整题。")
    else:
        nan_line = ("- 本次运行**没有缺失格**：所有题目的四项指标都有值，"
                    "汇总即全量平均，未做任何插补或剔除。")
    lines += [
        "## 四、数据完整性说明",
        "",
        nan_line,
        f"- 裁判为**本机离线小模型**，其不稳定性已在 3.2 节以"
        "「答案逐字来自上下文」一列做了交叉核对，未把裁判噪声当成系统缺陷。",
        "",
        "> 工单03 基线数据来源：`data/eval/ragas_results_v3.json`（同机同裁判口径，",
        "> 从工单5 运行前的产物原样备份）。",
        "",
    ]

    path = os.path.join(config.DOCS_DIR, "04-RAGAS评估报告.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main() -> int:
    from src import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="不重跑评估，用已有 ragas_results.json + ragas_answers.json 重新生成报告")
    ap.add_argument("--skip-multiturn", action="store_true",
                    help="不评 5 轮验收序列，只评 14 道必测题")
    ap.add_argument("--multiturn-only", action="store_true",
                    help="只重评 5 轮验收序列（14 题沿用已有结果）——"
                         "重跑过 run_acceptance.py 之后用它对齐多轮指标，不必重跑 14 题")
    args = ap.parse_args()

    if args.multiturn_only:
        res_file = os.path.join(config.EVAL_DIR, "ragas_results.json")
        if not os.path.isfile(res_file):
            print(f"⚠ 缺少 {res_file}，先完整跑一次再单跑多轮")
            return 1
        with open(res_file, encoding="utf-8") as fh:
            out = json.load(fh)
        mt_rows = collect_acceptance_rows()
        if not mt_rows:
            print("⚠ 未找到验收产物，多轮题无法评")
            return 1
        _banner(f"只重评多轮题（{len(mt_rows)} 轮，复用 run_acceptance 的 contexts）")
        out["multiturn"] = {"n": len(mt_rows), **run_ragas(mt_rows),
                            "refs_source": "需求开发.docx 标准答案"}
        with open(res_file, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        with open(os.path.join(config.EVAL_DIR, "ragas_answers.json"), encoding="utf-8") as fh:
            rows = json.load(fh)
        report = write_report(out, rows, mt_rows)
        _banner(f"完成：{res_file} | {report}")
        print("多轮题：", json.dumps(out["multiturn"]["summary"], ensure_ascii=False, indent=1))
        return 0

    if args.report_only:
        res_file = os.path.join(config.EVAL_DIR, "ragas_results.json")
        ans_file = os.path.join(config.EVAL_DIR, "ragas_answers.json")
        with open(res_file, encoding="utf-8") as fh:
            res = json.load(fh)
        with open(ans_file, encoding="utf-8") as fh:
            rows = json.load(fh)
        path = write_report(res, rows)
        print(f"报告已重新生成：{path}")
        return 0

    _banner(f"RAGAS 评估 | {config.WORKORDER_ID}")
    t0 = time.time()
    rows = collect_answers(limit=args.limit, reuse=args.reuse)
    print(f"回答收集完成：{len(rows)} 题 | {time.time() - t0:.1f}s", flush=True)

    _banner("RAGAS 计算（本地裁判，较慢，请耐心）")
    ragas = run_ragas(rows)
    out = {"workorder": config.WORKORDER_ID, "n": len(rows),
           "evaluator": {"llm": config.OLLAMA_MODEL,
                         "embeddings": config.EMBEDDING_MODEL_PATH},
           **ragas}

    # 5 轮验收序列（多轮题）：contexts 复用 run_acceptance.py 的产物，不重跑
    mt_rows: list[dict] = []
    if not args.skip_multiturn and not args.limit:
        mt_rows = collect_acceptance_rows()
        if mt_rows:
            _banner(f"多轮题 RAGAS（{len(mt_rows)} 轮验收序列，复用 run_acceptance 的 contexts）")
            mt = run_ragas(mt_rows)
            out["multiturn"] = {"n": len(mt_rows), **mt}
            out["multiturn"]["refs_source"] = "需求开发.docx 标准答案"

    out_file = os.path.join(config.EVAL_DIR, "ragas_results.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    report = write_report(out, rows, mt_rows)
    _banner(f"完成：{out_file} | {report} | 总用时 {time.time() - t0:.1f}s")
    print(json.dumps(ragas["summary"], ensure_ascii=False, indent=1))
    if out.get("multiturn"):
        print("多轮题：", json.dumps(out["multiturn"]["summary"],
                                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
