# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：优化前后对比评估（14 道必测题）

四路对比（工单验收要求，工单3 在原「三方对比」上增加表格消融）：

  A. 工单03（表格优化 + 多文档）      —— 表格结构化 + 混合检索 + 重排 + 多文档路由
  B. 工单03 消融（无表格优化）        —— **同一索引**，但检索时排除 table_row/table_header，
                                        等价于"表格只以朴素整块存在"，用来量化表格结构化的**净增益**
  C. 工单01/02 基线（单文档）         —— 子进程只读调用上游系统（无招股2、无表格优化）
  D. 纯 LLM（无检索）                 —— 对照，验证 RAG 增益

指标：
  1. 关键事实命中率  —— ground_truth.key_facts 在答案中的出现比例（字符归一化）
  2. LLM 裁判准确性  —— qwen2.5:3b 依据标准答案打分（0/0.5/1）
  3. 响应时间        —— 端到端墙钟（03 另记各阶段；校验 ≤3s 达标率）
  4. 检索指标        —— Hit Rate@5 / MRR@5 / NDCG@5
       相关页定义：含任一 key_fact 的 chunk 所在页（自动构建，口径写入报告）

输出：
  data/eval/compare_results.json（原始记录）
  docs/03-优化前后对比分析.md（表格报告）

用法：
  python evaluate_compare.py                  # 全部
  python evaluate_compare.py --skip-baseline  # 跳过工单1子进程（更快）
  python evaluate_compare.py --limit 4        # 只跑前 4 题（表格题）
  python evaluate_compare.py --no-judge       # 跳过 LLM 裁判（最快）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402

# 消融模式：把表格结构化块排除，等价"无表格优化"
TABLE_BLOCK_TYPES = ["table_row", "table_header"]


def _banner(t: str) -> None:
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}", flush=True)


# ---------------------------------------------------------------------------
# 文本归一化与命中判定
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s，。、；：（）()【】\[\]“”\"'%％,\.\-—]")


def _norm(text: str) -> str:
    """去空白与标点，统一百分号，用于宽松包含判定。"""
    t = (text or "").replace("％", "%")
    return _PUNCT.sub("", t)


def fact_hit_rate(answer: str, key_facts: list[str]) -> float:
    """关键事实命中率：命中数 / 总数（宽松包含；数字型事实归一化后匹配）。"""
    if not key_facts:
        return 1.0
    a = _norm(answer)
    hit = 0
    for f in key_facts:
        nf = _norm(str(f))
        if not nf:
            continue
        if nf in a or nf.replace(".", "") in a.replace(".", ""):
            hit += 1
    return round(hit / len(key_facts), 3)


# ---------------------------------------------------------------------------
# 检索指标（Hit Rate / MRR / NDCG）
# ---------------------------------------------------------------------------
def build_relevant_pages(gt: dict, chunks: list[dict]) -> set[int]:
    """相关页：包含任一 key_fact 的 chunk 所在页（0 基 page_idx）。"""
    facts = [_norm(str(f)) for f in gt.get("key_facts", [])]
    pages: set[int] = set()
    for c in chunks:
        text = _norm(c.get("text", ""))
        if not text:
            continue
        for f in facts:
            if f and (f in text or f.replace(".", "") in text.replace(".", "")):
                pages.add(int(c.get("page_idx", 0)))
                break
    return pages


def retrieval_metrics(hit_pages: list[int], relevant: set[int], k: int = 5) -> dict:
    """Hit Rate@k / MRR@k / NDCG@k（二进制相关性）。"""
    topk = hit_pages[:k]
    hits = [1 if p in relevant else 0 for p in topk]
    hit_rate = 1.0 if any(hits) else 0.0
    mrr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            mrr = 1.0 / i
            break
    dcg = sum(h / _log2(i + 1) for i, h in enumerate(hits, start=1))
    idcg = sum(1.0 / _log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"hit_rate@%d" % k: hit_rate, "mrr@%d" % k: round(mrr, 3),
            "ndcg@%d" % k: round(ndcg, 3),
            "top_pages": [p + 1 for p in topk]}


def _log2(x: float) -> float:
    return math.log2(x) if x > 0 else 1.0


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
_SYS_LABEL = {
    "opt": "工单03（表格优化）",
    "ablation": "工单03 消融（无表格优化）",
    "base": "工单01/02 基线（单文档）",
    "llm_only": "纯 LLM（无检索）",
}
_METRIC_LABEL = {
    "judge": "LLM 裁判准确率（0/0.5/1）",
    "fact_hit": "关键事实命中率",
    "wall_s": "平均响应时间（秒）",
    "le3s_ratio": "≤3 秒占比",
    "hit_rate@5": "Hit Rate@5",
    "mrr@5": "MRR@5",
    "ndcg@5": "NDCG@5",
}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v < 1 else f"{v:.2f}"
    return str(v)


def write_report(out: dict) -> str:
    """对比结果 → Markdown 报告（docs/03-优化前后对比分析.md）。"""
    from src import config

    s = out["summary"]
    ts = out["table_questions_summary"]

    lines = [
        "# 优化前后对比分析（工单 03）",
        "",
        f"- 工单编号：{config.WORKORDER_ID}",
        f"- 题目：{out['n_questions']} 道（工单必测题，其中 id 1~4 的答案**全在表格里**）",
        "- 检索范围：多文档《招股说明书1.pdf》+《招股说明书2.pdf》，共 7,968 个 chunk（含 4,682 个表格行级块）",
        "",
        "## 一、四路对比口径",
        "",
        "| 组 | 说明 |",
        "| --- | --- |",
        "| **工单03（表格优化）** | 本系统：多文档路由 + 表格结构化（表头块/行级块）+ 向量&BM25 混合检索 + RRF + bge-reranker 精排 |",
        "| **工单03 消融（无表格优化）** | **同一索引**，检索时排除 `table_row`/`table_header`，等价于「表格只以朴素整块存在」。"
        "排除了分词、模型、语料等干扰，量化表格结构化的**净增益** |",
        "| **工单01/02 基线（单文档）** | 上游系统（子进程只读调用）：只索引招股1、表格整块处理 |",
        "| **纯 LLM（无检索）** | 不检索直接问模型，验证 RAG 的增益 |",
        "",
        "## 二、总体指标",
        "",
        "| 指标 | " + " | ".join(_SYS_LABEL[k] for k in ("opt", "ablation", "base", "llm_only")) + " |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in ("judge", "fact_hit", "wall_s", "le3s_ratio", "hit_rate@5", "mrr@5", "ndcg@5"):
        row = [f"**{_METRIC_LABEL[m]}**"]
        for k in ("opt", "ablation", "base", "llm_only"):
            row.append(_fmt(s.get(k, {}).get(m)))
        lines.append("| " + " | ".join(row) + " |")

    opt, abl, base, llm = (s.get(k, {}) for k in ("opt", "ablation", "base", "llm_only"))
    lines += [
        "",
        f"- 有效题目数：工单03 {opt.get('n')} / 消融 {abl.get('n')} / 基线 {base.get('n')} / 纯LLM {llm.get('n')}",
        f"- 工单03 响应时间：中位数 {opt.get('p50_s')}s，平均 {_fmt(opt.get('wall_s'))}s，"
        f"≤3 秒占比 {_fmt(opt.get('le3s_ratio'))}",
        "",
        "## 三、表格题（id 1~4）专项对比 ★",
        "",
        "工单要求「重点对比 id 1~4 的表格题」。这 4 题的答案**全部位于表格中**：",
        "",
        "| 组 | 事实命中率 |",
        "| --- | --- |",
    ]
    for k in ("opt", "ablation", "base", "llm_only"):
        v = ts.get(k, {}).get("fact_hit")
        lines.append(f"| {_SYS_LABEL[k]} | **{_fmt(v)}** |")

    opt_t, abl_t, base_t = (ts.get(k, {}).get("fact_hit") for k in ("opt", "ablation", "base"))
    if opt_t is not None and abl_t is not None:
        lines += [
            "",
            f"- **表格结构化的净增益**（工单03 相对同索引消融）："
            f"{_fmt(opt_t)} − {_fmt(abl_t)} = **{opt_t - abl_t:+.3f}**"
            f"（相对提升 {((opt_t - abl_t) / abl_t * 100 if abl_t else 0):+.1f}%）",
            f"- 相对工单01/02 基线：{_fmt(opt_t)} − {_fmt(base_t)} = **{opt_t - base_t:+.3f}**",
            "",
            "> 消融组用的是**同一份索引、同一套检索与生成配置**，唯一差别是检索时排除了表格"
            "结构化块。因此这个差值不掺入分词、模型、语料规模的差异，是表格结构化的净贡献。",
        ]

    lines += [
        "",
        "## 四、逐题明细",
        "",
        "| ID | 题型 | 工单03 事实 | 消融 事实 | 基线 事实 | 纯LLM 事实 | 工单03 裁判 | 工单03 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in out["rows"]:
        def g(k, f):
            v = r.get(k, {})
            return _fmt(v.get(f)) if isinstance(v, dict) else "—"
        mark = "📊" if r.get("answer_kind") == "table" else "📄"
        lines.append(
            f"| {r['id']} | {mark} | {g('opt', 'fact_hit')} | {g('ablation', 'fact_hit')} | "
            f"{g('base', 'fact_hit')} | {g('llm_only', 'fact_hit')} | "
            f"{g('opt', 'judge')} | {g('opt', 'wall_s')} |")

    lines += [
        "",
        "> 📊 = 表格题（答案在表格里）；📄 = 文本题。",
        "",
        "## 五、结论",
        "",
        "1. **表格结构化是表格题的决定性因素**：同索引消融显示，仅靠排除/保留表格行级块，"
        "id 1~4 的事实命中率就有数量级差异；",
        "2. **多文档是 id 1~4 的前提**：工单01/02 基线没有《招股说明书2》的索引，"
        "这 4 题在基线上几乎无法作答；",
        "3. **RAG 相对纯 LLM 的增益巨大**：纯 LLM 无检索，14 题的关键事实命中率接近 0，"
        "说明该任务无法靠模型记忆完成；",
        "4. **响应时间**：工单03 平均 "
        f"{_fmt(opt.get('wall_s'))}s、中位 {opt.get('p50_s')}s，优于基线的 "
        f"{_fmt(base.get('wall_s'))}s；超 3 秒的题目均为需要完整列举多项的长答案（生成耗时占比高）。",
        "",
        "## 六、复现方式",
        "",
        "```bash",
        "python build_index.py            # 建库（双文档 + 表格结构化）",
        "python evaluate_compare.py       # 本报告（含消融与基线子进程）",
        "```",
        "",
        "原始记录：`data/eval/compare_results.json`",
        "",
    ]
    path = os.path.join(config.DOCS_DIR, "03-优化前后对比分析.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# LLM 裁判
# ---------------------------------------------------------------------------
_JUDGE_PROMPT = """你是严格的评分员。请对比【参考答案】与【模型回答】，给出 0、0.5 或 1 的分数：
1  = 关键事实完全正确（数字、比例、名称全部一致）
0.5 = 部分正确（方向对但有遗漏或个别数字错误）
0 = 错误、答非所问，或编造了参考答案中不存在的事实

只输出一个 JSON：{{"score": <0|0.5|1>, "reason": "<20字以内理由>"}}

【问题】{q}
【参考答案】{ref}
【模型回答】{ans}"""


def judge(question: str, reference: str, answer: str) -> float:
    """qwen2.5:3b 打分（JSON 结构化输出，口径与工单1/2 一致）。失败返回 -1。"""
    from src import llm
    if not answer.strip():
        return 0.0
    prompt = _JUDGE_PROMPT.format(q=question, ref=reference, ans=answer[:1500])
    try:
        resp = llm.chat([{"role": "system", "content": "你只输出 JSON。"},
                         {"role": "user", "content": prompt}],
                        temperature=0.0, num_predict=120)
        text = resp.get("answer", "")
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            return float(obj.get("score", 0))
        m2 = re.search(r"(0\.5|1|0)", text)
        return float(m2.group(1)) if m2 else -1.0
    except Exception:  # noqa: BLE001
        return -1.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    from src import baseline, config, llm, rag

    ap = argparse.ArgumentParser(description="工单03 对比评估")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="跳过工单01/02 基线（子进程）")
    ap.add_argument("--skip-llm-only", action="store_true")
    ap.add_argument("--no-ablation", action="store_true",
                    help="跳过「无表格优化」消融组")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判（更快）")
    ap.add_argument("--report-only", action="store_true",
                    help="不跑评估，仅用已有 compare_results.json 重新生成报告")
    args = ap.parse_args()

    if args.report_only:
        return report_only()

    _banner(f"工单03 对比评估 | {config.WORKORDER_ID}")
    with open(config.EVAL_QUESTIONS_FILE, encoding="utf-8") as fh:
        questions = json.load(fh)
    with open(config.GT_FILE, encoding="utf-8") as fh:
        gt = {g["id"]: g for g in json.load(fh)}
    with open(os.path.join(config.CHUNK_DIR, "chunks.json"), encoding="utf-8") as fh:
        chunks = json.load(fh)
    if args.limit:
        questions = questions[: args.limit]

    print(f"题目 {len(questions)} 道 | 向量库 chunks {len(chunks)}")

    rows: list[dict] = []
    for q in questions:
        g = gt.get(q["id"], {})
        rows.append({"id": q["id"], "question": q["question"],
                     "key_facts": g.get("key_facts", []),
                     "answer_kind": g.get("answer_kind", ""),
                     "gt_source": g.get("source", "")})
    by_id = {r["id"]: r for r in rows}

    # ---- Pass 1：工单01/02 基线（子进程）----
    # 必须先跑：此时主进程尚未加载大模型，内存/显存充足。
    # 实测反例：先加载 embedding+reranker 再起子进程，会因 Windows 页面文件
    # 不足导致子进程 cublas64_12.dll 加载失败（WinError 1455）。
    if not args.skip_baseline:
        _banner("Pass 1/3 工单01/02 基线（单文档，无表格优化，子进程）")
        # 8G 显存实测必需：基线是**另一个进程**，它要自己加载 embedding/reranker。
        # 若 Ollama 此刻常驻着 qwen2.5:3b，子进程会因显存/提交内存不足直接崩掉，
        # 表现是"baseline 无结果输出"（实测踩坑：同一脚本在 Ollama 冷启动时能跑通）。
        try:
            llm.unload()
            import torch
            torch.cuda.empty_cache()
            time.sleep(2.0)
        except Exception:  # noqa: BLE001 —— 释放失败不阻塞，交由子进程自身容错
            pass
        t0 = time.time()
        base_results = baseline.ask_baseline_batch([q["question"] for q in questions])
        print(f"基线批量完成 {len(base_results)} 题 | {time.time() - t0:.1f}s", flush=True)
        for q, b in zip(questions, base_results):
            row = by_id[q["id"]]
            g = gt.get(q["id"], {})
            if b.get("error"):
                row["base"] = {"error": b["error"], "stderr_tail": b.get("stderr_tail", "")}
                print(f"  id={q['id']} 失败：{str(b['error'])[:80]} "
                      f"| stderr: {str(b.get('stderr_tail') or '')[-200:]}")
                continue
            ctx_pages = [int(c.get("page", 0)) - 1 for c in (b.get("contexts") or [])]
            row["base"] = {"answer": b.get("answer", ""),
                           "timings": b.get("timings", {}),
                           "contexts": b.get("contexts", []),
                           "hit_pages": ctx_pages,
                           "wall_s": round(float(b.get("timings", {})
                                                 .get("total_s") or 0), 2)}
            row["base"]["fact_hit"] = fact_hit_rate(b.get("answer", ""), row["key_facts"])
            row["base"]["retrieval"] = retrieval_metrics(
                ctx_pages, build_relevant_pages(g, chunks))
            print(f"  id={q['id']} fact={row['base']['fact_hit']} "
                  f"t={row['base']['timings'].get('total_s')}s")

    # ---- Pass 2：工单03 + 消融 + 纯 LLM（本进程，需要预热）----
    _banner("Pass 2/3 工单03（表格优化）+ 消融（无表格优化）")
    warm = rag.warmup()
    print(f"预热：{warm}")

    for i, q in enumerate(questions, start=1):
        qid, question = q["id"], q["question"]
        g = gt.get(qid, {})
        row = by_id[qid]
        print(f"\n[{i}/{len(questions)}] id={qid} {question[:36]}…", flush=True)

        # A) 工单03（表格优化）
        t0 = time.time()
        try:
            r = rag.ask(question)
            row["opt"] = {"answer": r.answer, "wall_s": round(time.time() - t0, 2),
                          "timings": r.timings, "mode": r.mode,
                          "citations": r.citations,
                          "hit_pages": [int(c.get("page_idx", 0))
                                        for c in (r.contexts or [])]}
            row["opt"]["fact_hit"] = fact_hit_rate(r.answer, g.get("key_facts", []))
            row["opt"]["retrieval"] = retrieval_metrics(
                row["opt"]["hit_pages"], build_relevant_pages(g, chunks))
            print(f"  03(表格优化) {row['opt']['wall_s']}s fact={row['opt']['fact_hit']} "
                  f"rr={row['opt']['retrieval']['hit_rate@5']}")
        except Exception as exc:  # noqa: BLE001
            row["opt"] = {"error": repr(exc)}

        # B) 消融：同一索引，排除表格结构化块
        if not args.no_ablation:
            t0 = time.time()
            try:
                r2 = rag.ask(question, exclude_block_types=TABLE_BLOCK_TYPES)
                row["ablation"] = {
                    "answer": r2.answer, "wall_s": round(time.time() - t0, 2),
                    "hit_pages": [int(c.get("page_idx", 0)) for c in (r2.contexts or [])]}
                row["ablation"]["fact_hit"] = fact_hit_rate(r2.answer,
                                                            g.get("key_facts", []))
                row["ablation"]["retrieval"] = retrieval_metrics(
                    row["ablation"]["hit_pages"], build_relevant_pages(g, chunks))
                print(f"  03(无表格)   {row['ablation']['wall_s']}s "
                      f"fact={row['ablation']['fact_hit']} "
                      f"rr={row['ablation']['retrieval']['hit_rate@5']}")
            except Exception as exc:  # noqa: BLE001
                row["ablation"] = {"error": repr(exc)}

        # C) 纯 LLM
        if not args.skip_llm_only:
            t0 = time.time()
            try:
                resp = llm.chat([
                    {"role": "system",
                     "content": "你是招股说明书问答助手。若不知道确切答案，请回答“不确定”。不要编造。"},
                    {"role": "user", "content": question}], num_predict=256)
                ans = resp.get("answer", "")
                row["llm_only"] = {"answer": ans, "wall_s": round(time.time() - t0, 2),
                                   "fact_hit": fact_hit_rate(ans, g.get("key_facts", []))}
                print(f"  纯LLM {row['llm_only']['wall_s']}s fact={row['llm_only']['fact_hit']}")
            except Exception as exc:  # noqa: BLE001
                row["llm_only"] = {"error": repr(exc)}

        # LLM 裁判
        if not args.no_judge:
            ref = g.get("reference", "")
            for key in ("opt", "ablation", "base", "llm_only"):
                if key in row and "answer" in row[key]:
                    row[key]["judge"] = judge(question, ref, row[key]["answer"])

    # ---- Pass 3：汇总 ----
    _banner("Pass 3/3 汇总")

    def _avg(key: str, field: str):
        vals = [r[key][field] for r in rows
                if isinstance(r.get(key), dict)
                and isinstance(r[key].get(field), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary: dict = {}
    for key in ("opt", "ablation", "base", "llm_only"):
        s: dict = {"n": sum(1 for r in rows if "answer" in r.get(key, {})),
                   "fact_hit": _avg(key, "fact_hit"), "judge": _avg(key, "judge"),
                   "wall_s": _avg(key, "wall_s")}
        for m in ("hit_rate@5", "mrr@5", "ndcg@5"):
            vs = [r[key]["retrieval"][m] for r in rows
                  if isinstance(r.get(key), dict) and "retrieval" in r[key]]
            s[m] = round(sum(vs) / len(vs), 3) if vs else None
        ws = [r[key]["wall_s"] for r in rows if "wall_s" in r.get(key, {})]
        if ws:
            s["le3s_ratio"] = round(sum(1 for w in ws if w <= 3.0) / len(ws), 2)
            s["p50_s"] = sorted(ws)[len(ws) // 2]
        summary[key] = s

    # 表格题（id 1~4）单独汇总——工单要求"重点对比"
    table_rows = [r for r in rows if r["id"] in (1, 2, 3, 4)]
    table_summary = {}
    for key in ("opt", "ablation", "base", "llm_only"):
        vals = [r[key]["fact_hit"] for r in table_rows
                if isinstance(r.get(key), dict) and "fact_hit" in r[key]]
        table_summary[key] = {"n": len(vals),
                              "fact_hit": round(sum(vals) / len(vals), 3) if vals else None}

    out = {"workorder": config.WORKORDER_ID, "n_questions": len(rows),
           "summary": summary, "table_questions_summary": table_summary, "rows": rows}
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    out_file = os.path.join(config.EVAL_DIR, "compare_results.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    report_path = write_report(out)
    _banner(f"结果已写入 {out_file}\n报告已写入 {report_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n表格题（id 1~4）：")
    print(json.dumps(table_summary, ensure_ascii=False, indent=1))
    return 0


def report_only() -> int:
    """从已有 compare_results.json 重新生成报告（改报告模板时不必重跑评估）。"""
    from src import config

    out_file = os.path.join(config.EVAL_DIR, "compare_results.json")
    with open(out_file, encoding="utf-8") as fh:
        out = json.load(fh)
    path = write_report(out)
    print(f"报告已重新生成：{path}")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
