# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：优化前后对比评估（10 道必测题）

三方对比（工单验收要求）：
  A. 优化后（工单02）—— 父子块 + 混合检索 + 重排 + 查询改写
  B. 优化前（工单01）—— 基线系统（子进程只读调用）
  C. 纯 LLM（无检索）—— 对照，验证 RAG 增益

指标：
  1. 关键事实命中率   —— ground_truth.key_facts 在答案中的出现比例（字符归一化）
  2. LLM 裁判准确性   —— qwen2.5:3b 依据标准答案打分（0/0.5/1），与工单1同口径
  3. 响应时间         —— 端到端墙钟（优化后另记首 token 与各阶段）
  4. 检索指标         —— Hit Rate@5 / MRR@5 / NDCG@5
       相关页定义：含任一 key_fact 的 chunk 所在页（自动构建，口径写入报告）

输出：
  data/eval/compare_results.json（原始记录）
  docs/03-优化前后对比分析.md（表格报告）

用法：
  python evaluate_compare.py                # 全部（含工单1子进程）
  python evaluate_compare.py --skip-baseline
  python evaluate_compare.py --limit 3
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


# ---------------------------------------------------------------------------
# 文本归一化与命中判定
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s，。、；：（）()【】\[\]“”\"'%％,\.\-—]")


def _norm(text: str) -> str:
    """去空白与标点，统一百分号，用于宽松包含判定。"""
    t = (text or "").replace("％", "%")
    return _PUNCT.sub("", t)


def fact_hit_rate(answer: str, key_facts: list[str]) -> float:
    """关键事实命中率：命中数 / 总数（宽松包含；数字型事实两者归一化后匹配）。"""
    if not key_facts:
        return 1.0
    a = _norm(answer)
    hit = 0
    for f in key_facts:
        nf = _norm(str(f))
        if not nf:
            continue
        if nf in a:
            hit += 1
            continue
        # 千分位差异（7,360.00 vs 7360.00）已在 _norm 中去除逗号
        if nf.replace(".", "") in a.replace(".", ""):
            hit += 1
    return round(hit / len(key_facts), 3)


# ---------------------------------------------------------------------------
# 检索指标（Hit Rate / MRR / NDCG）
# ---------------------------------------------------------------------------
def build_relevant_pages(gt: dict, chunks: list[dict]) -> set[int]:
    """相关页：包含任一 key_fact 的 chunk 所在页（1 基页码）。"""
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
            "ndcg@%d" % k: round(ndcg, 3), "top_pages": [p + 1 for p in topk]}


def _log2(x: float) -> float:
    import math
    return math.log2(x) if x > 0 else 1.0


# ---------------------------------------------------------------------------
# LLM 裁判（与工单1同口径）
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
    """qwen2.5:3b 打分（JSON 结构化输出，口径与工单1 evaluate.py 一致）。

    1=完全正确，0.5=部分正确，0=错误/缺失；失败返回 -1。
    实测：让模型"只输出一个数字"会输出噪声（正确回答被判 0）；
    改为 JSON 格式约束后输出稳定（与工单1 的 JUDGE_PROMPT 同源）。
    """
    import json as _json
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
            obj = _json.loads(m.group(0))
            return float(obj.get("score", 0))
        m2 = re.search(r"(0\.5|1|0)", text)
        return float(m2.group(1)) if m2 else -1.0
    except Exception:  # noqa: BLE001
        return -1.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    from src import baseline, config, rag, llm

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-llm-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判（更快）")
    args = ap.parse_args()

    _banner(f"工单02 对比评估 | {config.WORKORDER_ID}")
    with open(config.EVAL_QUESTIONS_FILE, encoding="utf-8") as fh:
        questions = json.load(fh)
    with open(config.BASELINE_GT_FILE, encoding="utf-8") as fh:
        gt = {g["id"]: g for g in json.load(fh)}
    with open(os.path.join(config.CHUNK_DIR, "chunks.json"), encoding="utf-8") as fh:
        chunks = json.load(fh)
    if args.limit:
        questions = questions[: args.limit]

    print(f"题目 {len(questions)} 道 | 优化后 chunks {len(chunks)}")
    rows: list[dict] = []
    for q in questions:
        g = gt.get(q["id"], {})
        rows.append({"id": q["id"], "question": q["question"],
                     "key_facts": g.get("key_facts", [])})
    by_id = {r["id"]: r for r in rows}

    # ---- Pass 1：优化前（工单1，子进程）----
    # 必须先跑：此时主进程尚未加载大模型，内存/显存充足。
    # 实测反例：先加载 embedding+reranker 再起子进程，会因 Windows 页面文件
    # 不足导致子进程 cublas64_12.dll 加载失败（WinError 1455）。
    if not args.skip_baseline:
        _banner("Pass 1/2 优化前系统（工单1，单次子进程批量）")
        t0 = time.time()
        base_results = baseline.ask_baseline_batch([q["question"] for q in questions])
        print(f"基线批量完成 {len(base_results)} 题 | {time.time() - t0:.1f}s", flush=True)
        for q, b in zip(questions, base_results):
            row = by_id[q["id"]]
            g = gt.get(q["id"], {})
            if b.get("error"):
                row["base"] = {"error": b["error"]}
                print(f"  id={q['id']} 失败：{str(b['error'])[:80]}")
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

    # ---- Pass 2：优化后 + 纯 LLM（本进程，需要预热）----
    _banner("Pass 2/2 优化后系统 + 纯 LLM 对照")
    warm = rag.warmup()
    print(f"预热：{warm}")

    for i, q in enumerate(questions, start=1):
        qid, question = q["id"], q["question"]
        g = gt.get(qid, {})
        row = by_id[qid]
        print(f"\n[{i}/{len(questions)}] id={qid} {question[:36]}…", flush=True)

        # A) 优化后
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
            print(f"  优化后 {row['opt']['wall_s']}s fact={row['opt']['fact_hit']} "
                  f"rr={row['opt']['retrieval']['hit_rate@5']}")
        except Exception as exc:  # noqa: BLE001
            row["opt"] = {"error": repr(exc)}

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

        # LLM 裁判（对 A/B/C 的答案）
        if not args.no_judge:
            ref = g.get("reference", "")
            for key in ("opt", "base", "llm_only"):
                if key in row and "answer" in row[key]:
                    row[key]["judge"] = judge(question, ref, row[key]["answer"])

    # ---- 汇总 ----
    def _avg(rows, key, field):
        vals = [r[key][field] for r in rows
                if key in r and isinstance(r[key], dict) and isinstance(
                    r[key].get(field), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {}
    for key in ("opt", "base", "llm_only"):
        summary[key] = {
            "n": sum(1 for r in rows if key in r and "answer" in r.get(key, {})),
            "fact_hit": _avg(rows, key, "fact_hit"),
            "judge": _avg(rows, key, "judge"),
            "wall_s": _avg(rows, key, "wall_s"),
            "hit_rate@5": _avg(rows, key, "retrieval") if False else None,
        }
        rrs = [r[key]["retrieval"]["hit_rate@5"] for r in rows
               if key in r and isinstance(r.get(key), dict) and "retrieval" in r[key]]
        mrrs = [r[key]["retrieval"]["mrr@5"] for r in rows
                if key in r and isinstance(r.get(key), dict) and "retrieval" in r[key]]
        ndcgs = [r[key]["retrieval"]["ndcg@5"] for r in rows
                 if key in r and isinstance(r.get(key), dict) and "retrieval" in r[key]]
        if rrs:
            summary[key]["hit_rate@5"] = round(sum(rrs) / len(rrs), 3)
            summary[key]["mrr@5"] = round(sum(mrrs) / len(mrrs), 3)
            summary[key]["ndcg@5"] = round(sum(ndcgs) / len(ndcgs), 3)
        # 优化后：≤3s 占比
        if key == "opt":
            ws = [r[key]["wall_s"] for r in rows if "wall_s" in r.get(key, {})]
            if ws:
                summary[key]["le3s_ratio"] = round(
                    sum(1 for w in ws if w <= 3.0) / len(ws), 2)
                summary[key]["p50_s"] = sorted(ws)[len(ws) // 2]

    out = {"workorder": config.WORKORDER_ID, "n_questions": len(rows),
           "summary": summary, "rows": rows}
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    out_file = os.path.join(config.EVAL_DIR, "compare_results.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    _banner(f"结果已写入 {out_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
