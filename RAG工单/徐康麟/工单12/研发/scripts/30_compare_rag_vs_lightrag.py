# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""30_compare_rag_vs_lightrag.py —— 16 题 RAG vs LightRAG 检索结果对比。

流程（与 ``设计/检索与评估方案.md`` §2 一致）：
  ① 载入 ``研发/评估/测试问题集.jsonl``（**16 题，不得另造题**）；
  ② 逐题 × 逐后端（``rag`` / ``lightrag``）串行检索 + 生成；
  ③ 逐题落盘 ``研发/评估/逐题结果.jsonl``（边跑边 append，支持 --resume）；
  ④ 自动判分（无 LLM 判官）：关键词命中率 / 数字命中 / 页命中（**±1 页容差**）；
  ⑤ 汇总 → ``研发/评估/对比汇总.json`` + ``研发/评估/对比对照表.md``；
  ⑥ 对比产物 → ``优化/评估结果/对比汇总.csv`` + ``对比报告.md``。

判分口径（全文本先做**去空白 + 全角括号归一**，依据见方案 §3.3）：
  * ``answer_hit_rate``   = 参考答案关键词在**答案**中的命中比例
  * ``context_hit_rate``  = 参考答案关键词在**检索上下文**中的命中比例
  * ``number_hit_rate``   = 参考答案中的数值/比例在**答案**中的命中比例
  * ``page_hit``          = 任一命中块的页码落在 ``expected_pages`` ± ``eval.page_tolerance`` 内
  * ``answered``          = ``answer_hit_rate ≥ eval.answer_hit_threshold``

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/30_compare_rag_vs_lightrag.py --limit 2 --no-generate   # 冒烟
    pwsh -NoProfile -File E:\\gao6gongdan\\_共享工具\\安全跑.ps1 -WorkDir "E:\\gao6gongdan\\工单12" `
         -Command "pwsh -NoProfile -File run_py.ps1 研发/scripts/30_compare_rag_vs_lightrag.py" -LogName 30_compare
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from lightrag.config import load_config                                    # noqa: E402
from lightrag.logging_util import get_logger, write_csv, write_json        # noqa: E402
from lightrag.qa import QAEngine, extract_numbers                          # noqa: E402
from lightrag.routing import norm_ws                                       # noqa: E402

# 三个后端：rag=扁平基线；lightrag=双层（全量边）；lightrag_strict=双层（仅 constraint_ok 边）
# 后两者构成 captain 要求的「两组边口径」对比，用于量化 t2 遗留的 27.5% 相容率影响
BACKENDS = ("rag", "lightrag", "lightrag_strict")


# ---------------------------------------------------------------------------
def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_one(answer: str, contexts: str, row: dict, hits: list[dict], *, tolerance: int,
              threshold: float) -> dict:
    """按方案口径对单题单后端判分（纯字符串匹配，无 LLM 判官）。"""
    kws = [norm_ws(k) for k in (row.get("reference_keywords") or []) if str(k).strip()]
    ref_answer = norm_ws(str(row.get("reference_answer") or ""))
    a_norm, c_norm = norm_ws(answer), norm_ws(contexts)

    kw_in_ans = [k for k in kws if k in a_norm]
    kw_in_ctx = [k for k in kws if k in c_norm]
    ref_nums = [n for n in extract_numbers(ref_answer) if len(n) >= 3]   # 过滤「1」「2」这类噪声
    num_in_ans = [n for n in ref_nums if n in a_norm]

    expected = [int(p) for p in (row.get("expected_pages") or []) if str(p).strip().isdigit()]
    pages = [int(h.get("page") or 0) for h in hits]
    page_hit = any(abs(p - e) <= tolerance for p in pages for e in expected) if expected else False

    ans_rate = len(kw_in_ans) / len(kws) if kws else 0.0
    ctx_rate = len(kw_in_ctx) / len(kws) if kws else 0.0
    num_rate = len(num_in_ans) / len(ref_nums) if ref_nums else 1.0
    return {
        "answered": bool(ans_rate >= threshold),
        "answer_hit_rate": round(ans_rate, 4),
        "context_hit_rate": round(ctx_rate, 4),
        "number_hit_rate": round(num_rate, 4),
        "page_hit": bool(page_hit),
        "keyword_hits_in_answer": kw_in_ans,
        "keyword_misses_in_answer": [k for k in kws if k not in kw_in_ans],
        "keyword_hits_in_context": kw_in_ctx,
        "number_hits_in_answer": num_in_ans,
        "reference_numbers": ref_nums,
        "expected_pages": expected,
        "retrieved_pages": pages[:8],
        "top_chunk_ids": [h.get("chunk_id") for h in hits[:5]],
        "page_tolerance_used": int(tolerance),
    }


def _write_jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_done_pairs(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((int(row.get("id") or -1), str(row.get("backend") or "")))
    return done


def summarize(rows: list[dict]) -> dict:
    """按后端聚合。"""
    out: dict = {}
    for backend in BACKENDS:
        sub = [r for r in rows if r.get("backend") == backend]
        if not sub:
            continue
        n = len(sub)
        out[backend] = {
            "题数": n,
            "答对数": sum(1 for r in sub if r["score"]["answered"]),
            "答对率": round(sum(1 for r in sub if r["score"]["answered"]) / n, 4),
            "平均答案关键词命中率": round(sum(r["score"]["answer_hit_rate"] for r in sub) / n, 4),
            "平均上下文关键词命中率": round(sum(r["score"]["context_hit_rate"] for r in sub) / n, 4),
            "平均数字命中率": round(sum(r["score"]["number_hit_rate"] for r in sub) / n, 4),
            "页命中数": sum(1 for r in sub if r["score"]["page_hit"]),
            "页命中率": round(sum(1 for r in sub if r["score"]["page_hit"]) / n, 4),
            "平均检索耗时_s": round(sum(r["retrieve_s"] for r in sub) / n, 3),
            "平均生成耗时_s": round(sum(r["generate_s"] for r in sub) / n, 3),
            "平均总耗时_s": round(sum(r["latency_s"] for r in sub) / n, 3),
            "生成失败数": sum(1 for r in sub if r.get("error")),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="16 题 RAG vs LightRAG 对比")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--no-generate", action="store_true", help="只检索不生成（快速验证链路）")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有逐题结果重跑")
    ap.add_argument("--backends", default=",".join(BACKENDS),
                    help="要跑的后端（逗号分隔，默认三个全跑）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("30_compare", cfg, console=True)
    t_all = time.perf_counter()

    q_path = Path(str(cfg.eval.question_file))
    questions = load_questions(q_path)
    if args.limit:
        questions = questions[: args.limit]
    log.info("载入问题集 %s：%d 题", q_path.name, len(questions))

    per_q_path = Path(cfg.path_str("eval_dir")) / "逐题结果.jsonl"
    done = set() if args.no_resume else read_done_pairs(per_q_path)
    if args.no_resume:
        per_q_path.unlink(missing_ok=True)
    log.info("已有逐题结果 %d 条（resume=%s）", len(done), not args.no_resume)

    engine = QAEngine(cfg, logger=log)
    info = engine.load()
    log.info("索引与图谱载入：%s", json.dumps(info, ensure_ascii=False))

    tolerance = int(cfg.eval.page_tolerance)
    threshold = float(cfg.eval.get("answer_hit_threshold") or 0.8)
    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    total_tasks = len(questions) * len(backends)
    i = 0
    try:
        for row in questions:
            qid = int(row["id"])
            for backend in backends:
                i += 1
                if (qid, backend) in done:
                    log.info("跳过（已存在）：id=%s backend=%s", qid, backend)
                    continue
                log.info("【进度】%d/%d  id=%s backend=%s", i, total_tasks, qid, backend)
                res = engine.answer(str(row["question"]), backend=backend, generate=not args.no_generate)
                contexts = "\n".join(str((engine.chunks_by_id.get(h["chunk_id"]) or {}).get("content") or "")
                                     for h in res.hits[: int(cfg.answer.top_k_in_prompt)])
                sc = score_one(res.answer, contexts, row, res.hits,
                               tolerance=tolerance, threshold=threshold)
                record = {
                    "id": qid, "doc": row.get("doc"), "company": row.get("company"),
                    "question": row["question"], "backend": backend,
                    "reference_answer": row.get("reference_answer"),
                    "reference_keywords": row.get("reference_keywords"),
                    "answer": res.answer, "sources": res.sources,
                    "hits": res.hits, "trace": res.trace,
                    "retrieve_s": round(res.retrieve_s, 3), "generate_s": round(res.generate_s, 3),
                    "latency_s": round(res.latency_s, 3),
                    "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                    "error": res.error, "score": sc,
                }
                _write_jsonl_append(per_q_path, record)
                log.info("  → 答对=%s 答案命中=%.2f 上下文命中=%.2f 页命中=%s 延迟=%.1fs",
                         sc["answered"], sc["answer_hit_rate"], sc["context_hit_rate"],
                         sc["page_hit"], res.latency_s)
    finally:
        released = engine.unload()
        log.info("已卸载 Ollama 模型：%s", ", ".join(released) or "（无）")

    rows = [json.loads(l) for l in per_q_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = summarize(rows)
    summary["_meta"] = {
        "问题集": str(q_path), "题数": len({r["id"] for r in rows}),
        "后端": list(BACKENDS), "页号容差": tolerance,
        "答对阈值": threshold, "去空白归一": bool(cfg.eval.get("normalize_whitespace")),
        "top_k": int(cfg.retrieval.top_k), "生成模型": str(cfg.llm.gen_model),
        "小样本图谱": "见 研发/data/graph/graph_meta.json 的「抽样口径」字段",
        "耗时秒": round(time.perf_counter() - t_all, 3),
    }
    write_json(Path(cfg.path_str("eval_dir")) / "对比汇总.json", summary)
    write_json(Path(cfg.path_str("optimize_dir")) / "对比汇总.json", summary)

    # 对照表（Markdown，按题列两后端）
    write_compare_markdown(rows, Path(cfg.path_str("eval_dir")) / "对比对照表.md", summary)
    # CSV（优化目录）
    csv_rows = [{
        "id": r["id"], "question": r["question"][:40], "backend": r["backend"],
        "答对": "是" if r["score"]["answered"] else "否",
        "答案关键词命中率": r["score"]["answer_hit_rate"],
        "上下文关键词命中率": r["score"]["context_hit_rate"],
        "数字命中率": r["score"]["number_hit_rate"],
        "页命中": "是" if r["score"]["page_hit"] else "否",
        "检索耗时_s": r["retrieve_s"], "生成耗时_s": r["generate_s"], "总耗时_s": r["latency_s"],
        "答案": (r["answer"] or "").replace("\n", " ")[:120],
    } for r in sorted(rows, key=lambda x: (x["id"], x["backend"]))]
    write_csv(Path(cfg.path_str("optimize_dir")) / "对比汇总.csv", csv_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐题结果：{per_q_path}")
    print(f"对照表：{Path(cfg.path_str('eval_dir')) / '对比对照表.md'}")
    print(f"CSV：{Path(cfg.path_str('optimize_dir')) / '对比汇总.csv'}")
    return 0


def write_compare_markdown(rows: list[dict], path: Path, summary: dict) -> Path:
    """逐题两后端对照表（Markdown，数字全部来自实跑）。"""
    by_id: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_id.setdefault(int(r["id"]), {})[str(r["backend"])] = r
    lines = [
        "# 16 题 RAG vs LightRAG 检索结果对照表",
        "",
        "> **工单编号：人工智能NLP-RAG 项目-LightRAG 优化**",
        "> 口径：全文本先做去空白 + 全角括号归一；页命中判定用 **±1 页容差**（物理页 vs 逻辑页）。",
        f"> 判定阈值：``answer_hit_rate ≥ {summary['_meta']['答对阈值']}`` 记为答对；页容差 ±{summary['_meta']['页号容差']}。",
        "",
        "| id | 问题（截断） | RAG 答对 | RAG 答案命中 | LightRAG 答对 | LightRAG 答案命中 | RAG 页命中 | LG 页命中 |",
        "| ---: | --- | :---: | ---: | :---: | ---: | :---: | :---: |",
    ]
    for qid in sorted(by_id):
        r = by_id[qid].get("rag")
        l = by_id[qid].get("lightrag")
        q = (r or l)["question"]
        short = q[:38] + ("…" if len(q) > 38 else "")
        def cell(x, key):  # noqa: E306
            return "—" if not x else (("是" if x["score"][key] else "否") if key in ("answered", "page_hit")
                                      else f"{x['score'][key]:.2f}")
        lines.append(f"| {qid} | {short} | {cell(r,'answered')} | {cell(r,'answer_hit_rate')} | "
                     f"{cell(l,'answered')} | {cell(l,'answer_hit_rate')} | {cell(r,'page_hit')} | {cell(l,'page_hit')} |")
    lines += ["", "## 汇总", "", "| 指标 | RAG | LightRAG |", "| --- | ---: | ---: |"]
    for key in ("题数", "答对数", "答对率", "平均答案关键词命中率", "平均上下文关键词命中率",
                "平均数字命中率", "页命中数", "页命中率", "平均检索耗时_s", "平均生成耗时_s"):
        rag = summary.get("rag", {}).get(key, "—")
        lg = summary.get("lightrag", {}).get(key, "—")
        lines.append(f"| {key} | {rag} | {lg} |")
    lines += ["", "## 逐题明细（含两后端答案）", ""]
    for qid in sorted(by_id):
        first = by_id[qid].get("rag") or by_id[qid].get("lightrag")
        lines += [f"### id={qid}　{first['question']}", "",
                  f"- 参考答案：{first.get('reference_answer')}", ""]
        for backend in BACKENDS:
            r = by_id[qid].get(backend)
            if not r:
                continue
            sc = r["score"]
            lines += [f"**{backend}**　答对={'是' if sc['answered'] else '否'}　"
                      f"答案命中={sc['answer_hit_rate']:.2f}　上下文命中={sc['context_hit_rate']:.2f}　"
                      f"页命中={'是' if sc['page_hit'] else '否'}　延迟={r['latency_s']:.1f}s", "",
                      f"> {r['answer'].strip()[:400] if r['answer'].strip() else '（未生成/为空）'}", "",
                      f"- 命中块页码：{sc['retrieved_pages']}；Top5 块：{sc['top_chunk_ids']}", ""]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
