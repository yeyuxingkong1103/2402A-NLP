# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""31_ragas_eval.py —— RAGAS 四项指标（官方包缺失时自实现等价指标）。

**依赖现状（实测）**：``ragas`` / ``pandas`` / ``datasets`` 均未安装
（本机 pip 落点红线禁止裸装，且 pandas 缺失使官方 ragas 无法工作）。
故本脚本按 ``设计/检索与评估方案.md`` §3.4 的降级口径**自实现等价指标**，
字段名统一带 ``_kw`` 后缀以示区分，并在输出里写明与官方口径的差异。

自实现四项（全文本先做**去空白 + 全角括号归一**）：
  * ``faithfulness_kw``      = 答案中的可校验断言（参考关键词 + 参考数值）中被检索上下文支持的比例
  * ``answer_relevancy_kw``  = 答案与问题的字符 bigram Dice 相似度（≥ ``answer_relevancy_min_len`` 才计分）
  * ``context_precision_kw`` = 相关块（含参考关键词 或 与问题相似度 ≥ 阈值）在排序中的平均精度（AP）
  * ``context_recall_kw``    = 参考关键词在检索上下文中被覆盖的比例

与官方 RAGAS 的差异（必须与数字同时呈现）：
  官方用 **LLM 逐断言判定 + 嵌入语义相似**；本实现用 **关键词/数值覆盖 + 字符重叠**，
  方向性可用（能区分两后端谁更好），**绝对值不可与公开榜单直接比较**。

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/31_ragas_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from lightrag.config import load_config                                    # noqa: E402
from lightrag.entity_schema import char_similarity                        # noqa: E402
from lightrag.logging_util import get_logger, write_json                   # noqa: E402
from lightrag.qa import extract_numbers                                    # noqa: E402
from lightrag.routing import norm_ws                                       # noqa: E402

DIFF_NOTE = (
    "自实现等价指标，非 RAGAS 官方实现：official=LLM 逐断言判定 + 嵌入语义相似；"
    "self=关键词/数值覆盖 + 字符 bigram 重叠。方向性一致，绝对值不可与公开榜单比较。"
    "依赖缺失根因：ragas/pandas/datasets 均未安装（本机 pip 落点红线禁止裸装）。"
)


def load_chunks(cfg) -> dict[str, dict]:
    path = Path(cfg.path_str("processed_dir")) / "chunks.jsonl"
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[str(row.get("chunk_id"))] = row
    return out


def try_official() -> tuple[bool, str]:
    """探测官方 ragas 是否可用。"""
    try:
        import ragas  # noqa: F401

        return True, getattr(ragas, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def compute(question: str, answer: str, context: str, keywords: list[str], ref_answer: str, *,
            min_len: int, cp_threshold: float) -> dict:
    """计算单条记录的四项自实现指标。"""
    q, a, c = norm_ws(question), norm_ws(answer), norm_ws(context)
    kws = [norm_ws(k) for k in keywords if str(k).strip()]
    # 相关块集合（逐块判定，用于 context_precision）
    blocks: list[str] = [norm_ws(b) for b in context.split("\n\n") if b.strip()]

    # ① faithfulness_kw：答案里的可校验断言有多少被上下文支持
    ref_nums = [n for n in extract_numbers(norm_ws(ref_answer)) if len(n) >= 3]
    claims = [k for k in kws if k in a] + [n for n in ref_nums if n in a]
    supported = [x for x in claims if x in c]
    faithfulness = (len(supported) / len(claims)) if claims else 0.0

    # ② answer_relevancy_kw：答案与问题的字符 bigram Dice
    relevancy = char_similarity(a, q) if len(a) >= int(min_len) else 0.0

    # ③ context_precision_kw：相关块的 AP
    relevant_flags: list[bool] = []
    for blk in blocks:
        hit = any(k in blk for k in kws) if kws else False
        if not hit:
            hit = char_similarity(blk[:600], q) >= float(cp_threshold)
        relevant_flags.append(hit)
    hits_cnt = sum(1 for f in relevant_flags if f)
    if hits_cnt:
        acc = 0.0
        ap = 0.0
        for i, flag in enumerate(relevant_flags, start=1):
            if flag:
                acc += 1
                ap += acc / i
        precision = ap / hits_cnt
    else:
        precision = 0.0

    # ④ context_recall_kw：参考关键词在上下文中的覆盖比例
    recall = (sum(1 for k in kws if k in c) / len(kws)) if kws else 0.0

    return {
        "faithfulness_kw": round(faithfulness, 4),
        "answer_relevancy_kw": round(relevancy, 4),
        "context_precision_kw": round(precision, 4),
        "context_recall_kw": round(recall, 4),
        "_claims": len(claims), "_supported": len(supported),
        "_relevant_blocks": hits_cnt, "_blocks": len(blocks),
        "_keywords": len(kws), "_answer_chars": len(a),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAGAS 四项（自实现降级口径）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-question", default=None, help="逐题结果 jsonl（默认 研发/评估/逐题结果.jsonl）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("31_ragas", cfg, console=True)

    official_ok, official_note = try_official()
    log.info("官方 ragas 可用性：%s（%s）", official_ok, official_note)

    per_q = Path(args.per_question) if args.per_question else Path(cfg.path_str("eval_dir")) / "逐题结果.jsonl"
    if not per_q.is_file():
        log.error("找不到逐题结果：%s（先跑 30_compare_rag_vs_lightrag.py）", per_q)
        return 2
    records = [json.loads(l) for l in per_q.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = load_chunks(cfg)

    limit = int(cfg.answer.top_k_in_prompt)
    min_len = int(cfg.eval.answer_relevancy_min_len)
    cp_thr = float(cfg.eval.context_precision_threshold)

    scored: list[dict] = []
    for rec in records:
        hits = rec.get("hits") or []
        context = "\n\n".join(str((chunks.get(h.get("chunk_id")) or {}).get("content") or "")
                              for h in hits[:limit])
        m = compute(str(rec.get("question") or ""), str(rec.get("answer") or ""), context,
                    list(rec.get("reference_keywords") or []), str(rec.get("reference_answer") or ""),
                    min_len=min_len, cp_threshold=cp_thr)
        scored.append({"id": rec.get("id"), "backend": rec.get("backend"),
                       "question": rec.get("question"), **m})

    # 聚合
    metrics = ["faithfulness_kw", "answer_relevancy_kw", "context_precision_kw", "context_recall_kw"]
    summary: dict = {}
    for backend in sorted({str(s["backend"]) for s in scored}):
        sub = [s for s in scored if str(s["backend"]) == backend]
        n = len(sub) or 1
        summary[backend] = {m: round(sum(float(s[m]) for s in sub) / n, 4) for m in metrics}
        summary[backend]["题数"] = len(sub)

    payload = {
        "work_order": "人工智能NLP-RAG 项目-LightRAG 优化",
        "官方ragas可用": official_ok,
        "官方ragas探测结果": official_note,
        "口径差异说明": DIFF_NOTE,
        "指标口径": {
            "faithfulness_kw": "答案中可校验断言（参考关键词+参考数值）被检索上下文支持的比例；无断言时保守记 0",
            "answer_relevancy_kw": f"答案与问题的字符 bigram Dice；答案长度 < answer_relevancy_min_len({min_len}) 记 0",
            "context_precision_kw": f"相关块（含参考关键词 或 与问题 Dice ≥ {cp_thr}）的平均精度 AP",
            "context_recall_kw": "参考关键词在检索上下文中的覆盖比例",
            "文本归一化": "去空白 + 全角括号归一（cfg.eval.normalize_whitespace）",
        },
        "汇总": summary,
        "逐题": scored,
    }
    write_json(Path(cfg.path_str("eval_dir")) / "RAGAS指标.json", payload)
    write_json(Path(cfg.path_str("optimize_dir")) / "RAGAS指标.json", payload)

    print("=" * 78)
    print("RAGAS 四项指标（自实现降级口径，官方包不可用）")
    print("=" * 78)
    print(f"官方 ragas 可用：{official_ok}　（{official_note}）")
    print(f"{'后端':<18}{'faithfulness':>14}{'answer_relev':>14}{'ctx_precision':>15}{'ctx_recall':>13}")
    for backend, row in summary.items():
        print(f"{backend:<18}{row['faithfulness_kw']:>14.4f}{row['answer_relevancy_kw']:>14.4f}"
              f"{row['context_precision_kw']:>15.4f}{row['context_recall_kw']:>13.4f}")
    print("-" * 78)
    print(DIFF_NOTE)
    print(f"\n落盘：{Path(cfg.path_str('eval_dir')) / 'RAGAS指标.json'}")
    print(f"落盘：{Path(cfg.path_str('optimize_dir')) / 'RAGAS指标.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
