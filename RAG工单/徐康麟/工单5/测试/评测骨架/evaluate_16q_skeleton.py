# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/评测骨架/evaluate_16q_skeleton.py（**T6 骨架**：16 题评测 CLI 的参数解析与输出结构）

⚠️ 骨架约束（captain 下达，勿破）：
  1. 本文件**不产生任何成绩**：`--transport stub` 全部数字字段写 `null` + 「待 T6 实跑填充」；
  2. 数字只有一个来源：`--transport http` 时对真实服务的**实测值**；
  3. 不写结论性语句（不出现「达标/不达标」判断）；
  4. 判分与检索命中一律调用 `测试/common/golden.py`（唯一真值），禁止在评测脚本里另写关键词。

用法：
  # 骨架自检（不接引擎；产出 null 结构与空表，验证报告结构可用）
  pwsh -NoProfile -File run_py.ps1 测试/评测骨架/evaluate_16q_skeleton.py --transport stub

  # T6 正式评测（引擎就绪后）
  pwsh -NoProfile -File run_py.ps1 测试/评测骨架/evaluate_16q_skeleton.py --transport http \
      --base-url http://127.0.0.1:8123 --out-dir 测试/测试数据/评测_accept --run-tag accept
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PENDING = "待 T6 实跑填充"


def load_golden_lib():
    """按路径加载 测试/common/golden.py（目录名含中文，避免包导入差异）。"""
    spec = importlib.util.spec_from_file_location("golden_lib", ROOT / "测试" / "common" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数解析（骨架已冻结；T6 只增不改语义）。"""
    p = argparse.ArgumentParser(
        prog="evaluate_16q_skeleton.py",
        description="工单4：16 题评测骨架（判分走 测试/common/golden.py；未实跑一律 null）")
    p.add_argument("--golden", default=None, help="golden 路径（默认 研发/data/eval/golden_qa.jsonl）")
    p.add_argument("--transport", choices=["stub", "http"], default="stub",
                   help="stub=不接引擎只冻结结构；http=对真实服务实测")
    p.add_argument("--base-url", default="http://127.0.0.1:8123", help="服务地址（transport=http 时使用）")
    p.add_argument("--endpoint", default="/api/ask", help="问答端点路径")
    p.add_argument("--timeout", type=float, default=30.0, help="单题超时（秒）")
    p.add_argument("--run-tag", default="skeleton", help="运行标签（写入 run_meta 与文件名）")
    p.add_argument("--out-dir", dest="out_dir", default=None,
                   help="输出目录（与正式版 测试/在线/run_16q_eval.py 同名）。默认：stub → 系统临时目录"
                        "（避免 null 结构留在交付目录被误读为实测）；http → 测试/测试数据/评测_<run-tag>")
    p.add_argument("--out", dest="out_dir", help="兼容旧写法的别名（等同 --out-dir）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部；调试用，报告中须标注）")
    p.add_argument("--ids", default="", help="逗号分隔的题目 id 子集（调试用；报告中须标注）")
    p.add_argument("--no-warmup", action="store_true",
                   help="跳过预热自检（**调试口径**，报告中必须标注为 --no-warmup）")
    return p


def ask_service(base_url: str, endpoint: str, question: str, timeout: float) -> dict:
    """真实调用一次问答端点。返回 {answer, latency_total_ms, latency_first_token_ms, cited_pages, chunks, error}

    ⚠️ T6 待办：按 设计/接口设计.md 把响应字段映射补全（本函数已按契约预留 4 个字段名）。
    这里**不写任何默认值**：拿不到就返回 error，让上层写 null。
    """
    import requests  # 仅在此函数内导入，stub 模式不依赖网络库

    payload = {"question": question}
    t0 = time.perf_counter()
    try:
        resp = requests.post(base_url.rstrip("/") + endpoint, json=payload, timeout=timeout)
        total_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:                                   # noqa: BLE001 - 如实记录失败
        return {"answer": None, "latency_total_ms": None, "latency_first_token_ms": None,
                "cited_pages": None, "chunks": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "answer": data.get("answer"),
        "latency_total_ms": data.get("latency_ms", {}).get("total", total_ms),
        "latency_first_token_ms": data.get("latency_ms", {}).get("first_token"),
        "cited_pages": data.get("cited_pages") or data.get("citations"),
        "chunks": data.get("chunks") or data.get("retrieved"),
        "error": None,
    }


def evaluate(args) -> dict:
    """逐题评测（骨架）：判分/命中调用 golden 库；未实跑时所有成绩字段为 None。"""
    G = load_golden_lib()
    records = G.load_golden(args.golden) if args.golden else G.load_golden()
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        records = [r for r in records if r["id"] in wanted]
    if args.limit:
        records = records[: args.limit]

    rows = []
    for rec in records:
        row = {
            "order": rec["order"], "id": rec["id"], "category": rec["category"],
            "pdf": rec["pdf"], "question": rec["question"],
            "golden_answer": rec["answer"],
            "system_answer": None, "correct": None, "judge_detail": None,
            "evidence_pages_complete": rec["evidence_pages_complete"],
            "cited_pages": None, "citation_ok": None,
            # ① 权威层「完整证据命中」——唯一命中率主指标，且**按通道分列**（禁止只报合并值）
            "retrieval_hit_complete_bm25": None,
            "retrieval_hit_complete_vector": None,
            "retrieval_hit_complete_any": None,
            # ② 诊断层A「任证证据命中」（complete + supporting 的任一证据原文落块）
            "retrieval_hit_any_evidence_any": None,
            # ③ 诊断层B「页级命中」（chunk.page ∈ evidence_pages；**纯诊断，禁止用作命中率**）
            "retrieval_hit_page_level_diagnostic": None,
            "retrieval_detail": None,
            "extraction_method": None,
            "latency_total_ms": None, "latency_first_token_ms": None,
            "cache_hit": None, "error": None, "note": PENDING,
        }
        if args.transport == "http":
            res = ask_service(args.base_url, args.endpoint, rec["question"], args.timeout)
            row.update({"system_answer": res["answer"], "cited_pages": res["cited_pages"],
                        "latency_total_ms": res["latency_total_ms"],
                        "latency_first_token_ms": res["latency_first_token_ms"],
                        "error": res["error"]})
            if res["answer"] is not None:
                verdict = G.judge_answer(rec, res["answer"])
                row["correct"] = verdict["correct"]
                row["judge_detail"] = verdict
                row["cache_hit"] = False          # ⚠️ T6 待办：由服务返回的 debug 字段判定
            # ⚠️ T6 待办：`chunks_by_channel`（{bm25: [...], vector: [...]}）须由服务或本地检索复算提供；
            #    未提供时保持 None —— **不得**用合并 chunk 冒充分通道结果。
            by_ch = res.get("chunks_by_channel") or {}
            for ch in ("bm25", "vector"):
                if by_ch.get(ch) is not None:
                    layers = G.check_retrieval_hit_layers(rec, by_ch[ch])   # 三层口径（captain 裁定）
                    row[f"retrieval_hit_complete_{ch}"] = layers["hit_complete"]
                    if ch == "bm25":
                        row["retrieval_detail"] = layers
            if any(row[f"retrieval_hit_complete_{c}"] is not None for c in ("bm25", "vector")):
                row["retrieval_hit_complete_any"] = any(
                    row[f"retrieval_hit_complete_{c}"] is True for c in ("bm25", "vector"))
            merged = res.get("chunks") or [c for ch in ("bm25", "vector")
                                           for c in (by_ch.get(ch) or [])]
            if merged:
                L = G.check_retrieval_hit_layers(rec, merged)
                row["retrieval_hit_any_evidence_any"] = L["hit_any_evidence"]
                row["retrieval_hit_page_level_diagnostic"] = L["hit_page_level"]
        rows.append(row)

    lat = [r["latency_total_ms"] for r in rows if r["latency_total_ms"] is not None]
    hit_cols = ("complete_bm25", "complete_vector", "complete_any")
    summary = {
        "n_questions": len(rows),
        "n_correct": sum(1 for r in rows if r["correct"] is True) if any(
            r["correct"] is not None for r in rows) else None,
        "accuracy": None,                       # ⚠️ 待 T6 实跑填充（由判对题数/题数计算，且须标注口径）
        # ① 权威层（按通道 + 合并）
        "n_retrieval_hit_complete_bm25": None, "retrieval_hit_rate_complete_bm25": None,
        "n_retrieval_hit_complete_vector": None, "retrieval_hit_rate_complete_vector": None,
        "n_retrieval_hit_complete_any": None, "retrieval_hit_rate_complete_any": None,
        # ② 诊断层A / ③ 诊断层B（不得用作命中率）
        "n_retrieval_hit_any_evidence": None, "n_retrieval_hit_page_level_diagnostic": None,
        "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
        "latency_p95_ms": None,                 # ⚠️ 待 T6 实跑填充（P95 口径须写明）
        "latency_max_ms": round(max(lat), 1) if lat else None,
        "n_latency_over_3000ms": None,          # ⚠️ 待 T6 实跑填充
        "note": PENDING,
    }
    for ch in hit_cols:
        col = f"retrieval_hit_{ch}"
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            summary[f"n_retrieval_hit_{ch}"] = sum(1 for v in vals if v is True)
            summary[f"retrieval_hit_rate_{ch}"] = round(sum(1 for v in vals if v is True) / len(vals), 4)
    for col, key in (("retrieval_hit_any_evidence_any", "n_retrieval_hit_any_evidence"),
                     ("retrieval_hit_page_level_diagnostic", "n_retrieval_hit_page_level_diagnostic")):
        vals = [r[col] for r in rows if r[col] is not None]
        summary[key] = sum(1 for v in vals if v is True) if vals else None
    if summary["n_correct"] is not None:
        jd = [r for r in rows if r["correct"] is not None]
        summary["accuracy"] = round(summary["n_correct"] / len(jd), 4) if jd else None
        summary["n_latency_over_3000ms"] = sum(1 for ms in lat if ms > 3000.0)
        if lat:
            ordered = sorted(lat)
            idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
            summary["latency_p95_ms"] = round(ordered[idx], 1)
    return {"run_meta": {
        "run_tag": args.run_tag,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url if args.transport == "http" else None,
        "model": None,                          # ⚠️ 待 T6 实跑填充（如实填模型名/版本）
        "warmup_done": None,                    # ⚠️ 待 T6 实跑填充
        "exclusive_ollama_verified": None,      # ⚠️ 待 T6 实跑填充（附 ollama ps / ESTABLISHED 证据）
        "generation_top_k": None,               # ⚠️ 待 T6 实跑填充（用户要求 K=3，来源=T2/T5 配置）
        "hit_judgement_top_k": 5,               # golden 固定：前 K 条内命中即算
        "transport": args.transport,
        "debug_flags": {"no_warmup": bool(args.no_warmup), "limit": args.limit, "ids": args.ids},
        "note": PENDING,
    }, "per_question": rows, "summary": summary}


def write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qa_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in result["per_question"]) + "\n", encoding="utf-8")
    cols = ["order", "id", "category", "pdf", "question", "system_answer", "golden_answer", "correct",
            "evidence_pages_complete", "cited_pages", "citation_ok",
            "retrieval_hit_complete_bm25", "retrieval_hit_complete_vector", "retrieval_hit_complete_any",
            "retrieval_hit_any_evidence_any", "retrieval_hit_page_level_diagnostic", "extraction_method",
            "latency_total_ms", "latency_first_token_ms", "cache_hit", "error", "note"]
    with (out_dir / "qa_results.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in result["per_question"]:
            w.writerow({**r, "evidence_pages_complete": json.dumps(r["evidence_pages_complete"], ensure_ascii=False),
                        "cited_pages": json.dumps(r["cited_pages"], ensure_ascii=False)})
    (out_dir / "accuracy_report.json").write_text(
        json.dumps({"run_meta": result["run_meta"], "summary": result["summary"]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "run_meta.json").write_text(
        json.dumps(result["run_meta"], ensure_ascii=False, indent=1), encoding="utf-8")


def resolve_out_dir(args) -> Path:
    """默认输出目录：stub 一律写系统临时目录（**不得留在交付目录**，防止 null 结构被误读为实测）。

    ⚠️ 交付纪律：`测试/**` 下**不得**出现任何「成绩字段为 null」的 qa_results.csv——
    出报告时它可能被误认为实测结果。stub 产物用完即弃（随时可由同一条命令再生）。
    """
    if args.out_dir:
        return Path(args.out_dir)
    if args.transport == "stub":
        return Path(tempfile.gettempdir()) / "gongdan4_eval_skeleton_stub"
    return ROOT / "测试" / "测试数据" / f"评测_{args.run_tag}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    out_dir = resolve_out_dir(args)
    write_outputs(result, out_dir)
    print(f"[OK] 已写出评测结构（transport={args.transport}）→ {out_dir}")
    print(f"     逐题 {result['summary']['n_questions']} 行；qa_results.csv / qa_results.jsonl / "
          f"accuracy_report.json / run_meta.json")
    if args.transport == "stub":
        print("     ⚠️ 骨架模式：所有成绩字段为 null（待 T6 实跑填充），本文件不含任何成绩；")
        print("        产物在系统临时目录，**不得复制进 测试/**（会被误当成实测结果）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
