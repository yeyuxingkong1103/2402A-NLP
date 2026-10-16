# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/在线/run_16q_eval.py（T6：16 题端到端评测跑分器）

对**运行中的服务**（研发/app/main.py --serve，默认 127.0.0.1:8123）逐题发问，用
`测试/common/golden.py` 做**确定性判分**与**三层检索命中**，产出：
  测试/测试数据/qa_results.csv / qa_results.jsonl   逐题明细
  测试/测试数据/accuracy_report.json                汇总（准确率/命中率/延迟 P50-P95-max/版本）
  测试/测试数据/negatives.jsonl                     负例（拒答红线）
  测试/测试数据/bilingual.jsonl                     中英文同题
  测试/测试数据/run_meta.json                       运行口径（两个 K、索引 chunk_total、预热/独占证据）

口径（见 测试/测试数据/golden_复核报告.md §6 与 captain 指派）：
  · 判对 = golden.judge_answer（确定性关键字，非 LLM 自评）
  · 检索命中：① 权威层 hit_complete（命中率主指标）② hit_any_evidence ③ hit_page_level（仅诊断），
    且**按通道分列**（BM25 / 向量 / 合并）
  · 「引用页码可回溯」= **该页确实承载答案证据**（不在册但含证据 = 异页同证，计合格）；
    不是「引用页 == golden 证据页」
  · 延迟主口径 = 客户端「请求发出 → 完整答案返回」；服务端 latency_ms.total 作辅助
运行：
  pwsh -NoProfile -File run_py.ps1 测试/在线/run_16q_eval.py --run-tag t6
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PENDING = "待 T6 实跑填充"

NEGATIVES = [
    {"id": "N1", "type": "不存在的公司", "question": "某某科技有限公司的注册资本是多少？",
     "expect": "refused", "detail": "语料中不存在的公司 → 应回「不清楚」，不得编造"},
    {"id": "N2", "type": "不存在的数据", "question": "武汉兴图新科电子股份有限公司2025年营业收入是多少？",
     "expect": "refused", "detail": "语料未披露的未来年份数据 → 应回「不清楚」"},
    {"id": "N3", "type": "越界/无关问题", "question": "请写一段 Python 快速排序代码。",
     "expect": "refused", "detail": "与语料无关的任务 → 应回「不清楚」"},
    {"id": "N4", "type": "语料存在但未披露项", "question": "武汉力源信息技术股份有限公司募集资金投向第5项「其他与主营业务相关的营运资金」的计划总投资额是多少？",
     "expect": "no_fabrication", "detail": "原文为占位符 [ ◆ ] → 不得编造金额，应说明未披露"},
]

#: 双语判分口径说明（随每一行一起落盘，避免验收人把中文口径的 False 误读成「英文未通过」）
CRITERION_NOTE = ("英文结论以 en_criterion_passed（语言无关原子口径：数值按基本单位等价 + 英文枚举项）为准；"
                  "同一行的 correct 是**中文关键字口径**，对英文答案天然为 False，"
                  "**不代表英文未通过**。")

BILINGUAL = [
    {"id": 543, "zh": "武汉兴图新科电子股份有限公司注册资本是多少？",
     "en": "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
     # 英文路径判据（语言无关原子）：数值按基本单位等价 —— 中文关键字口径对英文答案不适用，
     # 故另立 EN 原子口径并**同时**保留中文确定性判分结果（见报告 §6 说明）
     "en_atoms": ["5,520.00万元", "capital"]},
    {"id": 5, "zh": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
     "en": "In the organization chart of Wuhan P&S Information Technology Co., Ltd., how many departments "
           "make up the Sales Department, and how many sales offices make up the Key Account Sales Department?",
     "en_atoms": ["4", "International Trade Department", "Key Account Sales Department",
                  "Channel Sales Department", "Telephone and Network Sales Department",
                  "6", "Beijing", "Guangzhou", "Chengdu", "Shenzhen", "Wuhan", "Zhuhai"]},
    {"id": 6, "zh": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
     "en": "According to the prospectus of Wuhan P&S Information Technology Co., Ltd., based on the 2008 China IC "
           "market application structure and growth chart, which industry has the fastest growth rate and which has negative growth?",
     "en_atoms": ["14.0", "IC", "fastest", "negative"]},
]


def load_golden_lib():
    spec = importlib.util.spec_from_file_location("golden_lib", ROOT / "测试" / "common" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def page_text(pdf_file: str, page: int) -> str:
    """取 PDF 物理页原文（引用页复核用；只读）。"""
    import fitz
    with fitz.open(ROOT / "研发" / "data" / "raw" / pdf_file) as doc:
        if not (1 <= page <= doc.page_count):
            return ""
        return doc[page - 1].get_text("text")


def citation_support(G, rec, pages_cache: dict, cited_pages) -> dict:
    """引用页码可用性：在册页 / 异页同证（该页确实承载证据）/ 不支撑。

    「异页同证」判定：该页原文（归一化）含该题任一 answer_keyword（数值按基本单位等价）
    或其证据原文的子句片段 —— 与 captain 的逐页核实方法一致。
    """
    in_register, same_evidence_elsewhere, unsupported = [], [], []
    for p in cited_pages or []:
        if p in rec["evidence_pages"]:
            in_register.append(p)
            continue
        if p not in pages_cache:
            pages_cache[p] = page_text(rec["pdf_file"].split("/")[-1], p)
        text = pages_cache[p]
        hit_kw = [kw for kw in rec["answer_keywords"] if G.page_carries(text, kw)]
        frags = [f for ev in (rec.get("retrieval_evidence") or [])
                 if ev.get("text")
                 for f in G.clause_fragments(ev["text"]) if G.normalize(f) in G.normalize(text)]
        if hit_kw or frags:
            same_evidence_elsewhere.append({"page": p, "supported_by_keywords": hit_kw[:4],
                                            "supported_by_fragments": frags[:2]})
        else:
            unsupported.append(p)
    ok = not unsupported
    return {"citation_ok": ok, "in_register_pages": in_register,
            "same_evidence_elsewhere": same_evidence_elsewhere, "unsupported_pages": unsupported,
            "rule": "合格 = 该页确实承载答案证据（在册页 或 异页同证）；"
                    "不是「引用页 == golden evidence_pages」"}


def ask(base_url: str, question: str, lang: str, use_cache: bool, top_k: int, timeout: float) -> tuple[dict, float]:
    body = json.dumps({"question": question, "lang": lang, "use_cache": use_cache, "top_k": top_k},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/api/ask", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, (time.perf_counter() - t0) * 1000.0


def channel_chunks(chunks, channel: str):
    """按通道给出该通道的候选 chunk（按通道名次升序，仅含该通道召回到的块）。"""
    key = "bm25_rank" if channel == "bm25" else "vector_rank"
    got = [c for c in (chunks or []) if c.get(key) is not None]
    return sorted(got, key=lambda c: c[key])


def pctl(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="工单4 T6：16 题端到端评测跑分器")
    ap.add_argument("--base-url", default="http://127.0.0.1:8123")
    ap.add_argument("--out-dir", default=str(ROOT / "测试" / "测试数据"))
    ap.add_argument("--run-tag", default="t6")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--use-cache", action="store_true", help="默认禁用缓存（延迟测量口径）")
    ap.add_argument("--no-warmup", action="store_true", help="跳过预热（**调试口径**，报告须标注）")
    ap.add_argument("--suites", default="main,neg,bilingual")
    args = ap.parse_args(argv)

    G = load_golden_lib()
    records = G.load_golden()
    diag = G.load_diagnostic_evidence()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "研发" / "data" / "index" / "index_manifest.json").read_text(encoding="utf-8"))
    run_meta = {
        "run_tag": args.run_tag,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url, "top_k_returned": args.top_k,
        "generation_prompt_top_k": None, "hit_judgement_top_k": 5,
        "use_cache": bool(args.use_cache), "warmup_done": None,
        "index_chunk_total": manifest.get("chunk_total"),
        "index_by_modality": manifest.get("chunk_by_modality"),
        "index_by_modality_sub": manifest.get("chunk_by_modality_sub"),
        "index_updated_at": (manifest.get("index_meta") or {}).get("updated_at"),
        "golden_sha256": __import__("hashlib").sha256(G.GOLDEN_CANONICAL.read_bytes()).hexdigest(),
        "backend": None, "model": None, "notes": [],
    }
    pages_cache: dict = {}
    suites = {s.strip() for s in args.suites.split(",") if s.strip()}

    # ---- 预热 + 独占证据 ----
    if not args.no_warmup:
        try:
            ask(args.base_url, "武汉兴图新科电子股份有限公司注册资本是多少？", "zh", False, 3, args.timeout)
            run_meta["warmup_done"] = True
        except Exception as exc:                                     # noqa: BLE001
            run_meta["warmup_done"] = False
            run_meta["notes"].append(f"预热失败：{type(exc).__name__}: {exc}")

    health = None
    try:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=60) as r:
            health = json.loads(r.read().decode("utf-8"))
        run_meta["backend"] = (health.get("checks", {}).get("llm") or {}).get("backend")
        run_meta["model"] = (health.get("checks", {}).get("llm") or {}).get("model")
        run_meta["generation_prompt_top_k"] = (health.get("config") or {}).get("prompt_top_k")
        run_meta["health_status"] = health.get("status")
    except Exception as exc:                                          # noqa: BLE001
        run_meta["notes"].append(f"健康检查失败：{type(exc).__name__}: {exc}")

    rows, summary = [], None
    if "main" in suites:
        for rec in records:
            row = {"order": rec["order"], "id": rec["id"], "category": rec["category"], "pdf": rec["pdf"],
                   "question": rec["question"], "golden_answer": rec["answer"],
                   "evidence_pages_complete": rec["evidence_pages_complete"],
                   "answer_form": rec.get("answer_form")}
            try:
                data, wall_ms = ask(args.base_url, rec["question"], "zh", args.use_cache, args.top_k, args.timeout)
            except Exception as exc:                                  # noqa: BLE001
                row.update({"system_answer": None, "correct": None, "error": f"{type(exc).__name__}: {exc}",
                            "latency_wall_ms": None, "note": "请求失败，如实记录，不补数"})
                rows.append(row)
                continue
            verdict = G.judge_answer(rec, data.get("answer") or "")
            chunks = data.get("chunks") or []
            merged = G.check_retrieval_hit_layers(rec, chunks, diagnostic=diag)
            bm = G.check_retrieval_hit_layers(rec, channel_chunks(chunks, "bm25"), diagnostic=diag)
            vec = G.check_retrieval_hit_layers(rec, channel_chunks(chunks, "vector"), diagnostic=diag)
            cit = citation_support(G, rec, pages_cache, data.get("cited_pages"))
            lat = data.get("latency_ms") or {}
            row.update({
                "system_answer": data.get("answer"), "refused": data.get("refused"),
                "generation_method": data.get("generation_method"), "cache": data.get("cache"),
                "cited_pages": data.get("cited_pages"), "citations": data.get("citations"),
                "correct": verdict["correct"], "judge_reason": verdict["reason"],
                "missing_keywords": [k["keyword"] for k in verdict["keyword_results"] if not k["hit"]],
                "forbidden_fired": [f.get("keyword") or f.get("regex") for f in verdict["forbidden_fired"]],
                "citation_ok": cit["citation_ok"], "citation_detail": cit,
                "hit_complete": merged["hit_complete"], "hit_any_evidence": merged["hit_any_evidence"],
                "hit_page_level": merged["hit_page_level"],
                "hit_complete_bm25": bm["hit_complete"], "hit_complete_vector": vec["hit_complete"],
                "n_chunks": len(chunks),
                "chunk_ids": [c.get("chunk_id") for c in chunks],
                "latency_wall_ms": round(wall_ms, 1),
                "latency_server_total_ms": lat.get("total"),
                "latency_first_token_ms": lat.get("first_token"),
                "latency_retrieve_ms": lat.get("retrieve"), "latency_generate_ms": lat.get("generate"),
                "note": "" if verdict["correct"] else "见 judge_reason / missing_keywords",
            })
            rows.append(row)
            print(f"[{rec['order']:>2}/16] id {rec['id']:>3} 判对={verdict['correct']!s:<5} "
                  f"权威层={merged['hit_complete']!s:<5} bm25={bm['hit_complete']!s:<5} "
                  f"向量={vec['hit_complete']!s:<5} 引用页={data.get('cited_pages')} wall={wall_ms:.0f}ms")

        lat_wall = [r["latency_wall_ms"] for r in rows if r.get("latency_wall_ms") is not None]
        lat_srv = [r["latency_server_total_ms"] for r in rows if r.get("latency_server_total_ms") is not None]
        n_correct = sum(1 for r in rows if r.get("correct") is True)
        judged = [r for r in rows if r.get("correct") is not None]
        summary = {
            "n_questions": len(rows), "n_judged": len(judged), "n_correct": n_correct,
            "accuracy": round(n_correct / len(judged), 4) if judged else None,
            "n_hit_complete": sum(1 for r in judged if r.get("hit_complete")),
            "retrieval_hit_rate_complete": round(sum(1 for r in judged if r.get("hit_complete")) / len(judged), 4) if judged else None,
            "n_hit_complete_bm25": sum(1 for r in judged if r.get("hit_complete_bm25")),
            "retrieval_hit_rate_complete_bm25": round(sum(1 for r in judged if r.get("hit_complete_bm25")) / len(judged), 4) if judged else None,
            "n_hit_complete_vector": sum(1 for r in judged if r.get("hit_complete_vector")),
            "retrieval_hit_rate_complete_vector": round(sum(1 for r in judged if r.get("hit_complete_vector")) / len(judged), 4) if judged else None,
            "n_hit_any_evidence_diag": sum(1 for r in judged if r.get("hit_any_evidence")),
            "n_hit_page_level_diag": sum(1 for r in judged if r.get("hit_page_level")),
            "n_citation_ok": sum(1 for r in judged if r.get("citation_ok")),
            "n_citation_same_evidence_elsewhere": sum(
                1 for r in judged if (r.get("citation_detail") or {}).get("same_evidence_elsewhere")),
            "latency_wall_p50_ms": pctl(lat_wall, 0.50), "latency_wall_p95_ms": pctl(lat_wall, 0.95),
            "latency_wall_max_ms": round(max(lat_wall), 1) if lat_wall else None,
            "latency_wall_min_ms": round(min(lat_wall), 1) if lat_wall else None,
            "latency_server_p50_ms": pctl(lat_srv, 0.50), "latency_server_p95_ms": pctl(lat_srv, 0.95),
            "latency_server_max_ms": round(max(lat_srv), 1) if lat_srv else None,
            "n_latency_over_3000ms": sum(1 for v in lat_wall if v > 3000.0),
            "first_token_min_ms": min((r["latency_first_token_ms"] for r in rows
                                       if r.get("latency_first_token_ms")), default=None),
            "first_token_max_ms": max((r["latency_first_token_ms"] for r in rows
                                       if r.get("latency_first_token_ms")), default=None),
            "note": "判分=golden.judge_answer（确定性）；命中率主指标=权威层 hit_complete，"
                    "另列 BM25/向量/诊断列；延迟主口径=客户端 wall（请求发出→完整答案返回）",
        }
        with (out_dir / "qa_results.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        cols = ["order", "id", "category", "pdf", "question", "system_answer", "golden_answer", "correct",
                "missing_keywords", "forbidden_fired", "cited_pages", "citation_ok", "hit_complete",
                "hit_complete_bm25", "hit_complete_vector", "hit_any_evidence", "hit_page_level",
                "n_chunks", "chunk_ids", "latency_wall_ms", "latency_server_total_ms",
                "latency_first_token_ms", "generation_method", "cache", "note"]
        with (out_dir / "qa_results.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "cited_pages": json.dumps(r.get("cited_pages"), ensure_ascii=False),
                            "missing_keywords": json.dumps(r.get("missing_keywords"), ensure_ascii=False),
                            "chunk_ids": json.dumps(r.get("chunk_ids"), ensure_ascii=False)})

    # ---- 负例（拒答红线）----
    neg_results = []
    if "neg" in suites:
        for case in NEGATIVES:
            try:
                data, wall_ms = ask(args.base_url, case["question"], "zh", args.use_cache, args.top_k, args.timeout)
            except Exception as exc:                                  # noqa: BLE001
                neg_results.append({**case, "answer": None, "refused": None, "passed": None,
                                    "error": f"{type(exc).__name__}: {exc}"})
                continue
            answer = data.get("answer") or ""
            refused = bool(data.get("refused"))
            said_unknown = "不清楚" in answer or "未披露" in answer or "无法" in answer
            fabricated = False
            if case["expect"] == "refused":
                passed = refused and said_unknown
            else:  # no_fabrication
                fabricated = bool(__import__("re").search(r"营运资金[^。；\n]{0,12}\d", G.normalize(answer)))
                passed = said_unknown and not fabricated
            neg_results.append({**case, "answer": answer, "refused": refused,
                                "refusal_reason": data.get("refusal_reason"),
                                "said_unknown": said_unknown, "fabricated_amount": fabricated,
                                "passed": passed, "latency_wall_ms": round(wall_ms, 1)})
            print(f"[neg {case['id']}] passed={passed!s:<5} refused={refused!s:<5} "
                  f"reason={data.get('refusal_reason')} answer={answer[:40]!r}")
        with (out_dir / "negatives.jsonl").open("w", encoding="utf-8") as fh:
            for r in neg_results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 中英文同题 ----
    bi_results = []
    if "bilingual" in suites:
        for case in BILINGUAL:
            rec = G.load_golden_index()[case["id"]]
            for lang in ("zh", "en"):
                q = case[lang]
                try:
                    data, wall_ms = ask(args.base_url, q, lang, args.use_cache, args.top_k, args.timeout)
                except Exception as exc:                              # noqa: BLE001
                    bi_results.append({"id": case["id"], "language": lang, "question": q,
                                       "error": f"{type(exc).__name__}: {exc}", "correct": None})
                    continue
                verdict = G.judge_answer(rec, data.get("answer") or "")
                merged = G.check_retrieval_hit_layers(rec, data.get("chunks") or [], diagnostic=diag)
                en_atoms = case.get("en_atoms") or []
                answer = data.get("answer") or ""
                lowered = answer.lower()
                atoms_missing = []
                for atom in en_atoms:
                    ok = G.keyword_hit(answer, atom)[0] or atom.lower() in lowered
                    if not ok:
                        atoms_missing.append(atom)
                bi_results.append({"id": case["id"], "language": lang, "question": q,
                                   "answer": answer, "correct": verdict["correct"],
                                   "criterion_note": CRITERION_NOTE,
                                   "judge_reason": verdict["reason"],
                                   "zh_keyword_judge": verdict["correct"] if lang == "zh" else verdict["correct"],
                                   "en_atoms_expected": en_atoms,
                                   "en_atoms_missing": atoms_missing,
                                   "en_criterion_passed": (not atoms_missing) if lang == "en" else None,
                                   "language_degraded": data.get("language_degraded"),
                                   "cited_pages": data.get("cited_pages"),
                                   "hit_complete": merged["hit_complete"],
                                   "latency_wall_ms": round(wall_ms, 1)})
                flag = verdict["correct"] if lang == "zh" else (not atoms_missing)
                print(f"[bi id {case['id']} {lang}] 通过={flag!s:<5} 中文口径={verdict['correct']!s:<5} "
                      f"引用页={data.get('cited_pages')} 缺原子={atoms_missing[:4]}")
        with (out_dir / "bilingual.jsonl").open("w", encoding="utf-8") as fh:
            for r in bi_results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report = {"run_meta": run_meta, "summary": summary,
              "negatives": {"n": len(neg_results), "n_passed": sum(1 for r in neg_results if r.get("passed") is True),
                            "items": neg_results},
              "bilingual": {"n": len(bi_results), "n_correct": sum(1 for r in bi_results if r.get("correct") is True),
                            "items": bi_results}}
    (out_dir / "accuracy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[OK] 落盘 → {out_dir}\\qa_results.csv / qa_results.jsonl / accuracy_report.json / "
          f"negatives.jsonl / bilingual.jsonl / run_meta.json")
    if summary:
        print(f"     准确率 {summary['n_correct']}/{summary['n_judged']} = {summary['accuracy']}; "
              f"权威层命中 {summary['n_hit_complete']}/{summary['n_judged']} "
              f"(BM25 {summary['n_hit_complete_bm25']} / 向量 {summary['n_hit_complete_vector']}); "
              f"延迟 wall P50 {summary['latency_wall_p50_ms']} / P95 {summary['latency_wall_p95_ms']} / "
              f"max {summary['latency_wall_max_ms']} ms; 引用可回溯 {summary['n_citation_ok']}/{summary['n_judged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
