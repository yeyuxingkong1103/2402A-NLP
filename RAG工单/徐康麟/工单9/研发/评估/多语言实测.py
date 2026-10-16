# -*- coding: utf-8 -*-
"""工单9 多语言问答实测：中文 + 英文各若干题，落 `测试/测试数据/多语言问答.jsonl`。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

工单条款：「多语言支持：支持中文和英文的问答」。
本脚本对**中英文配对**的问题各跑一次，记录：
    * 是否走英中术语映射（`rewrite_added_terms` 非空表示映射生效）；
    * 答案语言是否与提问语言一致；
    * 答案里的**数值**是否与中文提问时一致（数值不该因语言而改变）；
    * 响应时间与生成方式。

用法：
    pwsh -NoProfile -File run_py.ps1 研发/评估/多语言实测.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "研发"), str(_ROOT / "研发" / "评估")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.graphrag_opt.配置 import TEST_DATA_DIR, ensure_dirs, get_config  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 中英文配对问题（同一信息需求，两种语言）
PAIRS: list[dict[str, str]] = [
    {"pair": "邮储银行净利润",
     "zh": "邮储银行2019年年度报告中，该行净利润和同比增速分别是多少？",
     "en": "According to Postal Savings Bank of China's 2019 Annual Report, what were the bank's net profit and its year-on-year growth rate?"},
    {"pair": "平安银行营业收入与净利润",
     "zh": "平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？",
     "en": "In Ping An Bank's 2019 Annual Report, what were the bank's operating income and net profit in 2019?"},
    {"pair": "中国太保营业收入",
     "zh": "中国太保2021年年度报告中，集团2021年营业收入和净利润分别是多少？",
     "en": "What were China Pacific Insurance Group's total revenue and net profit in 2021?"},
    {"pair": "国泰君安归母净利润",
     "zh": "国泰君安2021年年度报告中，公司全年合并口径营业收入、归母净利润分别是多少？",
     "en": "In Guotai Junan's 2021 Annual Report, what were the company's consolidated operating income and net profit attributable to shareholders?"},
]


def _atoms(text: str) -> list[str]:
    import re

    pat = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|百万元|元|亿美元|亿港元)"
                     r"|\d+(?:\.\d+)?\s*(?:%|个百分点|倍)")
    return list(dict.fromkeys(re.sub(r"\s+", "", m) for m in pat.findall(str(text or ""))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 多语言实测")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_config()
    ensure_dirs(cfg)
    try:
        from app.core.logging_conf import setup_logging

        logger = setup_logging()
    except Exception:  # noqa: BLE001
        class _Null:
            def log_event(self, *a: Any, **k: Any) -> None:
                return None

            def bind(self, **k: Any) -> "_Null":
                return self

            def enter(self, *a: Any, **k: Any) -> "_Null":
                return self

        logger = _Null()

    from app.graphrag_opt.引擎 import OptimizedEngine

    engine = OptimizedEngine(cfg=cfg, profile="after", logger=logger)
    print("[预热]", engine.warmup(with_llm=not args.no_llm))

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEST_DATA_DIR / "多语言问答.jsonl"
    rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for item in PAIRS:
            for lang, question in (("zh", item["zh"]), ("en", item["en"])):
                t0 = time.perf_counter()
                try:
                    ans = engine.answer(question, top_k=int(args.top_k), language=lang,
                                        use_llm=not args.no_llm)
                    err = None
                except Exception as exc:  # noqa: BLE001 —— 单题异常不中断整轮
                    ans, err = None, f"{type(exc).__name__}: {exc}"
                ms = round((time.perf_counter() - t0) * 1000, 2)
                if ans is None:
                    row = {"pair": item["pair"], "lang": lang, "question": question, "error": err,
                           "latency_ms": ms}
                else:
                    diag = ans.diagnostics or {}
                    row = {
                        "pair": item["pair"], "lang": lang, "question": question,
                        "answer": ans.answer, "answer_language": ans.language,
                        "answer_chars": len(ans.answer),
                        "refused": bool(ans.refused),
                        "generation_method": ans.generation_method,
                        "latency_ms": float(ans.latency_ms.get("total") or 0.0),
                        "evidence_count": len(ans.evidence_chunks),
                        "answer_atoms": _atoms(ans.answer),
                        "english_term_mapping": diag.get("rewrite_added_terms") or [],
                        "base_query_rewritten": bool(diag.get("base_query_rewritten")),
                        "corpus_hint": diag.get("corpus_hint"),
                        "field_probe_fields": diag.get("field_probe_fields") or [],
                        "top_pages": [c.get("page") for c in ans.evidence_chunks[:6]],
                        "error": None,
                    }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")   # 边跑边 append
                print(f"  [{item['pair']}/{lang}] method={row.get('generation_method')} "
                      f"{row['latency_ms']:.0f}ms 原子={row.get('answer_atoms')}")

    # ---- 配对对比：同一信息需求在中英文下的数值一致性 ----
    summary: dict[str, Any] = {"工单": WORK_ORDER, "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "题对数": len(PAIRS), "逐题数": len(rows), "配对对比": []}
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair"]), {})[str(row["lang"])] = row
    consistent = 0
    for pair, langs in by_pair.items():
        zh, en = langs.get("zh") or {}, langs.get("en") or {}
        z_atoms, e_atoms = set(zh.get("answer_atoms") or []), set(en.get("answer_atoms") or [])
        shared = sorted(z_atoms & e_atoms)
        ok = bool(shared) or (not z_atoms and not e_atoms)
        consistent += 1 if ok else 0
        summary["配对对比"].append({
            "题目": pair,
            "中文答案数值": sorted(z_atoms), "英文答案数值": sorted(e_atoms),
            "共有数值": shared, "数值一致": ok,
            "中文耗时ms": zh.get("latency_ms"), "英文耗时ms": en.get("latency_ms"),
            "英文用了中文术语映射": bool(en.get("english_term_mapping")),
            "英文映射术语": en.get("english_term_mapping"),
        })
    summary["数值一致题对数"] = consistent
    summary["数值一致率"] = round(consistent / len(by_pair), 4) if by_pair else 0.0
    summary["判定"] = {"多语言可用": consistent == len(by_pair),
                       "英文走中文术语映射": all(bool((v.get("en") or {}).get("english_term_mapping"))
                                                for v in by_pair.values())}
    (TEST_DATA_DIR / "多语言汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print("=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
