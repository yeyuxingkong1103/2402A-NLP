# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 评估框架自检（``TC-E*`` 纯函数/纯文件用例）。

**本脚本不读索引、不调模型、不联网**，因此可以在 Ollama 忙时安全运行；
它校验的是「评估工具本身可信」（覆盖 ``设计/测试用例设计.md`` §5 的 ``TC-E03``~``TC-E08``、
``TC-E10``~``TC-E12``）：

======================  ==========================================================
用例编号                 核验内容
======================  ==========================================================
``TC-E03``              题库完整性：恰 17 行、字段齐备、id 唯一、``lang ∈ {zh,en}``
``TC-E05`` / ``TC-E06`` ``判分.py`` 内置自检（归一化 / 数字边界 / 拒答不判对）
``TC-E07``              汇总可重算：由 ``评估结果.jsonl`` 独立重算并与 ``评估汇总.json`` 比对
``TC-E08``              CSV 与 JSONL 对齐（行数、题号集合、关键字段）
``TC-E10``              缓存口径：``use_cache == false``
``TC-E11``              落盘五件齐备、UTF-8 可读、产物均在 ``测试/测试数据``
``TC-E12``              超线样本保留：``SLA.超时题号`` 每条都能在 JSONL 定位且 ``total_ms > 3000``
======================  ==========================================================

运行：
    python 测试\\回归测试\\test_评估框架自检.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _公共 as K  # noqa: E402

GROUP = "评估框架自检"
DATA = K.TEST_DATA_DIR
SUMMARY = DATA / "评估汇总.json"
JSONL = DATA / "评估结果.jsonl"
CSV_FILE = DATA / "评估结果.csv"
ANALYSIS = DATA / "检索问题分析.md"
RUN_LOG = DATA / "评估运行日志.txt"
QUESTION_SET = K.EVAL_DIR / "测试问题集.jsonl"

REQUIRED_Q_FIELDS = ("id", "question", "lang", "source_pdf", "source_file",
                     "expected_pages_hint", "reference_answer", "answer_keywords", "project")


def _rows() -> list[dict[str, Any]]:
    if not JSONL.exists():
        return []
    out = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def check_question_set(case: dict[str, Any]) -> None:
    """TC-E03 题库完整性。"""
    rows = [json.loads(l) for l in QUESTION_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [str(r.get("id")) for r in rows]
    missing = {f: [r.get("id") for r in rows if not r.get(f)] for f in REQUIRED_Q_FIELDS}
    bad_lang = [r.get("id") for r in rows if r.get("lang") not in ("zh", "en")]
    empty_kw = [r.get("id") for r in rows if not (r.get("answer_keywords") or [])]
    K.add_assert(case, "题库恰 17 行", len(rows) == 17, f"{len(rows)} 行")
    K.add_assert(case, "id 唯一", len(set(ids)) == len(ids), f"{len(set(ids))}/{len(ids)} 唯一")
    K.add_assert(case, "必备字段齐备", not any(missing.values()),
                 "齐备" if not any(missing.values()) else str({k: v for k, v in missing.items() if v})[:120])
    K.add_assert(case, "answer_keywords 非空", not empty_kw, f"空关键词题：{empty_kw or '无'}")
    K.add_assert(case, "lang ∈ {zh,en}", not bad_lang, f"非法 lang：{bad_lang or '无'}")
    case["关键实测"] = {"题数": len(rows), "题号": ids,
                     "关键词总数": sum(len(r.get("answer_keywords") or []) for r in rows)}


def check_judge_selftest(case: dict[str, Any]) -> None:
    """TC-E05 / TC-E06 判分器内置自检 + 归一化/数字边界断言。"""
    judge = K.judge_module()
    rc = judge._selftest()
    K.add_assert(case, "判分器内置自检全部通过", rc == 0, f"退出码 {rc}")

    forms = ["1,379.58", "1379.58", "１３７９.５８", "1 379.58", " 1,379.58 "]
    norms = {f: judge.normalize(f) for f in forms}
    loose = {f: judge._loose(judge.normalize(f)) for f in forms}
    # 实测口径：严格 normalize **保留单个空格**（防止相邻单元格数字粘连，见 判分.normalize docstring），
    # 因此「四种写法同串」要在**宽松层**（去空格）成立，并且**命中判定必须一致**——这才是可复核的判据。
    K.add_assert(case, "四种写法宽松归一化后同串", len(set(loose.values())) == 1,
                 json.dumps(loose, ensure_ascii=False))
    K.add_assert(case, "四种写法命中判定一致（均命中 1379.58）",
                 all(judge.text_matches(f, "1379.58") for f in forms),
                 json.dumps({f: judge.text_matches(f, "1379.58") for f in forms}, ensure_ascii=False))
    K.add_assert(case, "严格归一化千分位/全角已归并",
                 norms["1,379.58"] == norms["1379.58"] == norms["１３７９.５８"] == "1379.58",
                 json.dumps(norms, ensure_ascii=False))
    K.add_assert(case, "数字边界：11.65 不命中 1.65",
                 not judge.text_matches("11.65", "1.65"), "不命中")
    K.add_assert(case, "数字边界：18.25 不命中 18.2",
                 not judge.text_matches("18.25", "18.2"), "不命中")
    K.add_assert(case, "正常数值仍命中（18.2% 命中 18.2%）",
                 judge.text_matches("同比增长18.2%", "18.2%"), "命中")
    # 拒答不得算判对
    K.add_assert(case, "拒答不计判对",
                 not judge.is_correct(1.0, True, 0.6), "coverage=1.0 且 refused=True ⇒ False")
    case["关键实测"] = {"严格归一化": norms, "宽松归一化": loose, "自检退出码": rc}


def check_recompute(case: dict[str, Any]) -> None:
    """TC-E07 由逐题记录独立重算汇总指标并与 评估汇总.json 比对。"""
    rows = _rows()
    valid = [r for r in rows if not r.get("error")]
    summ = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    n = len(valid)
    if not n:
        K.add_assert(case, "存在逐题记录", False, "评估结果.jsonl 无有效行")
        return
    judge_of = [(r.get("judge") or {}) for r in valid]
    strict = sum(1 for r in valid if (r.get("judge") or {}).get("is_correct"))
    partial = sum(1 for r in valid if (r.get("judge") or {}).get("partial_correct"))
    cov = sum(float(j.get("keyword_coverage") or 0) for j in judge_of) / n
    r5 = sum(float(j.get("recall_at_5") or 0) for j in judge_of) / n
    r10 = sum(float(j.get("recall_at_10") or 0) for j in judge_of) / n
    sla = sum(1 for r in valid if r.get("sla_ok")) / n
    refused = sum(1 for r in valid if r.get("refused")) / n
    # 注意：判分.py 的 page_hit 是 {"hit": bool, ...}，必须解包后再取真值
    page_hit = sum(1 for j in judge_of if (j.get("page_hit") or {}).get("hit")) / n

    acc = summ.get("准确率") or {}
    rec = summ.get("检索召回") or {}
    sla_s = summ.get("SLA") or {}
    refus = summ.get("拒答") or {}
    page = summ.get("页命中") or {}
    pairs = [
        ("严格准确率", round(strict / n, 4), acc.get("严格准确率")),
        ("宽松准确率", round((strict + partial) / n, 4), acc.get("宽松准确率(严格+部分)")),
        ("平均关键词覆盖率", round(cov, 4), acc.get("平均关键词覆盖率")),
        ("平均recall@5", round(r5, 4), rec.get("平均recall@5")),
        ("平均recall@10", round(r10, 4), rec.get("平均recall@10")),
        ("SLA达标率", round(sla, 4), sla_s.get("达标率")),
        ("拒答率", round(refused, 4), refus.get("拒答率")),
        ("页命中率", round(page_hit, 4), page.get("命中率")),
    ]
    diffs = [f"{k}: 重算{v1} vs 汇总{v2}" for k, v1, v2 in pairs
             if v2 is None or abs(float(v1) - float(v2)) > 1e-4]
    K.add_assert(case, "逐字段可重算一致", not diffs, "一致" if not diffs else "; ".join(diffs)[:200])
    case["关键实测"] = {"题数": n, "重算": {k: v1 for k, v1, _ in pairs},
                     "汇总": {k: v2 for k, _, v2 in pairs}}


def check_csv_jsonl(case: dict[str, Any]) -> None:
    """TC-E08 CSV 与 JSONL 对齐。"""
    rows = _rows()
    if not CSV_FILE.exists():
        K.add_assert(case, "CSV 存在", False, str(CSV_FILE))
        return
    with CSV_FILE.open(encoding="utf-8-sig", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
        fh.seek(0)
        header = next(csv.reader(fh))
    K.add_assert(case, "行数与 JSONL 一致", len(csv_rows) == len(rows),
                 f"CSV {len(csv_rows)} 行 / JSONL {len(rows)} 行")
    ids_csv = {r.get("题号") for r in csv_rows}
    ids_jsonl = {str(r.get("id")) for r in rows}
    K.add_assert(case, "题号集合一致", ids_csv == ids_jsonl,
                 f"CSV-only={sorted(ids_csv - ids_jsonl)} JSONL-only={sorted(ids_jsonl - ids_csv)}")
    col_count = len(header)
    K.add_assert(case, "CSV 列数与表头一致", col_count == len(header), f"{col_count} 列")
    # 关键字段逐行比对
    by_id = {str(r.get("id")): r for r in rows}
    mismatch: list[str] = []
    for crow in csv_rows:
        qid = str(crow.get("题号"))
        row = by_id.get(qid)
        if not row:
            continue
        j = row.get("judge") or {}
        for csv_key, actual in (("是否判对", bool(j.get("is_correct"))),
                                ("关键词覆盖率", float(j.get("keyword_coverage") or 0)),
                                ("recall@5", float(j.get("recall_at_5") or 0)),
                                ("recall@10", float(j.get("recall_at_10") or 0)),
                                ("页命中", bool((j.get("page_hit") or {}).get("hit"))),
                                ("是否拒答", bool(row.get("refused")))):
            raw = (crow.get(csv_key) or "").strip()
            if csv_key == "关键词覆盖率":
                ok = abs(float(raw or 0) - actual) < 1e-4
            elif csv_key in ("recall@5", "recall@10"):
                ok = abs(float(raw or 0) - actual) < 1e-4
            else:
                ok = (raw == "True") == actual
            if not ok:
                mismatch.append(f"{qid}.{csv_key}: CSV={raw} vs JSONL={actual}")
    K.add_assert(case, "关键字段逐行一致", not mismatch,
                 "一致" if not mismatch else "; ".join(mismatch[:4])[:200])
    case["关键实测"] = {"CSV列数": col_count, "行数": len(csv_rows), "字段 mismatch": len(mismatch)}


def check_artifacts(case: dict[str, Any]) -> None:
    """TC-E10 / TC-E11 / TC-E12 落盘五件、缓存口径、超线样本保留。"""
    files = {"评估结果.jsonl": JSONL, "评估结果.csv": CSV_FILE, "评估汇总.json": SUMMARY,
             "检索问题分析.md": ANALYSIS, "评估运行日志.txt": RUN_LOG}
    missing = [name for name, path in files.items() if not path.exists()]
    K.add_assert(case, "五件产物齐备", not missing, f"缺失：{missing or '无'}")
    K.add_assert(case, "产物均落在 测试/测试数据", all(p.parent == DATA for p in files.values()),
                 str(DATA.relative_to(K.REPO_ROOT)))
    enc_ok, enc_bad = True, []
    for name, path in files.items():
        if not path.exists():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            enc_ok = False
            enc_bad.append(f"{name}: {exc}")
    K.add_assert(case, "JSONL/JSON/MD/日志 为 UTF-8 可读", enc_ok, "可读" if enc_ok else "; ".join(enc_bad))
    csv_bom = CSV_FILE.exists() and CSV_FILE.read_bytes()[:3] == b"\xef\xbb\xbf"
    K.add_assert(case, "CSV 为 utf-8-sig（带 BOM）", csv_bom, "带 BOM" if csv_bom else "无 BOM")

    summ = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    use_cache = (summ.get("配置") or {}).get("use_cache")
    K.add_assert(case, "评估口径 use_cache == false", use_cache is False, str(use_cache))
    rows = _rows()
    zero_latency = [r.get("id") for r in rows if float((r.get("latency_ms") or {}).get("total") or 0) <= 0]
    K.add_assert(case, "无 0 毫秒级缓存命中记录", not zero_latency, f"异常题：{zero_latency or '无'}")

    timeout_ids = (summ.get("SLA") or {}).get("超时题号") or []
    unresolved = [qid for qid in timeout_ids
                  if not any(str(r.get("id")) == qid and float(r.get("total_ms") or 0) > 3000 for r in rows)]
    K.add_assert(case, "超线题号均可回溯（样本保留）", not unresolved,
                 f"超线 {len(timeout_ids)} 题，未回溯 {unresolved or '无'}")
    K.add_assert(case, "运行日志非空", RUN_LOG.exists() and RUN_LOG.stat().st_size > 0,
                 f"{RUN_LOG.stat().st_size if RUN_LOG.exists() else 0} 字节")
    case["关键实测"] = {"use_cache": use_cache, "超时题号": timeout_ids,
                     "日志字节": RUN_LOG.stat().st_size if RUN_LOG.exists() else 0}


CHECKS = [
    ("TC-E03", "题库完整性", check_question_set),
    ("TC-E05/E06", "判分器自检 + 归一化/数字边界", check_judge_selftest),
    ("TC-E07", "汇总指标可重算", check_recompute),
    ("TC-E08", "CSV 与 JSONL 对齐", check_csv_jsonl),
    ("TC-E10/E11/E12", "缓存口径 / 落盘留痕 / 超线保留", check_artifacts),
]


def run(verbose: bool = True) -> dict[str, Any]:
    K.bootstrap_env()
    import time

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for case_id, name, fn in CHECKS:
        case = K.new_case(case_id, name, group=GROUP, expect="核验通过")
        try:
            fn(case)
        except Exception as exc:  # noqa: BLE001
            import traceback

            case["异常"] = f"{type(exc).__name__}: {exc}"
            case["异常堆栈"] = traceback.format_exc()[-1500:]
            K.add_assert(case, "核验过程未抛异常", False, case["异常"])
        K.finalize_case(case)
        cases.append(case)
        if verbose:
            flag = "PASS" if case["通过"] else "FAIL"
            print(f"[{flag}] {case_id} {name}")
            for item in case["断言"]:
                print(f"        [{'OK ' if item['通过'] else 'NG '}] {item['名称']} ⇒ "
                      f"{str(item['实测'])[:150]}")
    elapsed = (time.perf_counter() - started) * 1000
    result = K.summarize(GROUP, cases, elapsed)
    if verbose:
        K.print_case_table(cases, f"评估框架自检（{len(cases)} 组核验）")
        print(f"通过 {result['通过']} / {result['用例数']}，失败 {result['失败']}")
    return result


def main() -> int:
    facts = K.env_facts()
    print(f"== {K.WORK_ORDER} / 评估框架自检（不读索引、不调模型）==")
    print(f"解释器: {facts['解释器']} ({facts['解释器版本']})")
    result = run()
    payload = {"工单": K.WORK_ORDER, "生成时间": K.now_str(), "环境": facts, "组": result}
    K.save_json(K.TEST_DATA_DIR / "回归_评估框架自检.json", payload)
    print(f"\n结果已落盘：{K.TEST_DATA_DIR / '回归_评估框架自检.json'}")
    return 0 if result["失败"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
