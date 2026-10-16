# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

多轮对话**自检**（一键端到端体检）—— 对应工单4 的 `部署/脚本/run_app.ps1 -HealthCheckOnly`，
但体检对象是**多轮链路**。

做三件事：
    1. 起引擎并预热，记录冷启动与预热耗时；
    2. 跑工单指定的 **5 轮中文多轮序列**，用确定性判分器判定；
    3. 跑 **中英双语多轮序列**（英文单轮 / 英文指代 / 英文省略式换主体），
       验证「支持中英文问答」这一条。

落盘：
    * ``部署/日志/多轮自检.json`` —— 结构化自检结果（供脚本断言与交付核验）
    * ``测试/测试数据/multiturn_bilingual.jsonl`` —— 中英双语逐轮实测明细

用法：
    pwsh -NoProfile -File run_py.ps1 测试/多轮/多轮自检.py
退出码：0 = 自检通过；3 = 链路不可用；1 = 存在未通过项。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

EXIT_OK, EXIT_FAIL, EXIT_UNAVAILABLE = 0, 1, 3

#: 工单指定的中文多轮序列（逐字，不得改写）
ZH_SEQ: list[dict[str, Any]] = [
    {"id": "Q1", "session": "zh",
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "must_contain": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
     "expected_strategy": "none"},
    {"id": "Q2", "session": "zh",
     "question": "他参与的哪个工程荣获了国家科技进步一等奖？",
     "must_contain": ["国家科技进步一等奖"], "expected_strategy": "pronoun"},
    {"id": "Q3", "session": "zh",
     "question": "这个公司的法定代表人是谁？",
     "must_contain": ["程家明"], "expected_strategy": "pronoun"},
    {"id": "Q4", "session": "zh",
     "question": "那武汉力源信息技术股份有限公司呢？",
     "must_contain": ["赵马克"], "expected_strategy": "ellipsis_entity_switch",
     "expected_rewrite": "武汉力源信息技术股份有限公司的法定代表人是谁？"},
    {"id": "Q5", "session": "zh",
     "question": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
     "must_contain": ["大客户销售部", "北京销售处", "深圳销售处", "广州销售处",
                      "成都销售处", "珠海销售处", "武汉销售处"],
     "must_not_contain": ["上海销售处", "7个销售处"], "expected_strategy": "none"},
]

#: 中英双语多轮序列（**英文名取自语料原文**：招股说明书里披露的英文公司名）
#:
#: ⚠️ 断言口径说明：英文路径**用英文作答**，人名会以**拼音**出现
#: （「程家明」→ "Cheng Jiaming"、「赵马克」→ "Zhao Mak"）。
#: 因此这里用 ``must_contain_any`` 同时接受中文与拼音写法 ——
#: 若只写中文人名会把**正确答案判成失败**（本脚本第一版就踩了这个坑）。
EN_SEQ: list[dict[str, Any]] = [
    {"id": "E1", "session": "en",
     "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
     "must_contain_any": ["5,520"], "expected_strategy": "none",
     "note": "英文单轮 + 显式实体：验证英文问答路径可用（兴图新科注册资本 5,520.00 万元，"
             "与 golden 543 一致）"},
    {"id": "E2", "session": "en",
     "question": "Who is its legal representative?",
     "must_contain_any": ["Cheng Jiaming", "程家明"], "expected_strategy": "pronoun",
     "note": "英文指代消解：「its」→ 焦点实体（兴图新科）"},
    {"id": "E3", "session": "en",
     "question": "What about Wuhan P&S Information Technology Co., Ltd.?",
     "must_contain_any": ["Zhao Mak", "Mark Zhao", "赵马克"],
     "expected_strategy": "ellipsis_entity_switch",
     "note": "英文省略式 + 换主体：复用上一轮问句骨架，主体换成力源"},
]


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_seq(engine: Any, cases: list[dict[str, Any]], *, use_cache: bool) -> list[dict[str, Any]]:
    """按 session 顺序跑一段序列，返回逐轮实测明细。"""
    out: list[dict[str, Any]] = []
    for case in cases:
        t0 = time.perf_counter()
        res = engine.ask(case["question"], session_id=case["session"], use_cache=use_cache)
        wall = round((time.perf_counter() - t0) * 1000, 2)
        ans = res.answer or ""
        missing = [m for m in case.get("must_contain", []) if m not in ans]
        any_of = case.get("must_contain_any") or []
        any_hit = (not any_of) or any(a in ans for a in any_of)
        forbidden = [m for m in case.get("must_not_contain", []) if m in ans]
        str_ok = res.strategy == case.get("expected_strategy")
        exp_rw = case.get("expected_rewrite")
        rw_ok = (exp_rw is None) or (exp_rw in res.rewritten_query)
        passed = ((not missing) and any_hit and (not forbidden)
                  and str_ok and rw_ok and bool(ans.strip()))
        out.append({
            "id": case["id"], "session": case["session"], "question": case["question"],
            "rewritten_query": res.rewritten_query, "strategy": res.strategy,
            "expected_strategy": case.get("expected_strategy"),
            "focus_entity": res.focus_entity, "answer": ans, "refused": res.refused,
            "cited_pages": res.cited_pages, "wall_ms": wall,
            "latency_ms": res.latency_ms, "missing": missing,
            "must_contain_any": any_of, "any_of_hit": any_hit,
            "forbidden": forbidden, "strategy_ok": str_ok, "rewrite_ok": rw_ok,
            "passed": passed, "note": case.get("note", ""),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"多轮对话自检（工单：{WORK_ORDER}）")
    ap.add_argument("--no-cache", action="store_true", help="禁用缓存（延迟测量口径）")
    ap.add_argument("--json-out", default=None, help="自检结果落盘路径")
    args = ap.parse_args(argv)

    json_out = Path(args.json_out) if args.json_out else (REPO_ROOT / "部署" / "日志" / "多轮自检.json")
    bilingual_out = REPO_ROOT / "测试" / "测试数据" / "multiturn_bilingual.jsonl"
    use_cache = not args.no_cache

    started_at = datetime.now().isoformat(timespec="seconds")
    print("=" * 78)
    print(f"工单5 多轮对话自检  |  工单：{WORK_ORDER}")
    print("=" * 78)

    try:
        from app.multiturn_engine import MultiTurnEngine
    except ImportError as exc:  # noqa: BLE001
        print(f"[不可用] 无法导入多轮引擎：{exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    t0 = time.perf_counter()
    try:
        engine = MultiTurnEngine()
        cold_ms = round((time.perf_counter() - t0) * 1000, 2)
        warm = engine.warmup(with_generate=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[不可用] 引擎初始化/预热失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    print(f"[预热] 索引 {warm['base'].get('index_chunks')} 块，实体 {warm['entities']} 个，"
          f"预热 {warm['elapsed_ms']:.0f} ms（构造 {cold_ms:.0f} ms）")

    # 清一遍会话，保证从干净状态开始
    for sid in ("zh", "en"):
        engine.reset(sid)

    zh = _run_seq(engine, ZH_SEQ, use_cache=use_cache)
    print("\n-- 中文多轮序列（工单指定 5 轮）--")
    for r in zh:
        flag = "✅" if r["passed"] else "❌"
        print(f"  [{flag}] {r['id']} {r['wall_ms']:7.1f}ms strategy={r['strategy']:24s} "
              f"| {r['question'][:30]}")
        if not r["passed"]:
            print(f"        改写: {r['rewritten_query']}")
            print(f"        答案: {r['answer'][:110]!r}")
            if r["missing"]:
                print(f"        缺: {r['missing']}  禁含命中: {r['forbidden']}")

    engine.reset("en")
    en = _run_seq(engine, EN_SEQ, use_cache=use_cache)
    print("\n-- 中英双语多轮序列（英文单轮 / 指代 / 省略式）--")
    for r in en:
        flag = "✅" if r["passed"] else "❌"
        print(f"  [{flag}] {r['id']} {r['wall_ms']:7.1f}ms strategy={r['strategy']:24s} | {r['question'][:46]}")
        print(f"        改写: {r['rewritten_query'][:110]}")
        print(f"        答案: {r['answer'][:100]!r}")
        if not r["passed"]:
            print(f"        缺: {r['missing']}  策略期望 {r['expected_strategy']}")

    zh_ok = sum(1 for r in zh if r["passed"])
    en_ok = sum(1 for r in en if r["passed"])
    all_ok = zh_ok == len(zh) and en_ok == len(en)

    payload = {
        "工单": WORK_ORDER,
        "脚本": "测试/多轮/多轮自检.py",
        "开始时间": started_at,
        "结束时间": datetime.now().isoformat(timespec="seconds"),
        "解释器": sys.executable,
        "python": sys.version.split()[0],
        "缓存": use_cache,
        "预热": {"index_chunks": warm["base"].get("index_chunks"),
                 "entities": warm["entities"], "elapsed_ms": warm["elapsed_ms"],
                 "构造耗时_ms": cold_ms},
        "中文多轮": {"通过": zh_ok, "总数": len(zh), "明细": zh},
        "中英双语": {"通过": en_ok, "总数": len(en), "明细": en},
        "汇总": {"中文多轮": f"{zh_ok}/{len(zh)}", "中英双语": f"{en_ok}/{len(en)}",
                 "全部通过": all_ok},
        "结论": "自检通过" if all_ok else "存在未通过项（详见明细）",
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    _jsonl_write(bilingual_out, [
        {k: v for k, v in r.items() if k in
         ("id", "session", "question", "rewritten_query", "strategy", "expected_strategy",
          "focus_entity", "answer", "refused", "cited_pages", "wall_ms", "passed", "note")}
        for r in en])

    print("\n" + "=" * 78)
    print(f"  中文多轮：{zh_ok}/{len(zh)}　中英双语：{en_ok}/{len(en)}"
          f"　=> {'自检通过' if all_ok else '存在未通过项'}")
    print(f"  产物：{json_out}")
    print(f"        {bilingual_out}")
    print("=" * 78)
    return EXIT_OK if all_ok else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
