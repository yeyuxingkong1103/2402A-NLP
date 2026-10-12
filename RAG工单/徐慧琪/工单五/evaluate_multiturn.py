# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""多轮评测：工单3 单轮系统 / 工单5 多轮系统 / 纯 LLM 三方对比。

三路各拿到什么（口径必须写清，否则数字没有意义）：

  · **工单3 单轮系统**（优化前基线）—— 只读子进程调用 `D:\\xinzg6\\zy-gq\\工单3`
    的 `rag.ask`（其 collection 为 `zhaogu_v3_table`）。喂进去的是**本测试集每轮的
    原始问题**（含「他…」「这个公司…」「那 X 呢？」这类跟进句），**不带任何对话状态**
    ——它的 API 也没有多轮入口。这是「优化前」的真实能力边界，不是把它做残。
    时延对比前先跑一条**丢弃结果**的预热题，剔除子进程模型冷启动（实测首题 +28.7s），
    与工单5 的 `rag.warmup()` 口径对等。
  · **工单5 多轮系统** —— `rag.ask_turn(question, state)`，逐会话维护 DialogueState，
    本轮问题先经 coref 改写再检索生成。
  · **纯 LLM（无检索）** —— 同一批原始问题直接问 qwen2.5:3b，无检索、无历史。

指标：逐轮关键事实命中率、改写准确率（含/不含措辞）、指代消解实体准确率、
      「完整问题不误触发改写」率、无候选时的反问率、响应时间 p50/p90/≤3s 占比、
      引用溯源率、改写降级率（走 L4 规则兜底的比例）。

计时口径（硬约束 A2）：`timings["total_ms"]` **已含**多轮改写延迟，直接读，
**不**再加 `rewrite["latency_ms"]`（叠加会虚增 ~200ms/代词轮）。

用法：
    python evaluate_multiturn.py                 # 三方全跑 → data/eval/multiturn_compare.json
    python evaluate_multiturn.py --skip-baseline # 跳过工单3 子进程
    python evaluate_multiturn.py --skip-pure-llm # 跳过纯 LLM 一路
"""
from src import bootstrap  # noqa: F401  —— 必须最先导入（离线 + 大栈，缺失会 exit 139）

import argparse
import json
import os
import re
import statistics
import sys
import time

from src import baseline, config, dialogue, llm, rag

DATA_FILE = os.path.join(config.ROOT, "data", "test_dialogues.json")
UNRESOLVED = "<unresolved>"

_SYS_ONLY_LLM = ("你是招股说明书问答助手。仅凭你的知识回答。"
                 "若不知道确切答案，请直接回答“不确定”，不要编造。")


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    """去空白与中英标点（保留 % 与小数点），用于宽松包含判定。"""
    return re.sub(r"[\s，,、。；;：:（）()【】\[\]“”\"']", "", s or "")


def _facts_hit(answer: str, facts: list[str]) -> tuple[int, int]:
    a = _norm(answer)
    hit = sum(1 for f in facts if _norm(f) in a)
    return hit, len(facts)


def _norm_ms(timings: dict) -> float:
    """上游（工单3）报 total_s，本工单报 total_ms——统一到毫秒，兼容两种口径。"""
    if not isinstance(timings, dict):
        return 0.0
    if timings.get("total_ms"):
        return float(timings["total_ms"])
    if timings.get("total_s"):
        return float(timings["total_s"]) * 1000.0
    return 0.0


def _resolved_ok(expected: str, resolved: str) -> bool:
    """改写准确率（严格口径）：改写结果**包含**期望的独立问题（或与之等价）。

    期望问题本身对措辞敏感（英文轮尤甚），所以另有一路 entity_ok
    只判定「有没有绑到正确的公司」，两个数字都报，避免措辞差异掩盖真实能力。
    """
    if expected == UNRESOLVED:
        return False                      # 由 unresolved_ok 单独判定
    e, g = _norm(expected), _norm(resolved)
    return bool(g) and (e in g or g == e)


# ---------------------------------------------------------------------------
# 第 1 路：工单5 多轮系统
# ---------------------------------------------------------------------------
def score_multiturn(data: dict) -> list[dict]:
    rows: list[dict] = []
    for d in data["dialogues"]:
        dialogue.reset_all()
        st = dialogue.get_state(d["dialogue_id"])
        for t in d["turns"]:
            r, turn = rag.ask_turn(t["question"], st)
            facts = t.get("expected_answer_facts") or []
            hit, n = _facts_hit(r.answer, facts) if facts else (0, 0)
            expected = t["expected_resolved"]
            adversarial = expected == UNRESOLVED
            rows.append({
                "dialogue": d["dialogue_id"], "turn": t["turn"],
                "coref_type": t["coref_type"], "system": "v5_multiturn",
                "question": t["question"],
                "expected_resolved": expected,
                "expected_entity": t.get("expected_entity", ""),
                "resolved": turn.resolved_question,
                # 对抗轮本就**不该**产出改写，不计入改写准确率，由 unresolved_ok 单独判定
                "resolved_ok": (None if adversarial
                                else _resolved_ok(expected, turn.resolved_question)),
                "entity_ok": (None if adversarial or not t.get("expected_entity")
                              else t["expected_entity"] in turn.resolved_question),
                "unresolved_ok": ((turn.rewrite_mode == "unresolved"
                                   and r.mode == "clarify") if adversarial else None),
                "answer": r.answer, "facts_hit": hit, "facts_total": n,
                "mode": r.mode,
                "total_ms": _norm_ms(r.timings),
                # A2：total_ms 已含改写延迟，这里只留痕，不再叠加
                "rewrite_ms": (r.rewrite or {}).get("latency_ms"),
                "rewrite_mode": turn.rewrite_mode,
                "degraded": bool(turn.rewrite_mode.endswith("_rule")),
                "citation_ok": bool(r.citations),
            })
    return rows


# ---------------------------------------------------------------------------
# 第 2 路：工单3 单轮系统（只读子进程）
# ---------------------------------------------------------------------------
# 基线子进程的**预热题**：工单3 的模型是首题懒加载的，实测第一题要付 ~28.7s
# （进程内加载 bge-m3 + reranker），那是进程冷启动而不是「单轮响应时间」。
# 本工单一路有 `rag.warmup()`，基线也必须有对等的一次性预热，否则 p50/p90/≤3s
# 的对比就是拿「冷启动」比「热启动」——那不是基线的真实水平（A3）。
# 预热题的**结果整条丢弃**，不进任何指标；也不改变 20 道正式题的输入与顺序。
BASELINE_WARMUP_QUESTION = "公司注册资本是多少？"


def baseline_questions(data: dict) -> list[str]:
    """基线拿到的输入：每轮的**原始问题**，逐轮独立，无历史（与其它两路同题同序）。"""
    return [t["question"] for d in data["dialogues"] for t in d["turns"]]


def score_baseline_single(data: dict, results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    i = 0
    for d in data["dialogues"]:
        for t in d["turns"]:
            b = results[i] if i < len(results) else {"error": "结果缺失"}
            i += 1
            facts = t.get("expected_answer_facts") or []
            answer = b.get("answer") or ""
            hit, n = _facts_hit(answer, facts) if facts else (0, 0)
            rows.append({
                "dialogue": d["dialogue_id"], "turn": t["turn"],
                "coref_type": t["coref_type"], "system": "v3_single_turn",
                "question": t["question"],
                "expected_resolved": t["expected_resolved"],
                "expected_entity": t.get("expected_entity", ""),
                # 基线没有改写环节：给它的就是原始跟进问题，记 n/a 而不是 0
                "resolved": t["question"], "resolved_ok": None, "entity_ok": None,
                "unresolved_ok": None,
                "answer": answer,
                "error": b.get("error"),
                "facts_hit": hit, "facts_total": n,
                "mode": "single_turn",
                "total_ms": _norm_ms(b.get("timings") or {}),
                "citation_ok": bool(b.get("citations")),
            })
    return rows


# ---------------------------------------------------------------------------
# 第 3 路：纯 LLM（无检索、无历史）
# ---------------------------------------------------------------------------
def score_pure_llm(data: dict) -> list[dict]:
    rows: list[dict] = []
    for d in data["dialogues"]:
        for t in d["turns"]:
            t0 = time.time()
            err = None
            try:
                resp = llm.chat([{"role": "system", "content": _SYS_ONLY_LLM},
                                 {"role": "user", "content": t["question"]}],
                                temperature=0.0, num_predict=256)
                ans = (resp.get("answer") or "").strip()
            except Exception as exc:                        # noqa: BLE001
                ans, err = f"<error: {exc}>", repr(exc)
            facts = t.get("expected_answer_facts") or []
            hit, n = _facts_hit(ans, facts) if facts else (0, 0)
            rows.append({
                "dialogue": d["dialogue_id"], "turn": t["turn"],
                "coref_type": t["coref_type"], "system": "pure_llm",
                "question": t["question"],
                "expected_resolved": t["expected_resolved"],
                "expected_entity": t.get("expected_entity", ""),
                "resolved": None, "resolved_ok": None, "entity_ok": None,
                "unresolved_ok": None,
                "answer": ans, "error": err,
                "facts_hit": hit, "facts_total": n, "mode": "llm_only",
                "total_ms": round((time.time() - t0) * 1000, 1),
                "citation_ok": False,
            })
    return rows


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _pct(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _rate(rows: list[dict], key: str) -> float | None:
    """只对**声明了该指标**的行求比例（None = 该路没有这个环节，记 n/a）。"""
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(rows: list[dict]) -> dict:
    systems = sorted({r["system"] for r in rows})
    by_system: dict = {}
    for s in systems:
        sub = [r for r in rows if r["system"] == s]
        ms = sorted(r["total_ms"] for r in sub if r["total_ms"])
        scored = [r for r in sub if r["facts_total"]]
        hits = sum(r["facts_hit"] for r in scored)
        total = sum(r["facts_total"] for r in scored)
        n = len(ms)
        by_system[s] = {
            "n_turns": len(sub),
            "n_error": sum(1 for r in sub if r.get("error")),
            # 事实命中率：只统计有标准事实的轮次（对抗轮无事实可判，各路口径一致）
            "n_fact_turns": len(scored),
            "fact_accuracy": _pct(hits, total),
            "rewrite_accuracy": _rate(sub, "resolved_ok"),
            # 严格改写准确率把英文措辞差异也算失败，故单列实体绑定准确率
            "entity_accuracy": _rate(sub, "entity_ok"),
            "clarify_accuracy": _rate(sub, "unresolved_ok"),
            "no_rewrite_accuracy": _rate([r for r in sub if r["coref_type"] == "none"],
                                         "resolved_ok"),
            "degraded_ratio": (_pct(sum(1 for r in sub if r.get("degraded")), len(sub))
                               if any("degraded" in r for r in sub) else None),
            "p50_ms": round(statistics.median(ms), 1) if ms else None,
            "p90_ms": round(ms[min(int(len(ms) * 0.9), len(ms) - 1)], 1) if ms else None,
            "under_3s_ratio": _pct(sum(1 for m in ms if m <= 3000), n),
            "citation_ratio": _rate(sub, "citation_ok"),
        }
    # 按指代类型分组：本工单最直观的证据面
    by_type: dict = {}
    for r in rows:
        cell = by_type.setdefault(r["coref_type"], {})
        slot = cell.setdefault(r["system"], {"hit": 0, "total": 0, "n": 0})
        slot["n"] += 1
        slot["hit"] += r["facts_hit"]
        slot["total"] += r["facts_total"]
    by_type_fact = {k: {s: _pct(v["hit"], v["total"]) for s, v in d2.items()}
                    for k, d2 in by_type.items()}
    by_type_v5 = {}
    for k, d2 in by_type.items():
        v5 = [r for r in rows if r["coref_type"] == k and r["system"] == "v5_multiturn"]
        by_type_v5[k] = {
            "n": len(v5),
            "rewrite": _rate(v5, "resolved_ok"),
            "entity": _rate(v5, "entity_ok"),
            "rewrite_modes": sorted({r.get("rewrite_mode") for r in v5}),
        }
    return {"by_system": by_system, "by_coref_type_fact": by_type_fact,
            "by_coref_type_v5_rewrite": by_type_v5}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args) -> dict:
    with open(DATA_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    n_turns = sum(len(d["turns"]) for d in data["dialogues"])
    print(f"测试集：{len(data['dialogues'])} 个会话 / {n_turns} 轮", flush=True)

    rows: list[dict] = []

    # ---- Pass 1：工单3 单轮基线（子进程，先跑） ----
    # 必须先跑：此时本进程尚未加载 bge-m3 / reranker。实测反例见 evaluate_compare.py
    # Pass 1/2 注释——先加载大模型再起子进程，会因页面文件/显存不足让子进程崩掉，
    # 那样得到的是「基线跑不起来」的假数据，不是基线的真实水平。
    base_info = baseline.info(config.GY3_ROOT, config.GY3_COLLECTION)
    if not args.skip_baseline:
        print(f"[Pass 1] 工单3 单轮基线：{base_info}", flush=True)
        if not base_info["available"]:
            print("  ⚠ 工单3 基线不可用（目录/索引缺失），跳过该路", flush=True)
        else:
            try:
                llm.unload()
                import torch
                torch.cuda.empty_cache()
                time.sleep(2.0)
            except Exception:                               # noqa: BLE001
                pass
            t0 = time.time()
            qs = baseline_questions(data)
            batch = baseline.ask_baseline_batch([BASELINE_WARMUP_QUESTION] + qs,
                                                root=config.GY3_ROOT)
            warm_row, results = batch[0], batch[1:]      # 预热行丢弃（见上方注释）
            n_err = sum(1 for b in results if b.get("error"))
            print(f"  子进程完成 {len(results)} 轮 | {time.time() - t0:.1f}s | 失败 {n_err}"
                  f" | 预热题耗时 {_norm_ms(warm_row.get('timings') or {}):.0f}ms（已丢弃）",
                  flush=True)
            rows += score_baseline_single(data, results)

    # ---- Pass 2：工单5 多轮系统 ----
    warm = {"skipped": True}
    if not args.skip_warmup:
        warm = rag.warmup()
        print(f"[warmup] {warm}", flush=True)
    if not args.skip_multiturn:
        v5 = score_multiturn(data)
        bad = [r for r in v5 if r.get("mode") == "error"]
        print(f"[Pass 2] 工单5 多轮：{len(v5)} 轮完成，检索失败 {len(bad)}", flush=True)
        rows += v5

    # ---- Pass 3：纯 LLM（无检索） ----
    if not args.skip_pure_llm:
        p = score_pure_llm(data)
        print(f"[Pass 3] 纯 LLM：{len(p)} 轮完成", flush=True)
        rows += p

    summary = summarize(rows)
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    payload = {
        "workorder": config.WORKORDER_ID,
        "dataset": os.path.basename(DATA_FILE),
        "n_dialogues": len(data["dialogues"]),
        "n_turns": n_turns,
        "arms": {
            "v3_single_turn": {
                "desc": "工单3 系统（优化前）：只读子进程调用其 rag.ask",
                "input": "每轮原始问题，逐轮独立，无对话状态",
                "root": config.GY3_ROOT, "collection": config.GY3_COLLECTION,
                "available": base_info["available"],
                "warmup_question": BASELINE_WARMUP_QUESTION,
                "warmup_note": "与工单5 的 rag.warmup() 对等的一次性预热；"
                               "该题结果已丢弃、不计入任何指标，只剔除基线的进程冷启动",
            },
            "v5_multiturn": {
                "desc": "工单5 系统：带 DialogueState 的 ask_turn 链路",
                "input": "每轮原始问题 + 会话对话状态（焦点公司/上一信息点/历史窗口）",
                "collection": config.COLLECTION_OPT,
            },
            "pure_llm": {
                "desc": "纯 LLM（无检索）：qwen2.5:3b 直接作答",
                "input": "每轮原始问题，无检索、无历史",
                "model": config.OLLAMA_MODEL,
            },
        },
        "timing_note": "total_ms 已含多轮改写延迟（A2），未叠加 rewrite.latency_ms",
        "warmup": warm,
        "summary": summary,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return payload


def _print_report(payload: dict) -> None:
    s = payload["summary"]["by_system"]

    def fmt(v):
        return "n/a" if v is None else v

    print("\n===== 三方对比（逐轮口径，n=%d）=====" % payload["n_turns"])
    head = f"{'指标':<22}" + "".join(f"{k:>18}" for k in sorted(s))
    print(head)
    for key, label in (("n_turns", "轮数"), ("n_error", "系统级错误"),
                       ("fact_accuracy", "事实命中率"),
                       ("rewrite_accuracy", "改写准确率(严格)"),
                       ("entity_accuracy", "指代实体准确率"),
                       ("clarify_accuracy", "无候选反问率"),
                       ("no_rewrite_accuracy", "首轮不误改写率"),
                       ("degraded_ratio", "改写降级率"),
                       ("p50_ms", "p50 耗时(ms)"), ("p90_ms", "p90 耗时(ms)"),
                       ("under_3s_ratio", "≤3s 占比"),
                       ("citation_ratio", "引用溯源率")):
        print(f"{label:<22}" + "".join(f"{str(fmt(s[k].get(key))):>18}" for k in sorted(s)))

    print("\n----- 按指代类型：事实命中率 -----")
    by_type = payload["summary"]["by_coref_type_fact"]
    systems = sorted(s)
    print(f"{'coref_type':<26}" + "".join(f"{k:>18}" for k in systems))
    for t in sorted(by_type):
        print(f"{t:<26}" + "".join(
            f"{str(fmt(by_type[t].get(k))):>18}" for k in systems))
    print("\n----- 按指代类型：工单5 改写/实体准确率 -----")
    for t, v in sorted(payload["summary"]["by_coref_type_v5_rewrite"].items()):
        print(f"  {t:<26} n={v['n']} 严格={fmt(v['rewrite'])} "
              f"实体={fmt(v['entity'])} modes={v['rewrite_modes']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单5 多轮三方对比评测")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="跳过工单3 单轮基线（只读子进程）")
    ap.add_argument("--skip-multiturn", action="store_true", help="跳过工单5 多轮一路")
    ap.add_argument("--skip-pure-llm", action="store_true", help="跳过纯 LLM 一路")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="不做预热（首轮会含模型加载耗时，仅用于冷启动诊断）")
    ap.add_argument("--out", default=os.path.join(config.EVAL_DIR,
                                                  "multiturn_compare.json"))
    args = ap.parse_args(argv)

    payload = run(args)
    _print_report(payload)
    print(f"\n落盘 {args.out}")

    # 诚实的退出码：只有**系统级失败**（本系统检索报错、或基线子进程整路没跑起来）
    # 才返回 1。答错题不算「运行失败」——那是被测量的结果，要如实记录而不是让脚本崩。
    v5 = [r for r in payload["rows"] if r["system"] == "v5_multiturn"]
    n_v5_err = sum(1 for r in v5 if r.get("mode") == "error")
    base_rows = [r for r in payload["rows"] if r["system"] == "v3_single_turn"]
    if base_rows:
        n_base_err = sum(1 for r in base_rows if r.get("error"))
        if n_base_err == len(base_rows):
            print(f"⚠ 工单3 基线整路失败（{n_base_err}/{len(base_rows)} 轮）："
                  "请检查子进程输入/环境——整路失败是「没跑起来」，不是基线的水平")
            return 1
        if n_base_err:
            print(f"⚠ 工单3 基线部分失败 {n_base_err}/{len(base_rows)} 轮，已在 rows 中标注")
    llm_rows = [r for r in payload["rows"] if r["system"] == "pure_llm"]
    if llm_rows and all(r.get("error") for r in llm_rows):
        print(f"⚠ 纯 LLM 一路整路失败（{len(llm_rows)} 轮）：Ollama 不可用？"
              "该路成绩无效，不能当成「纯 LLM 很差」")
        return 1
    return 1 if n_v5_err else 0


if __name__ == "__main__":
    # 必须在 main 之前包好大栈线程：main 会加载 bge-m3 / reranker（A1）
    sys.exit(bootstrap.run_with_large_stack(main))
