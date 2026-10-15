# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：优化前后对比报告生成（硬性要求2、6）

输入：`w9.run_eval` 落盘的逐题记录（08 基线 / 09 优化各一份）
输出：
  · `docs/04-评估报告.md` —— RAGAS 指标 + 辅助指标 + 响应时间，逐题明细
  · `data/eval/compare_summary.json` —— 机器可读的对比结果

设计原则：**差值不够，要能看到逐题**。均值会掩盖「某一题暴跌」这类问题，
所以报告里两个系统的逐题指标并列，任何一项退化都能被定位到具体题目。
"""
from __future__ import annotations

import io
import json
import os

from w9 import config9
from w9 import metrics_aux

DOCS = config9.DOCS9


def _load(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def aux_metrics_for(records: list[dict]) -> list[dict]:
    """对每条记录算辅助检索指标（用其 contexts 作为「检索命中」）。"""
    rows: list[dict] = []
    for r in records:
        if r.get("error"):
            continue
        hits = [{"text": c, "source": "", "page_idx": -1}
                for c in (r.get("contexts") or [])]
        m = metrics_aux.evaluate_hits(hits, r.get("reference") or "",
                                      facts=r.get("key_facts") or None,
                                      page_hints=r.get("page_hints") or None,
                                      k=config9.FINAL9_TOP_K)
        rows.append({"id": r.get("id"), **m})
    return rows


def build(set_name: str = "main") -> dict:
    runs = {}
    for sysname in ("wo08", "wo09"):
        runs[sysname] = _load(os.path.join(config9.EVAL9_DIR,
                                           f"run_{set_name}_{sysname}.json"))
    ragas = {}
    for sysname in ("wo08", "wo09"):
        p = os.path.join(config9.EVAL9_DIR, f"ragas_{set_name}_{sysname}.json")
        ragas[sysname] = _load_obj(p)

    out: dict = {"set": set_name, "systems": {}}
    for sysname, recs in runs.items():
        if not recs:
            continue
        ok = [r for r in recs if not r.get("error")]
        times = [r["elapsed_s"] for r in ok]
        aux = aux_metrics_for(ok)
        out["systems"][sysname] = {
            "n": len(recs), "n_ok": len(ok),
            "time_mean_s": round(sum(times) / len(times), 3) if times else None,
            "time_max_s": round(max(times), 3) if times else None,
            "n_over_budget": sum(1 for t in times if t > config9.RESPONSE_BUDGET_S),
            "ragas": (ragas.get(sysname) or {}).get("scores", {}),
            "ragas_nan": (ragas.get(sysname) or {}).get("nan_counts", {}),
            "aux": metrics_aux.aggregate(aux),
            "aux_per_question": aux,
            "per_question_time": {r["id"]: r["elapsed_s"] for r in ok},
        }
    return out


def _load_obj(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------
def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _delta(before, after) -> str:
    if before is None or after is None:
        return "—"
    d = after - before
    return f"{d:+.4f}"


def render_markdown(cmp: dict) -> str:
    s08 = cmp["systems"].get("wo08", {})
    s09 = cmp["systems"].get("wo09", {})
    L: list[str] = []
    A = L.append
    A("# 优化前后对比评估报告")
    A("")
    A("> **工单编号**：人工智能NLP-RAG-Graph RAG优化任务")
    A(f"> 评测集：**sample_questions.pdf 的 {cmp.get('n', 4)} 题**（主集，工单强制比对）")
    A("> 评估框架：**RAGAS 0.2.15**（裁判 LLM：本机 Ollama `qwen2.5:3b`）")
    A("")
    # ---- 口径一致性自检（实测踩坑后加的护栏）--------------------------------
    # 曾经发生过：08 的数字来自 `--ragas full` 那次运行（判分失败 3/4 → 被推到 1.0），
    # 09 的数字来自另一次 `--ragas core` 运行，两者**测量配置不同**，
    # 报告却把它们并排显示成「1.0000 vs 0.6529，退化 -0.3471」——
    # 一个纯粹由口径错配制造出来的假结论。
    # 因此这里强制检查两边是否跑在同一套指标上，不一致就**在最顶部报警**。
    m08 = set((s08.get("ragas") or {}).keys())
    m09 = set((s09.get("ragas") or {}).keys())
    if m08 and m09 and m08 != m09:
        A("## ⛔ 口径不一致 —— 本报告的对比无效")
        A("")
        A(f"- 08 基线跑过的指标：`{sorted(m08)}`")
        A(f"- 09 优化跑过的指标：`{sorted(m09)}`")
        A("")
        A("两边**不是同一套测量配置**，数值不可并排比较。"
          "请用同一条命令重跑两边后再次生成本报告：")
        A("")
        A("```bash")
        A("python -m w9.run_eval --set main --systems wo08,wo09 --ragas core")
        A("python -m w9.compare")
        A("```")
        A("")
    A("## 〇、测量可靠性警告（先读这一节）")
    A("")
    A("**本报告的 RAGAS 数值不稳定，单次运行的数字不可过度解读。**")
    A("")
    A("裁判是本机 `qwen2.5:3b`（工单零下载约束下唯一可用的 LLM）。实测同一份")
    A("08 基线、同一批题、同一个裁判，两次运行的 `context_precision` 分别是：")
    A("")
    A("| 运行 | 08 基线 context_precision | 判分失败数 |")
    A("| --- | --- | --- |")
    A("| 第 1 次 | **0.4672** | 0 / 4 |")
    A("| 第 2 次 | **1.0000** | 3 / 4 |")
    A("")
    A("判分失败的样本被 RAGAS 记为 `NaN` 并**从均值中剔除** —— 于是「失败 3 条、")
    A("剩 1 条恰好判满分」就会把这一项推高到 1.0。也就是说：")
    A("")
    A("- 失败**不是随机噪声**，而是**系统性地丢掉样本**，剩下的样本不代表整体；")
    A("- 4 题的样本量下，单次运行的均值**置信度很低**。")
    A("")
    A("**因此**：本报告的两系统对比只能作为**方向性参考**，不作为精确结论。")
    A("若要拿到可信数字，需要①换用更强裁判模型，或②人工标注 + 增大样本量 + 多次")
    A("重复取分布。这两条在当前「零下载、2 人日」的约束下均未做。")
    A("")
    A("09 侧两次运行分别为 0.6810 / 0.6717（波动 0.009），比 08 侧稳定，")
    A("原因是 09 的判分失败数更少（0 与 1），这一点本身是可靠的。")
    A("")
    A("## 一、核心指标（RAGAS）")
    A("")
    A("| 指标 | 工单要求 | 08 基线（优化前） | 09 优化后 | 变化 | 是否达标 |")
    A("| --- | --- | --- | --- | --- | --- |")
    req = {"context_precision": 0.8, "context_recall": 0.9}
    for key, label in (("context_precision", "context precision（上下文精度）"),
                       ("context_recall", "context recall（上下文召回）")):
        b = (s08.get("ragas") or {}).get(key)
        a = (s09.get("ragas") or {}).get(key)
        need = req[key]
        ok = "✅ 达标" if (a is not None and a >= need) else \
             ("❌ 未达标" if a is not None else "—")
        A(f"| {label} | ≥ {need} | {_fmt(b)} | {_fmt(a)} | "
          f"{_delta(b, a)} | {ok} |")
    A("")
    nan08 = s08.get("ragas_nan") or {}
    nan09 = s09.get("ragas_nan") or {}
    if any(nan08.values()) or any(nan09.values()):
        A(f"> ⚠️ 判分失败（记为 NaN 未计入均值）：08 基线 {nan08}；09 优化 {nan09}")
        A("")

    A("## 二、辅助检索指标（词面重合代理，**非** RAGAS 语义判定的替代）")
    A("")
    A("| 指标 | 08 基线 | 09 优化后 | 变化 |")
    A("| --- | --- | --- | --- |")
    keys = ("precision@k", "recall@k", "hit_rate@k", "mrr", "ndcg@k",
            "truth_coverage")
    labels = {"precision@k": "Precision@k", "recall@k": "Recall@k（论断覆盖）",
              "hit_rate@k": "Hit Rate@k", "mrr": "MRR", "ndcg@k": "NDCG@k",
              "truth_coverage": "金标准论断覆盖率"}
    for key in keys:
        b = (s08.get("aux") or {}).get(key)
        a = (s09.get("aux") or {}).get(key)
        A(f"| {labels.get(key, key)} | {_fmt(b)} | {_fmt(a)} | {_delta(b, a)} |")
    A("")
    A(f"> 相关性判定：块与参考答案共享 ≥{metrics_aux.PHRASE_N} 字连续片段，"
      f"或命中金标准页码；k = {config9.FINAL9_TOP_K}。")
    A("")
    A("> ⚠️ **本组指标衡量的是词面重合，不是「对回答有没有用」，"
      "与 RAGAS 的语义判定排序可能不一致**（实测反例：SQ2 在本组指标下近乎满分，"
      "而 RAGAS 判其 context precision 仅 0.122 —— 召回的块虽与答案有词面重合，"
      "却并未回答问题）。根因是参考答案为**改写摘要**而非原文摘录。"
      "判断检索质量**以第一节的 RAGAS 指标为准**，本组仅作零模型参与的确定性旁证。")
    A("")
    A("> 📌 **Recall@k 与「金标准论断覆盖率」为 0 的原因（非缺陷）**：这两项依赖"
      "逐题的 `key_facts` 金标准论断列表，而 **sample_questions.pdf 只给了参考答案、"
      "没有给论断列表**。故在主集上这两项无输入、恒为 0，是**口径所致**，"
      "不代表检索没命中。它们仅在辅助集（工单08 的 14 题，带 `key_facts`）上有意义。")
    A("")

    A("## 三、响应时间")
    A("")
    A("| 项 | 工单要求 | 08 基线 | 09 优化后 |")
    A("| --- | --- | --- | --- |")
    A(f"| 平均响应时间 | ≤ {config9.RESPONSE_BUDGET_S} 秒 | "
      f"{_fmt(s08.get('time_mean_s'), 2)}s | {_fmt(s09.get('time_mean_s'), 2)}s |")
    A(f"| 最慢单题 | ≤ {config9.RESPONSE_BUDGET_S} 秒 | "
      f"{_fmt(s08.get('time_max_s'), 2)}s | {_fmt(s09.get('time_max_s'), 2)}s |")
    A(f"| 超 {config9.RESPONSE_BUDGET_S}s 题数 | 0 | "
      f"{s08.get('n_over_budget', '—')} | {s09.get('n_over_budget', '—')} |")
    A("")

    A("## 四、逐题响应时间")
    A("")
    A("| 题号 | 08 基线 | 09 优化后 |")
    A("| --- | --- | --- |")
    t08 = s08.get("per_question_time") or {}
    t09 = s09.get("per_question_time") or {}
    for qid in sorted(set(t08) | set(t09)):
        A(f"| {qid} | {_fmt(t08.get(qid), 2)}s | {_fmt(t09.get(qid), 2)}s |")
    A("")
    return "\n".join(L)


def main() -> int:
    cmp = build("main")
    # 题目数
    recs = _load(os.path.join(config9.EVAL9_DIR, "run_main_wo09.json")) or \
        _load(os.path.join(config9.EVAL9_DIR, "run_main_wo08.json"))
    cmp["n"] = len(recs)

    p = os.path.join(config9.EVAL9_DIR, "compare_summary.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(cmp, f, ensure_ascii=False, indent=2)

    md = render_markdown(cmp)
    mp = os.path.join(DOCS, "04-评估报告.md")
    os.makedirs(DOCS, exist_ok=True)
    with io.open(mp, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"已写出 {p}\n已写出 {mp}")
    print("\n" + md[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
