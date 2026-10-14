# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：Graph RAG（08）与普通 RAG（07）的对比分析（硬性要求4）

对比口径说明
------------
两份结果的问题集**同源**（Q1~Q10 完全一致，含同一套 key_facts 金标准），
因此可逐题严格对齐；Q11~Q14 为 08 扩展题，07 未覆盖，单列标注。

对比维度
--------
· 关键事实覆盖率（fact_recall）—— 答案中命中了多少条金标准关键事实；
· 响应时间；
· 检索模式与图谱事实条数（07 无此概念，用于解释差异来源）。

金标准来源：本工程 data/eval/ground_truth.json（Q1~Q10 复制自 07）。
07 的结果：只读引用工单07 工程产物，不修改。

格式适配说明（相对计划的一处修正）
----------------------------------
工单07 的 `rag_results_hybrid.json` 实际是 `{"strategy":…, "rows":[…]}` 包装结构，
且耗时字段为 `answer_wall_ms`（毫秒）。加载器与耗时解析已按真实产物适配，
同时保留对裸 list 的兼容（测试桩口径）。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

WO07_ROOT = r"D:\xinzg6\zy-gq\工单7"
WO07_HYBRID = os.path.join(WO07_ROOT, "data", "eval", "rag_results_hybrid.json")
WO08_RESULTS = os.path.join("data", "eval", "graph_qa_results.json")
GROUND_TRUTH = os.path.join("data", "eval", "ground_truth.json")

_NUM_RE = re.compile(r"[,\s，]")


def _norm(text: str) -> str:
    """归一：去千分位/空白/全角逗号，便于数值比对。"""
    return _NUM_RE.sub("", str(text or ""))


def fact_recall(answer: str, key_facts: list[str]) -> float:
    """关键事实覆盖率：key_facts 中被答案命中的比例（0~1）。"""
    if not key_facts:
        return 0.0
    a = _norm(answer)
    hit = sum(1 for f in key_facts if _norm(f) and _norm(f) in a)
    return hit / len(key_facts)


def _extract_rows(data) -> list[dict]:
    """兼容三种产物形态：裸 list / {"rows": [...]} / {id: item}。"""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            return [x for x in data["rows"] if isinstance(x, dict)]
        # {题号: 记录} 形式：值里带 id 即可
        return [v for v in data.values() if isinstance(v, dict)]
    return []


def load_wo07_results(path: str | None = None) -> dict[int, dict]:
    """读工单07 的逐题结果，按题号索引。文件缺失返回空 dict（不报错）。"""
    path = path or WO07_HYBRID
    if not os.path.isfile(path):
        return {}
    try:
        with io.open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    out: dict[int, dict] = {}
    for item in _extract_rows(data):
        if "id" in item:
            try:
                out[int(item["id"])] = item
            except (TypeError, ValueError):
                continue
    return out


def summarize(rows: list[dict]) -> dict:
    """汇总统计。recall_07 为 None 表示 07 未覆盖该题。"""
    n = len(rows)
    if not n:
        return {"n": 0}

    def _avg(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    paired = [r for r in rows if r.get("recall_07") is not None]
    improved = sum(1 for r in paired if r["recall_08"] > r["recall_07"])
    regressed = sum(1 for r in paired if r["recall_08"] < r["recall_07"])
    tied = len(paired) - improved - regressed

    return {
        "n": n,
        "avg_recall_08": _avg("recall_08"),
        "avg_recall_07": _avg("recall_07"),
        "avg_seconds_08": _avg("seconds_08"),
        "avg_seconds_07": _avg("seconds_07"),
        "n_paired": len(paired),
        "n_improved": improved,
        "n_regressed": regressed,
        "n_tied": tied,
        "n_wo07_uncovered": sum(1 for r in rows if r.get("recall_07") is None),
    }


def _wo07_answer(item: dict | None) -> str:
    if not item:
        return ""
    for k in ("answer", "final_answer", "response", "generated_answer"):
        if item.get(k):
            return str(item[k])
    return ""


def _sum_ms(d: dict) -> float:
    """把 *_ms 计时项求和（毫秒），跳过 *_total_ms 防止重复计入。"""
    ms = [v for k, v in d.items()
          if k.endswith("_ms") and not k.endswith("_total_ms")
          and isinstance(v, (int, float))]
    return float(sum(ms))


def _wo07_seconds(item: dict | None) -> float | None:
    """07 结果耗时（秒）。真实产物用 answer_wall_ms（毫秒）。"""
    if not item:
        return None
    for k in ("seconds", "elapsed", "latency", "total_seconds", "wall_seconds"):
        v = item.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    for k in ("answer_wall_ms", "wall_ms", "elapsed_ms", "total_ms"):
        v = item.get(k)
        if isinstance(v, (int, float)):
            return round(float(v) / 1000.0, 3)
    for key in ("answer_timings", "timings"):
        t = item.get(key)
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("total"), (int, float)):
            return float(t["total"])
        if isinstance(t.get("total_ms"), (int, float)):
            return round(float(t["total_ms"]) / 1000.0, 3)
        ms = _sum_ms(t)
        if ms:
            return round(ms / 1000.0, 3)
    return None


def compare(wo08_path: str | None = None, wo07_path: str | None = None,
            gt_path: str | None = None) -> dict:
    """逐题对比，返回 {"rows": [...], "summary": {...}}。"""
    with io.open(gt_path or GROUND_TRUTH, encoding="utf-8") as f:
        gt = json.load(f)

    wo08_path = wo08_path or WO08_RESULTS
    if not os.path.isfile(wo08_path):
        raise FileNotFoundError(
            f"缺少本工单结果 {wo08_path}\n请先运行 python graph_qa_cli.py --all")

    with io.open(wo08_path, encoding="utf-8") as f:
        res08 = {int(r["id"]): r for r in json.load(f) if "id" in r}

    res07 = load_wo07_results(wo07_path)

    rows: list[dict] = []
    for item in sorted(gt, key=lambda x: x["id"]):
        qid = int(item["id"])
        r8 = res08.get(qid, {})
        r7 = res07.get(qid)

        ans8 = str(r8.get("answer") or "")
        ans7 = _wo07_answer(r7) if r7 else ""
        facts = item.get("key_facts") or []

        rows.append({
            "id": qid, "qtype": item.get("qtype"),
            "origin": item.get("origin", "wo08"),
            "question": item["question"],
            "mode_08": r8.get("mode"),
            "graph_facts_08": len(r8.get("graph_facts") or []),
            "recall_08": fact_recall(ans8, facts) if ans8 else 0.0,
            "recall_07": fact_recall(ans7, facts) if r7 else None,
            "seconds_08": (r8.get("timings") or {}).get("total"),
            "seconds_07": _wo07_seconds(r7),
            "degraded_08": bool(r8.get("degraded")),
            "degraded_reason_08": r8.get("degraded_reason", ""),
            "answer_08": ans8[:600],
            "answer_07": ans7[:600],
        })

    return {"rows": rows, "summary": summarize(rows),
            "wo07_source": wo07_path or WO07_HYBRID}


def to_markdown(cmp: dict) -> str:
    """生成 docs/05 逐题检索结果 + docs/06 对比分析的正文。"""
    s = cmp["summary"]
    out = [
        "# eval_question 检索结果（工单08 · 硬性要求2/4）", "",
        "> 本文件由 `python src/compare_with_wo07.py` 自动生成。",
        "> 问题集见根目录 `eval_question.md`（Q1~Q10 来源工单07，Q11~Q14 为 08 扩展）。",
        "", "## 一、总览", "",
        f"- 题目数：**{s['n']}**",
        f"- 关键事实覆盖率均值（08 Graph RAG）：**{s['avg_recall_08']:.3f}**",
        f"- 关键事实覆盖率均值（07 普通 RAG，仅可比题）：**{s['avg_recall_07']:.3f}**",
        f"- 平均响应时间（08）：**{s['avg_seconds_08']:.2f} s**",
        f"- 可比题 {s.get('n_paired', 0)} 道：08 更优 "
        f"**{s.get('n_improved', 0)}** 道 / 持平 {s.get('n_tied', 0)} 道 / "
        f"落后 {s.get('n_regressed', 0)} 道",
        f"- 07 未覆盖题（Q11~Q14）：**{s.get('n_wo07_uncovered', 0)}** 道", "",
        "## 二、逐题结果", "",
        "| 题 | 类型 | 来源 | 08模式 | 图谱事实 | 覆盖率08 | 覆盖率07 | "
        "耗时08(s) |", "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in cmp["rows"]:
        r7 = f"{r['recall_07']:.2f}" if r["recall_07"] is not None else "07未覆盖"
        sec = f"{r['seconds_08']:.2f}" if isinstance(r["seconds_08"], (int, float)) else "—"
        out.append(f"| Q{r['id']:02d} | {r['qtype']} | {r['origin']} | "
                   f"{r['mode_08']} | {r['graph_facts_08']} | "
                   f"**{r['recall_08']:.2f}** | {r7} | {sec} |")

    out += ["", "## 三、题目与答案明细", ""]
    for r in cmp["rows"]:
        out += [f"### Q{r['id']:02d}　{r['question']}", "",
                f"- 类型：{r['qtype']}　来源：{r['origin']}　"
                f"图形检索模式：`{r['mode_08']}`　图谱事实：{r['graph_facts_08']} 条",
                "",
                "**08 Graph RAG 答案**", "", f"> {r['answer_08'] or '（空）'}", ""]
        if r["recall_07"] is not None:
            out += ["**07 普通 RAG 答案**", "",
                    f"> {r['answer_07'] or '（空）'}", ""]
        if r.get("degraded_08"):
            out += [f"⚠️ 降级原因：{r['degraded_reason_08']}", ""]

    return "\n".join(out)


def to_compare_markdown(cmp: dict) -> str:
    """生成 docs/06 对比分析（含优势/不足的定性归因）。"""
    s = cmp["summary"]
    rows = cmp["rows"]
    improved = [r for r in rows if r["recall_07"] is not None
                and r["recall_08"] > r["recall_07"]]
    regressed = [r for r in rows if r["recall_07"] is not None
                 and r["recall_08"] < r["recall_07"]]
    extended = [r for r in rows if r["recall_07"] is None]

    out = [
        "# 与工单07 的对比分析：Graph RAG vs 普通 RAG（硬性要求4）", "",
        "> 本文件由 `python src/compare_with_wo07.py` 自动生成。", "",
        "## 一、对比方法", "",
        "- **同源问题集**：Q1~Q10 完整复用工单07 的测试集与金标准（`key_facts`），"
        "因此可逐题严格对齐；",
        "- **同一底层检索**：08 复用 07 的向量/全文检索与 Reranker 精排"
        "（检索核心零改动），**图谱是唯一增量** —— 差异可归因到 Graph RAG 本身；",
        "- **同一指标口径**：关键事实覆盖率 = 答案命中的 `key_facts` 数 / 总 `key_facts` 数。", "",
        "## 二、量化结果", "",
        f"| 指标 | 工单08 Graph RAG | 工单07 普通 RAG |",
        f"| --- | --- | --- |",
        f"| 关键事实覆盖率（均值） | **{s['avg_recall_08']:.3f}** | "
        f"{s['avg_recall_07']:.3f} |",
        f"| 平均响应时间 | {s['avg_seconds_08']:.2f} s | "
        f"{s['avg_seconds_07']:.2f} s |",
        f"| 可比题数 | {s.get('n_paired', 0)} | {s.get('n_paired', 0)} |", "",
        f"**逐题胜负**：08 更优 {s.get('n_improved', 0)} 道 / "
        f"持平 {s.get('n_tied', 0)} 道 / 落后 {s.get('n_regressed', 0)} 道。", "",
        "## 三、Graph RAG 的优势（逐题证据）", "",
    ]
    if improved:
        for r in improved:
            note = ("　⚠️ 本题图谱通道本次为空，提升来自文本通道，"
                    "不能算作 Graph RAG 的功劳" if r["graph_facts_08"] == 0 else "")
            out.append(f"- **Q{r['id']:02d}（{r['qtype']}）**："
                       f"{r['recall_07']:.2f} → **{r['recall_08']:.2f}**；"
                       f"检索模式 `{r['mode_08']}`，图谱事实 {r['graph_facts_08']} 条。"
                       f"题目：{r['question'][:60]}…{note}")
    else:
        out.append("- （本次无覆盖率提升的可比题）")

    out += ["", "### 结构性优势（不依赖单题）", "",
            "1. **多跳/跨文档比较**：07 需向量检索分别召回两份文档，再由 LLM 比较，"
            "任一文档漏召回即失败；08 直接遍历 "
            "`Company -HAS_METRIC-> Metric` 边取数，**结果确定**。",
            "2. **组织/股权结构**：07 对组织结构图只能靠图像描述文本，层级关系易丢失；"
            "08 由第 1 层规则从「X 下设 A、B、C」句式确定性抽取 `BELONGS_TO` 边"
            "（本次 63 条，覆盖 9 家公司中的 7 家：董事会/监事会/专门委员会/行业组），"
            "再由 org_tree 模式还原层级。Q13（中信证券附录组织结构）即由图谱事实作答。"
            "**但覆盖是有边界的**：Q10（邮储银行总行板块）在语料中没有对应句式，"
            "图谱通道为空，该题仍靠文本通道 —— 详见下节缺口清单。",
            "3. **可回溯性**：08 的每条图谱事实自带 "
            "`source_doc + page + evidence`，答案引用可精确定位到页；"
            "07 的证据来自文本块相似度，页码需另行推断。",
            "4. **研报生成**：08 的结构化取数表（公司×指标×期间）"
            "为长文生成提供确定性骨架。", "",
            "## 四、Graph RAG 的不足（逐题证据）", ""]
    if regressed:
        for r in regressed:
            out.append(f"- **Q{r['id']:02d}（{r['qtype']}）**："
                       f"{r['recall_07']:.2f} → {r['recall_08']:.2f}；"
                       f"模式 `{r['mode_08']}`，图谱事实 {r['graph_facts_08']} 条。")
    else:
        out.append("- （本次无可比题覆盖率下降）")

    out += ["", "### 结构性不足", "",
            "1. **图谱覆盖决定上限**：叙述性内容（战略、文化、定性表述）"
            "难进入图谱，这类问题图谱通道无贡献，仍依赖文本通道；",
            "2. **抽取误差传播**：词典未覆盖的别名、LLM 抽取的漏抽/误抽，"
            "会直接表现为图谱事实缺失或错误；",
            "3. **多一跳开销**：实体链接 + 子图检索带来额外耗时，"
            "对纯事实型单点问题的收益有限。", "",
            "## 五、本次实测的图谱覆盖缺口（如实记录）", ""]
    # 模式声称走图谱、却一条事实都没取到的题：不粉饰，逐题列出
    gaps = [r for r in rows if r["graph_facts_08"] == 0]
    if gaps:
        for r in gaps:
            why = ("该题判定为纯文本问题，未触发图谱模式"
                   if r["mode_08"] == "none" else
                   "图谱通道无贡献（该题所需结构未入图），答案完全来自文本通道")
            out.append(f"- **Q{r['id']:02d}（{r['qtype']}）**：模式 `{r['mode_08']}`，"
                       f"图谱事实 0 条 —— {why}。")
    else:
        out.append("- （本次全部题目都有图谱事实支撑）")

    out += ["", "## 六、工单08 扩展题的表现（07 未覆盖）", ""]
    if extended:
        for r in extended:
            out.append(f"- **Q{r['id']:02d}（{r['qtype']}）**：覆盖率 "
                       f"{r['recall_08']:.2f}，模式 `{r['mode_08']}`，"
                       f"图谱事实 {r['graph_facts_08']} 条。题目：{r['question'][:60]}…")
    else:
        out.append("- （无扩展题）")

    out += ["", "## 七、结论", "",
            "Graph RAG 的增量集中在**关系型、多跳型、结构型**问题上："
            "这类问题普通 RAG 靠相似度召回文本块、把比较交给 LLM，"
            "而 Graph RAG 用图查询给出确定答案。"
            "对单点事实型问题，两者表现接近 —— 图谱的价值不在替换，而在**补上关系这条腿**。", ""]
    return "\n".join(out)


def write_reports(cmp: dict | None = None) -> dict:
    cmp = cmp or compare()
    p5 = os.path.join("docs", "05-eval_question检索结果.md")
    p6 = os.path.join("docs", "06-与07工单对比分析.md")
    pj = os.path.join("data", "eval", "compare_wo07.json")
    os.makedirs("docs", exist_ok=True)
    os.makedirs(os.path.dirname(pj), exist_ok=True)
    with io.open(p5, "w", encoding="utf-8") as f:
        f.write(to_markdown(cmp))
    with io.open(p6, "w", encoding="utf-8") as f:
        f.write(to_compare_markdown(cmp))
    with io.open(pj, "w", encoding="utf-8") as f:
        json.dump(cmp, f, ensure_ascii=False, indent=2)
    return {"docs_05": p5, "docs_06": p6, "json": pj}


def main() -> int:
    cmp = compare()
    paths = write_reports(cmp)
    s = cmp["summary"]
    print(f"题目 {s['n']} 道；08 覆盖率 {s['avg_recall_08']:.3f} "
          f"（07 可比题 {s['avg_recall_07']:.3f}）")
    print(f"08 更优 {s['n_improved']} / 持平 {s['n_tied']} / "
          f"落后 {s['n_regressed']}；07 未覆盖 {s['n_wo07_uncovered']}")
    for k, v in paths.items():
        print(f"  已写入 {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
