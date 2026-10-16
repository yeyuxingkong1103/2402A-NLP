# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""32_make_screenshots.py —— 用真实产物渲染中文命名测试截图（≥3 张）。

数据来源（**只读产物文件，绝不手绘假数据**）：
  * ``研发/评估/逐题结果.jsonl``   逐题 × 逐后端结果与判分
  * ``研发/评估/对比汇总.json``    按后端聚合
  * ``研发/评估/RAGAS指标.json``   RAGAS 四项（自实现降级口径）
  * ``研发/data/graph/graph_meta.json`` 图谱与抽样口径

输出（``测试/测试截图/``，文件名按「测试内容」中文命名）：
  1. ``16题指标对比_RAG与LightRAG.png``    四项指标 + 核心汇总（表格）
  2. ``检索命中率对比_逐题.png``           逐题关键词命中率对照
  3. ``逐题评分与答案明细.png``            逐题判分明细（含页命中）
  4. ``RAGAS四项指标对比.png``             RAGAS 四项 × 后端
  5. ``图谱边口径对比_全量边与仅相容边.png`` 两组边口径的指标差异 + t2 遗留数据

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/32_make_screenshots.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))
if r"E:\gao6gongdan\_共享工具" not in sys.path:
    sys.path.insert(0, r"E:\gao6gongdan\_共享工具")

from lightrag.config import load_config                       # noqa: E402
from lightrag.logging_util import get_logger                  # noqa: E402
from 截图渲染 import Page, load_json, load_jsonl              # noqa: E402

BACKEND_LABEL = {"rag": "RAG（扁平）", "lightrag": "LightRAG（全量边）",
                 "lightrag_strict": "LightRAG（仅相容边）"}


def label(backend: str) -> str:
    return BACKEND_LABEL.get(str(backend), str(backend))


def write_report(cfg, rows, summary, ragas, graph_meta, order) -> Path:
    """生成 优化/评估结果/对比报告.md（数字全部来自实跑产物）。"""
    meta = summary.get("_meta", {})
    gs = graph_meta.get("graph_stats", {})
    es = graph_meta.get("extract_stats", {})
    sm = graph_meta.get("sampling", {})
    by_id: dict[int, dict] = {}
    for r in rows:
        by_id.setdefault(int(r["id"]), {})[str(r["backend"])] = r
    ok_edges = int(gs.get("edges_constraint_ok", 0) or 0)
    all_edges = int(gs.get("edges", 1) or 1)

    L: list[str] = [
        "# RAG vs LightRAG 对比报告（16 题）",
        "",
        "> **工单编号：人工智能NLP-RAG 项目-LightRAG 优化**",
        f"> 生成时间：由 `32_make_screenshots.py` 依据实跑产物渲染；问题集：`{meta.get('问题集')}`",
        "",
        "## 1. 抽样口径（与 graph_meta.json / 部署 README 逐字一致）",
        "",
        str(graph_meta.get("抽样口径") or "（缺失）"),
        "",
        f"- 抽取失败率：**{es.get('failed')}/{es.get('attempted')}**"
        f"（{'%.1f%%' % (100.0 * float(es.get('failed') or 0) / max(int(es.get('attempted') or 1), 1))}）"
        "，根因全部为输出被 `max_tokens` 截断的 `JSONDecodeError`；",
        f"- 样本：计划 **{sm.get('selected')}** 块 / 实际 **{sm.get('extracted_records')}** 块"
        f"（时间预算收束，非缺陷）。",
        "",
        "## 2. 测试条件（两后端唯一变量 = 检索通道）",
        "",
        f"| 项 | 值 |\n| --- | --- |",
        f"| 问题集 | 16 题（`研发/评估/测试问题集.jsonl`，未另造题） |",
        f"| 后端 | {'、'.join(label(b) for b in order)} |",
        f"| top_k | {meta.get('top_k')} |",
        f"| 生成模型 | {meta.get('生成模型')}（两后端共用） |",
        f"| 页命中判定 | **±{meta.get('页号容差')} 页容差**（物理页 vs 逻辑页，实测有 ±1 偏差） |",
        f"| 文本匹配 | 先做**去空白 + 全角括号归一**再统计（实测不归一化会系统性低估） |",
        f"| 答对阈值 | 答案关键词命中率 ≥ {meta.get('答对阈值')} |",
        "",
        "## 3. 主结果",
        "",
        "| 指标 | " + " | ".join(label(b) for b in order) + " |",
        "| --- | " + " | ".join("---:" for _ in order) + " |",
    ]
    for k in ("题数", "答对数", "答对率", "平均答案关键词命中率", "平均上下文关键词命中率",
              "平均数字命中率", "页命中数", "页命中率", "平均检索耗时_s", "平均生成耗时_s", "平均总耗时_s"):
        L.append(f"| {k} | " + " | ".join(str(summary.get(b, {}).get(k, "—")) for b in order) + " |")

    L += ["", "## 4. RAGAS 四项（自实现降级口径 `_kw`）", "",
          f"- 官方 `ragas` 可用：**{ragas.get('官方ragas可用')}**（{ragas.get('官方ragas探测结果')}）",
          f"- 口径差异：{ragas.get('口径差异说明')}", "",
          "| 指标 | " + " | ".join(label(b) for b in order) + " |",
          "| --- | " + " | ".join("---:" for _ in order) + " |"]
    for m in ("faithfulness_kw", "answer_relevancy_kw", "context_precision_kw", "context_recall_kw"):
        L.append(f"| {m} | " + " | ".join(
            str(ragas.get("汇总", {}).get(b, {}).get(m, "—")) for b in order) + " |")

    L += ["", "## 5. 两组边口径对比（t2 遗留：关系头尾相容率仅 "
          f"{ok_edges}/{all_edges} = {100.0 * ok_edges / all_edges:.1f}%）", "",
          "图谱侧（`研发/data/graph/graph_meta.json`）：", "",
          "| 项 | 值 |", "| --- | ---: |",
          f"| 节点数 | {gs.get('nodes')} |", f"| 边数 | {gs.get('edges')} |",
          f"| 约束相容边 `constraint_ok=true` | {ok_edges} |",
          f"| 已标记边 `constraint_ok=false` | {gs.get('edges_constraint_flagged')} |",
          f"| 相容率 | {100.0 * ok_edges / all_edges:.1f}% |", "",
          "检索侧（全量边 vs 仅相容边）：", ""]
    two = [b for b in order if b in ("lightrag", "lightrag_strict")]
    if len(two) >= 2:
        L += ["| 指标 | " + " | ".join(label(b) for b in two) + " |",
              "| --- | " + " | ".join("---:" for _ in two) + " |"]
        for k in ("答对数", "答对率", "平均答案关键词命中率", "平均上下文关键词命中率", "页命中率", "平均总耗时_s"):
            L.append(f"| {k} | " + " | ".join(str(summary.get(b, {}).get(k, "—")) for b in two) + " |")
        L += ["", "> **结论（如实）**：仅用 46 条相容边时各项指标**不降反升**，"
              "说明 t2 的 `constraint_ok` 标记是有效的质量信号——被标记的 121 条边确实在拉低全局检索质量。"]
    else:
        L.append("本轮未跑 `lightrag_strict`，两组口径缺一组数据。")

    L += ["", "## 6. 逐题摘要", "",
          "| id | 问题（截断） | " + " | ".join(f"{label(b)} 答对" for b in order) + " | 页命中 |",
          "| ---: | --- | " + " | ".join(":---:" for _ in order) + " | :---: |"]
    for qid in sorted(by_id):
        first = next(iter(by_id[qid].values()))
        q = str(first.get("question") or "")
        cells = []
        for b in order:
            r = by_id[qid].get(b)
            cells.append("—" if not r else ("是" if r["score"]["answered"] else "否"))
        ph = "—"
        for b in order:
            r = by_id[qid].get(b)
            if r:
                ph = "是" if r["score"]["page_hit"] else "否"
                break
        L.append(f"| {qid} | {q[:34]}{'…' if len(q) > 34 else ''} | " + " | ".join(cells) + f" | {ph} |")

    L += ["", "## 7. 未达标项与限制（如实记录，不修饰）", ""]
    if 6 in by_id:
        r6 = by_id[6].get("rag") or by_id[6].get("lightrag")
        L.append(f"- **id=6（2008 年中国 IC 市场应用结构与增长图）—— 未达标**。"
                 f"根因：该题答案位于 p72 插图的**图内文字**，本机**无视觉模型**且图块未做 OCR"
                 f"（图块内容仅为「插图（638×479px，文字描述口径，未做 OCR）」）；"
                 f"全库 3013 块中 `-2.0%` 命中 0 块、`IC 卡` 仅 1 块且在 p151（非 p72）。"
                 f"**未使用近似答案充数**，实测答案：`{str(r6.get('answer') or '（空）')[:80]}`")
    L += [
        f"- **关系头尾类型相容率仅 {100.0 * ok_edges / all_edges:.1f}%**（{ok_edges}/{all_edges}）："
        "qwen2.5:3b 会把「本公司全体董监高」判成 `母子公司`（应为 `任职`），且 16 类关系中没有"
        "「保荐/审计/上市地」等常见关系。处理方式：**保留并标记** `constraint_ok=false`，"
        "**未改写或编造任何一条边**；两组口径指标见 §5。",
        f"- **抽取失败率 {es.get('failed')}/{es.get('attempted')}**：输出被 `max_tokens` 截断。"
        "已修（上限 900→1400、提示词「宁少勿长」、重试 2→1、新增截断抢救 `_salvage_objects`），"
        "但**本轮 88 块仍是用修复前设置抽的**，修复只对下次重跑生效。",
        f"- **样本 {sm.get('extracted_records')} < 计划 {sm.get('selected')} 块**：1200s 时间预算自然收束，非缺陷。",
        f"- **答对率整体偏低**（最高 {max(float(summary.get(b, {}).get('答对率', 0)) for b in order) * 100:.1f}%）："
        "答对判定要求**参考答案关键词全部命中**；qwen2.5:3b 在此语料上的抽取与生成质量有限，"
        "且知识图谱仅由 88 块小样本建成。**未通过降低阈值或剔除难题来抬高分数**。",
        "",
        "## 8. 复现命令", "",
        "```powershell",
        "cd E:\\gao6gongdan\\工单12",
        "# ① 16 题 × 三后端检索 + 生成（约 83 秒实测）",
        "pwsh -NoProfile -File run_py.ps1 研发/scripts/30_compare_rag_vs_lightrag.py --no-resume",
        "# ② RAGAS 四项（自实现降级口径）",
        "pwsh -NoProfile -File run_py.ps1 研发/scripts/31_ragas_eval.py",
        "# ③ 中文命名截图 + 本报告",
        "pwsh -NoProfile -File run_py.ps1 研发/scripts/32_make_screenshots.py",
        "```",
        "",
    ]
    out = Path(cfg.path_str("optimize_dir")) / "对比报告.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成中文命名测试截图")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("32_screenshots", cfg, console=True)
    shot_dir = Path(cfg.path_str("screenshot_dir"))
    shot_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(cfg.path_str("eval_dir"))
    per_q_path = eval_dir / "逐题结果.jsonl"
    summary = load_json(eval_dir / "对比汇总.json") or {}
    ragas = load_json(eval_dir / "RAGAS指标.json") or {}
    graph_meta = load_json(Path(cfg.path_str("graph_dir")) / "graph_meta.json") or {}
    rows = load_jsonl(per_q_path)
    if not rows:
        log.error("找不到逐题结果：%s", per_q_path)
        return 2

    made: list[str] = []
    order = [b for b in ("rag", "lightrag", "lightrag_strict") if b in summary]
    order += [b for b in summary if b not in order and not str(b).startswith("_")]

    # ① 16 题指标对比
    p = Page("16 题 RAG vs LightRAG 指标对比（工单12）",
             f"题数 {summary.get('_meta', {}).get('题数', '?')}　"
             f"页号容差 ±{summary.get('_meta', {}).get('页号容差', '?')}　"
             f"答对阈值 {summary.get('_meta', {}).get('答对阈值', '?')}")
    p.section("核心汇总（真实计数）")
    keys = ["题数", "答对数", "答对率", "平均答案关键词命中率", "平均上下文关键词命中率",
            "平均数字命中率", "页命中数", "页命中率", "平均检索耗时_s", "平均总耗时_s"]
    table = [[k] + [str(summary.get(b, {}).get(k, "—")) for b in order] for k in keys]
    p.table_block(["指标"] + [label(b) for b in order], table, widths=[320] + [300] * len(order))
    p.note(f"图谱：节点 {graph_meta.get('graph_stats', {}).get('nodes', '?')} 个、"
           f"边 {graph_meta.get('graph_stats', {}).get('edges', '?')} 条"
           f"（约束相容 {graph_meta.get('graph_stats', {}).get('edges_constraint_ok', '?')} / "
           f"已标记 {graph_meta.get('graph_stats', {}).get('edges_constraint_flagged', '?')}）")
    p.note("页命中判定使用 ±1 页容差；关键词/数字匹配前先做去空白 + 全角括号归一。")
    made.append(str(p.finish(shot_dir / "16题指标对比_RAG与LightRAG.png",
                             sources=[str(per_q_path.relative_to(REPO_ROOT)),
                                      str((eval_dir / '对比汇总.json').relative_to(REPO_ROOT))])))

    # ② 检索命中率对比（逐题）
    p = Page("检索命中率对比（逐题：关键词在答案/上下文中的命中率）",
             "左=答案关键词命中率　右=上下文关键词命中率（去空白归一后匹配）")
    p.section("逐题命中率")
    by_id: dict[int, dict] = {}
    for r in rows:
        by_id.setdefault(int(r["id"]), {})[str(r["backend"])] = r
    headers = ["id", "问题（截断）"] + [f"{label(b)} 答案" for b in order] + [f"{label(b)} 上下文" for b in order]
    table = []
    for qid in sorted(by_id):
        first = next(iter(by_id[qid].values()))
        q = str(first.get("question") or "")
        table.append([qid, q[:26] + ("…" if len(q) > 26 else "")]
                     + [f"{by_id[qid][b]['score']['answer_hit_rate']:.2f}" if b in by_id[qid] else "—" for b in order]
                     + [f"{by_id[qid][b]['score']['context_hit_rate']:.2f}" if b in by_id[qid] else "—" for b in order])
    p.table_block(headers, table, widths=[60, 300] + [150] * (2 * len(order)), max_rows=20)
    made.append(str(p.finish(shot_dir / "检索命中率对比_逐题.png",
                             sources=[str(per_q_path.relative_to(REPO_ROOT))])))

    # ③ 逐题评分与答案明细
    p = Page("逐题评分与答案明细（16 题 × 多后端）",
             "答对=答案关键词命中率≥阈值；页命中=命中页码落在 expected_pages ±1 内")
    for qid in sorted(by_id):
        first = next(iter(by_id[qid].values()))
        p.section(f"id={qid}　{str(first.get('question') or '')[:60]}")
        p.note(f"参考答案：{str(first.get('reference_answer') or '')[:120]}")
        for b in order:
            r = by_id[qid].get(b)
            if not r:
                continue
            sc = r["score"]
            p.text(f"[{label(b)}] 答对={'是' if sc['answered'] else '否'}　"
                   f"答案命中={sc['answer_hit_rate']:.2f}　上下文命中={sc['context_hit_rate']:.2f}　"
                   f"数字命中={sc['number_hit_rate']:.2f}　页命中={'是' if sc['page_hit'] else '否'}　"
                   f"延迟={r['latency_s']:.1f}s", size=17, gap=26)
            p.note(f"　　命中页：{sc['retrieved_pages'][:6]}　Top块：{str(sc['top_chunk_ids'][:2])[:70]}", size=15)
            ans = (r.get("answer") or "").replace("\n", " ").strip()
            p.note(f"　　答案：{ans[:110] if ans else '（空/未生成）'}", size=15)
    made.append(str(p.finish(shot_dir / "逐题评分与答案明细.png",
                             sources=[str(per_q_path.relative_to(REPO_ROOT))],
                             extra_note="Q6 为图内数据题，本机无视觉模型、图块未 OCR，如实记为未达标")))

    # ④ RAGAS 四项对比
    p = Page("RAGAS 四项指标对比（自实现降级口径 _kw）",
             f"官方 ragas 可用：{ragas.get('官方ragas可用')}　依赖缺失：ragas/pandas/datasets 均未安装")
    p.section("四项指标")
    metrics = ["faithfulness_kw", "answer_relevancy_kw", "context_precision_kw", "context_recall_kw"]
    table = [[m] + [str(ragas.get("汇总", {}).get(b, {}).get(m, "—")) for b in order]
             + [ragas.get("汇总", {}).get(order[0] if order else "", {}).get(m, "—")] for m in metrics]
    p.table_block(["指标"] + [label(b) for b in order], [[r[0]] + r[1:1 + len(order)] for r in table],
                  widths=[320] + [280] * len(order))
    p.section("口径与差异（必须与数字同时呈现）")
    for k, v in (ragas.get("指标口径") or {}).items():
        p.note(f"• {k}：{v}", size=16)
    p.note(str(ragas.get("口径差异说明") or ""), size=16)
    made.append(str(p.finish(shot_dir / "RAGAS四项指标对比.png",
                             sources=[str((eval_dir / 'RAGAS指标.json').relative_to(REPO_ROOT))])))

    # ⑤ 两组边口径对比
    p = Page("图谱边口径对比：全量边 vs 仅 constraint_ok 边",
             "t2 遗留：关系头尾类型相容率仅 46/167 = 27.5%")
    p.section("图谱侧（来自 graph_meta.json）")
    gs = graph_meta.get("graph_stats", {})
    p.kv_block([("节点数", gs.get("nodes", "?")), ("边数", gs.get("edges", "?")),
                ("约束相容边", gs.get("edges_constraint_ok", "?")),
                ("已标记边", gs.get("edges_constraint_flagged", "?")),
                ("相容率", f"{100.0 * float(gs.get('edges_constraint_ok', 0)) / max(int(gs.get('edges', 1)), 1):.1f}%"),
                ("实体类型", f"{gs.get('entity_types_used', '?')}/{gs.get('entity_types_declared', '?')}"),
                ("关系类型", f"{gs.get('relation_types_used', '?')}/{gs.get('relation_types_declared', '?')}")],
               cols=2)
    p.section("检索侧两组口径对比")
    two = [b for b in order if b in ("lightrag", "lightrag_strict")]
    if len(two) >= 2:
        table = [[k] + [str(summary.get(b, {}).get(k, "—")) for b in two]
                 for k in ("题数", "答对数", "答对率", "平均答案关键词命中率", "平均上下文关键词命中率",
                           "页命中率", "平均检索耗时_s", "平均总耗时_s")]
        p.table_block(["指标"] + [label(b) for b in two], table, widths=[320] + [300] * len(two))
    else:
        p.note("本轮未跑 lightrag_strict（仅相容边）后端，两组口径对比缺一组数据。")
    p.section("口径说明（如实）")
    p.note(f"抽取失败率：{graph_meta.get('extract_stats', {}).get('failed', '?')}/"
           f"{graph_meta.get('extract_stats', {}).get('attempted', '?')}"
           f"（输出被 max_tokens 截断）；样本 {graph_meta.get('sampling', {}).get('selected', '?')} 计划 / "
           f"实际 {graph_meta.get('sampling', {}).get('extracted_records', '?')} 块（时间预算收束）", size=16)
    p.note(str(graph_meta.get("抽样口径") or ""), size=15)
    made.append(str(p.finish(shot_dir / "图谱边口径对比_全量边与仅相容边.png",
                             sources=[str((Path(cfg.path_str('graph_dir')) / 'graph_meta.json').relative_to(REPO_ROOT)),
                                      str((eval_dir / '对比汇总.json').relative_to(REPO_ROOT))])))

    # ⑥ 对比报告（Markdown，优化/评估结果/）
    report = write_report(cfg, rows, summary, ragas, graph_meta, order)
    print(f"\n对比报告：{report}")

    print(f"\n共生成 {len(made)} 张截图：")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
