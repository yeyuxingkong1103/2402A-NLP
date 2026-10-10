# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""优化前后对比：baseline_03（代表工单01/02/03）vs full_04（本工单）。"""
from __future__ import annotations

from pathlib import Path

_COMPARE_KEYS = (
    "answer_accuracy", "hit_rate", "mrr",
    "precision@5", "recall@5", "ndcg@5",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
)

_HEADLINE = {
    "answer_accuracy": "答案准确率",
    "hit_rate": "命中率 Hit Rate",
    "mrr": "MRR",
    "precision@5": "Precision@5",
    "recall@5": "Recall@5",
    "ndcg@5": "NDCG@5",
    "latency_mean_ms": "平均响应时间(ms)",
    "latency_p50_ms": "P50 响应时间(ms)",
    "latency_p95_ms": "P95 响应时间(ms)",
}

# 耗时类指标按毫秒作差，不能套百分点公式（_delta 会得 -40000.0pp 这类无意义值）。
_MS_KEYS = ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms")


def _num(value) -> float | None:
    """数值才参与作差；None（英文模式 answer_accuracy 按设计置空）等一律不算。"""
    return value if isinstance(value, (int, float)) else None


def _delta(full, base) -> str:
    """返回百分点差值字符串，如 '+15.0pp'；任一侧无值时返回 '—'。

    Fix 3：英文模式结果 ``answer_accuracy=None``，原实现直接相减会抛裸
    TypeError（整份对比报告写不出来）。
    """
    f, b = _num(full), _num(base)
    if f is None or b is None:
        return "—"
    return f"{(f - b) * 100:+.1f}pp"


def _delta_ms(full, base) -> str:
    """返回毫秒差值字符串，如 '-400.0ms'（负数表示优化后更快）；缺值返回 '—'。"""
    f, b = _num(full), _num(base)
    if f is None or b is None:
        return "—"
    return f"{f - b:+.1f}ms"


def _val(value) -> str:
    return "不适用" if value is None else str(value)


def _cov_note(base_cov, full_cov) -> str:
    """图像题说明列：只写可核算的差值，不写与数字冲突的断言。

    Fix 3：原实现固定写「baseline 无图像能力，覆盖率为 0 属预期」，与实测
    0.1/0.2 的覆盖率直接矛盾。
    """
    b, f = _num(base_cov), _num(full_cov)
    if b is None or f is None:
        return "覆盖率未上报（英文模式不判答案正确性），本次不给结论"
    if f > b:
        return f"覆盖率 {b:.2f} → {f:.2f}（提升 {(f - b) * 100:+.1f}pp）"
    if f == b:
        return f"覆盖率两模式相同（{b:.2f}）：本次测量未见图像题差异"
    return f"覆盖率 {b:.2f} → {f:.2f}（下降 {(f - b) * 100:+.1f}pp）"


def write_comparison(baseline: dict, full: dict, path: Path) -> Path:
    """产出优化前后对比 Markdown。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bm, fm = baseline["metrics"], full["metrics"]

    lines = [
        "# 优化前后对比分析",
        "",
        "## 一、对比口径",
        "",
        "| 维度 | baseline_03（代表工单01/02/03） | full_04（工单04） |",
        "| --- | --- | --- |",
        "| 模态 | 仅文本 | 文本 + 表格 + 图像 |",
        "| 分块 | 固定 512 字 | 按 block_type 语义分块 |",
        "| 检索 | 纯稠密 | 稠密 + 稀疏 + CLIP，RRF 融合 |",
        "| 重排 | 无 | bge-reranker-base |",
        "| 图像 | 不解析 | 图区检测 + 多模态解析 + CLIP 向量 |",
        "",
        # 口径脚注：上表描述的是**设计口径**；baseline 的实测数据是在
        # use_hybrid 还是死开关时产出的（见 retrieve/hybrid.py 的 Fix 7），
        # 不写明会让读者以为基线是纯稠密，从而误读下降/上升幅度。
        "> 注：baseline_03 的实测数据产出时 `use_hybrid` 尚未接线（实际执行的"
        "是稠密+稀疏），故其召回强于纯稠密基线；该差异只会低估本工单的优化"
        "幅度，不会夸大。如需纯稠密基线需重建其索引（约 2.6 小时），已登记为"
        "后续项。",
        "",
        # 口径脚注 2：两侧时延不是同一时点的测量。baseline 评估早于
        # `ollama_url` 的 IPv4 修复（见 config.py），full_04 最终那次在修复后 ——
        # 时延差值不能全部归因于工单04 的流水线改动，不写明会被误读为纯优化增益。
        "> 注 2：两侧时延不是同一时点的测量——baseline_03 的评估早于 "
        "`ollama_url` 的 IPv4 修复（`src/rag04/config.py`，修复前每次嵌入多付约 2 秒），"
        "full_04 为修复后所测；故时延差值不能全部归因于工单04 的流水线改动。"
        "full_04 的 p95 含首题冷启动（模型加载），属进程内首次调用的固定开销。"
        "检索与准确率指标不受该修复影响，对比仍然成立。",
        "",
        "两模式共用同一套代码、同一批 16 题、同一份语料，变量可控。",
        "",
        "## 二、指标对比",
        "",
        "| 指标 | baseline_03 | full_04 | 变化 |",
        "| --- | --- | --- | --- |",
    ]
    for k in _COMPARE_KEYS:
        if k in bm and k in fm:
            change = _delta_ms(fm[k], bm[k]) if k in _MS_KEYS else _delta(fm[k], bm[k])
            # 中文表头后跟上原始指标键：只写中文会让报告无法与评估结果 JSON 对账。
            lines.append(
                f"| {_HEADLINE.get(k, k)}（`{k}`） | {_val(bm[k])} | {_val(fm[k])} "
                f"| {change} |"
            )

    # 图像题专项
    img_ids = {5, 6}
    b5 = {r["qid"]: r for r in baseline["rows"] if r["qid"] in img_ids}
    f5 = {r["qid"]: r for r in full["rows"] if r["qid"] in img_ids}
    lines += [
        "", "## 三、图像题专项（id 5 / id 6）", "",
        "这两题的答案既不在文本也不在表格中，是工单04 的核心增量。", "",
        "| id | 题型 | baseline_03 覆盖率 | full_04 覆盖率 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for qid in sorted(img_ids):
        b = b5.get(qid, {}).get("coverage")
        f = f5.get(qid, {}).get("coverage")
        lines.append(
            f"| {qid} | 图像 | {_val(b)} | {_val(f)} | {_cov_note(b, f)} |"
        )

    # 结论只复述表里可核算的数值；作差前先判有无值（英文模式 answer_accuracy=None）。
    lines += ["", "## 四、结论", ""]
    acc_b, acc_f = _num(bm.get("answer_accuracy")), _num(fm.get("answer_accuracy"))
    lat_b, lat_f = _num(bm.get("latency_mean_ms")), _num(fm.get("latency_mean_ms"))
    if acc_f is None or acc_b is None:
        lines.append("- 答案准确率变化：**不适用**（该模式未上报答案正确性，"
                     "见执行器口径说明），故本报告不给出准确率结论。")
    else:
        lines.append(f"- 答案准确率变化：**{_delta(acc_f, acc_b)}**")
    if lat_f is None or lat_b is None:
        lines.append("- 平均响应时间变化：**不适用**（有一侧未上报耗时）。")
    else:
        lines.append(f"- 平均响应时间变化：**{lat_f - lat_b:+.1f} ms**")
    for qid in sorted(img_ids):
        note = _cov_note(b5.get(qid, {}).get("coverage"),
                         f5.get(qid, {}).get("coverage"))
        lines.append(f"- 图像题 id {qid}：{note}")
    lines += [
        "- 以上仅为本次测量的数值差异；各组件（图像解析 / 混合检索 / 重排）的"
        "增益归因需消融实验，本报告不作因果断言。",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
