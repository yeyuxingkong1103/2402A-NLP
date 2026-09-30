# -*- coding: utf-8 -*-
"""评测报告：指标总表 + 每题明细 + 结论（低分指标点名）。"""
from datetime import datetime
# 解析：时间模块（报告时间戳）


# 生成 RAGAS 评测报告（指标总览 + 每题明细 + 低分结论）
def format_report(
    # 解析：生成评测报告
    name: str,
    # 解析：报告名
    role_name: str,
    # 解析：角色名
    scores: dict[str, float],
    # 解析：指标总分
    rows: list[dict],
    # 解析：每题明细行
    model: str = "",
    # 解析：大模型名（可选）
    low_threshold: float = 0.7,
    # 解析：低分阈值（低于则点名）
) -> str:
    lines = [f"# RAGAS {name}", ""]
    # 解析：标题
    lines.append(f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # 解析：时间戳
    lines.append(f"- 评测角色：{role_name}")
    # 解析：角色
    if model:
        # 解析：有模型名
        lines.append(f"- 生成/评测大模型：{model}")
        # 解析：模型
    lines.append(f"- 题目数：{len(rows)}")
    # 解析：题目数

    if scores:
        # 解析：有分数
        lines += ["", "## 指标总览", "", "| 指标 | 分数 | 说明 |", "|---|---|---|"]
        # 解析：总览表头
        metric_notes = {
            # 解析：指标中文说明
            "faithfulness": "回答是否忠于检索上下文（幻觉检测）",
            # 解析：忠实度
            "answer_relevancy": "回答与问题的相关程度",
            # 解析：相关度
            "context_precision": "检索到的上下文与问题的相关程度（精确率）",
            # 解析：检索精确率
            "context_recall": "标准上下文被检索到的比例（召回率）",
            # 解析：检索召回率
            "answer_correctness": "回答相对参考答案的正确性",
            # 解析：回答正确性
        }
        for metric, score in scores.items():
            # 解析：逐指标
            lines.append(f"| {metric} | {score:.3f} | {metric_notes.get(metric, '')} |")
            # 解析：一行一个指标

        lines += ["", "## 结论", ""]
        # 解析：结论段
        low = [(m, s) for m, s in scores.items() if s < low_threshold]
        # 解析：低于阈值的指标
        if low:
            # 解析：有低分
            lines.append(f"以下指标低于 {low_threshold:.2f}，是优先优化方向：")
            # 解析：点名
            for metric, score in low:
                # 解析：逐低分指标
                lines.append(f"- **{metric} = {score:.3f}**：{metric_notes.get(metric, '')}")
                # 解析：列出分数与说明
        else:
            # 解析：全达标
            lines.append(f"所有指标均不低于 {low_threshold:.2f}。")
            # 解析：结论

    if rows:
        # 解析：有明细
        lines += ["", "## 每题明细", ""]
        # 解析：明细段
        headers = ["#", "问题"] + sorted(
            # 解析：表头（序号+问题+各指标排序）
            {k for row in rows for k in row if k != "question"}
            # 解析：收集全部指标列名
        )
        lines.append("| " + " | ".join(headers) + " |")
        # 解析：表头行
        lines.append("|" + "---|" * len(headers))
        # 解析：分隔行
        for i, row in enumerate(rows, 1):
            # 解析：逐题
            cells = [str(i), row["question"]]
            # 解析：序号与问题
            cells += [
                # 解析：逐指标单元格
                format_cell(row.get(metric)) for metric in headers[2:]
                # 解析：格式化分数
            ]
            lines.append("| " + " | ".join(cells) + " |")
            # 解析：一行一题
        lines.append("")
        # 解析：空行
        lines.append("> 注：`—` 表示该题的该项判分未成功（judge 调用失败返回 nan），总分按有效样本聚合。")
        # 解析：脚注说明

    return "\n".join(lines) + "\n"
    # 解析：返回 markdown 报告


def format_cell(value) -> str:
    """分数单元格：float 保留三位；nan/缺失显示 —。"""
    if isinstance(value, float):
        # 解析：浮点数
        if value != value:  # NaN
            # 解析：NaN（自比较不等）
            return "—"
            # 解析：显示破折号
        return f"{value:.3f}"
        # 解析：保留三位
    if value is None:
        # 解析：缺失
        return "—"
        # 解析：破折号
    return str(value)
    # 解析：其他转字符串


def _fmt_delta(old: float, new: float) -> str:
    # 解析：格式化两轮差值
    if old != old or new != new:  # NaN 参与对比时不做判断
        # 解析：任一方 NaN
        return "—"
        # 解析：破折号
    delta = new - old
    # 解析：差值
    return f"{delta:+.3f}"
    # 解析：带符号三位小数（如 +0.213）


# 生成两轮评测对比报告（总分 delta + 每题 delta + 提升/回归）
def build_comparison(
    # 解析：生成对比报告
    base_name: str,
    # 解析：基线名称
    cur_name: str,
    # 解析：本轮名称
    base_scores: dict[str, float],
    # 解析：基线总分
    base_rows: list[dict],
    # 解析：基线明细
    cur_scores: dict[str, float],
    # 解析：本轮总分
    cur_rows: list[dict],
    # 解析：本轮明细
) -> str:
    lines = [f"# RAGAS 评测对比：{base_name} → {cur_name}", ""]
    # 解析：标题

    metrics = [m for m in cur_scores if m in base_scores]
    # 解析：两轮共有的指标
    lines += ["## 指标总览对比", "", "| 指标 | 基线 | 本轮 | Δ |", "|---|---|---|---|"]
    # 解析：对比表头
    for m in metrics:
        # 解析：逐指标
        lines.append(
            # 解析：一行对比
            f"| {m} | {base_scores[m]:.3f} | {cur_scores[m]:.3f} "
            # 解析：基线与本轮分数
            f"| {_fmt_delta(base_scores[m], cur_scores[m])} |"
            # 解析：差值
        )

    improved = [m for m in metrics if _fmt_delta(base_scores[m], cur_scores[m]) not in ("—", "+0.000") and cur_scores[m] > base_scores[m]]
    # 解析：提升指标（差值有效且为正）
    regressed = [m for m in metrics if _fmt_delta(base_scores[m], cur_scores[m]) not in ("—", "+0.000") and cur_scores[m] < base_scores[m]]
    # 解析：回归指标（差值为负）
    lines += ["", "## 结论", ""]
    # 解析：结论段
    if improved:
        # 解析：有提升
        lines.append("提升的指标：")
        # 解析：标题
        for m in improved:
            # 解析：逐提升
            lines.append(f"- {m}：{base_scores[m]:.3f} → {cur_scores[m]:.3f}（{_fmt_delta(base_scores[m], cur_scores[m])}）")
            # 解析：前后与差值
    if regressed:
        # 解析：有回归
        lines.append("回归的指标：")
        # 解析：标题
        for m in regressed:
            # 解析：逐回归
            lines.append(f"- {m}：{base_scores[m]:.3f} → {cur_scores[m]:.3f}（{_fmt_delta(base_scores[m], cur_scores[m])}）")
            # 解析：前后与差值
    if not improved and not regressed:
        # 解析：持平
        lines.append("各指标与基线持平。")
        # 解析：结论

    # 每题对比（按题目对齐，指标列取两轮共有的）
    cur_by_q = {r["question"]: r for r in cur_rows}
    # 解析：本轮按问题索引
    base_by_q = {r["question"]: r for r in base_rows}
    # 解析：基线按问题索引
    row_metrics = sorted(
        # 解析：两轮共有的指标列
        {k for r in base_rows + cur_rows for k in r if k != "question"}
        # 解析：收集全部指标名
    )
    lines += ["", "## 每题明细对比", ""]
    # 解析：明细段
    headers = ["#", "问题"] + [f"{m}(基→本轮)" for m in row_metrics]
    # 解析：表头
    lines.append("| " + " | ".join(headers) + " |")
    # 解析：表头行
    lines.append("|" + "---|" * len(headers))
    # 解析：分隔行
    questions = list(dict.fromkeys([r["question"] for r in base_rows] + [r["question"] for r in cur_rows]))
    # 解析：两轮全部问题（去重保序）
    for i, q in enumerate(questions, 1):
        # 解析：逐题
        cells = [str(i), q]
        # 解析：序号与问题
        for m in row_metrics:
            # 解析：逐指标
            old = base_by_q.get(q, {}).get(m)
            # 解析：基线值
            new = cur_by_q.get(q, {}).get(m)
            # 解析：本轮值
            if isinstance(old, float) and isinstance(new, float) and old == old and new == new:
                # 解析：双方有效
                cells.append(f"{old:.3f} → {new:.3f}（{_fmt_delta(old, new)}）")
                # 解析：前后与差值
            elif isinstance(new, float) and new == new:
                # 解析：仅本轮有效
                cells.append(f"— → {new:.3f}")
                # 解析：显示本轮
            elif isinstance(old, float) and old == old:
                # 解析：仅基线有效
                cells.append(f"{old:.3f} → —")
                # 解析：显示基线
            else:
                # 解析：都无效
                cells.append("—")
                # 解析：破折号
        lines.append("| " + " | ".join(cells) + " |")
        # 解析：一行一题

    return "\n".join(lines) + "\n"
    # 解析：返回对比报告
