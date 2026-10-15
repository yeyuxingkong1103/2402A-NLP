# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：把评估结果渲染为 JSON 与 Markdown 报告（含优化前后指标对比与性能统计）

import os                                                       # 导入 os 用于路径处理
import json                                                     # 导入 json 用于报告序列化
from datetime import datetime                                   # 导入日期时间用于报告时间戳
import config                                                   # 导入本工单配置模块

# 报告中展示的指标名称与中文标题的映射
METRIC_TITLES = {                                               # 指标键 -> 中文标题
    "context_precision": "上下文精度 (Context Precision)",       # 上下文精度
    "context_recall": "上下文召回 (Context Recall)",             # 上下文召回
    "faithfulness": "忠实度 (Faithfulness)",                     # 忠实度
    "answer_relevancy": "答案相关性 (Answer Relevancy)",         # 答案相关性
}


def save_json(name, payload):                                   # 定义 JSON 报告保存函数
    """把任意可序列化对象写入 evaluation/<name>.json。"""
    config.ensure_dirs()                                        # 确保目录存在
    path = os.path.join(config.EVAL_DIR, name + ".json")        # 计算输出路径
    with open(path, "w", encoding="utf-8") as f:                # 以 UTF-8 打开文件
        json.dump(payload, f, ensure_ascii=False, indent=2)     # 格式化写入便于人工阅读
    return path                                                 # 返回输出路径


def summarize(records):                                         # 定义指标汇总函数
    """对记录列表按模式求各指标平均值，并汇总性能指标。"""
    summary = {}                                                # 初始化汇总字典
    for mode in ("baseline", "optimized"):                      # 遍历两种模式
        items = [r for r in records if r.get("mode") == mode]   # 过滤该模式的记录
        if not items:                                           # 无记录则跳过
            summary[mode] = {}                                  # 置为空字典
            continue                                            # 继续下一模式
        entry = {}                                              # 该模式的汇总项
        for key in METRIC_TITLES:                               # 遍历全部指标
            vals = [r["metrics"].get(key, 0.0) for r in items if "metrics" in r]  # 收集指标值
            entry[key] = round(sum(vals) / len(vals), 4) if vals else 0.0  # 计算平均值
        rts = [r.get("retrieve_seconds", 0.0) for r in items]    # 收集检索耗时
        tts = [r.get("timing", {}).get("total", 0.0) for r in items]  # 收集总耗时
        entry["retrieve_seconds_avg"] = round(sum(rts) / len(rts), 4) if rts else 0.0  # 检索平均耗时
        entry["retrieve_seconds_max"] = round(max(rts), 4) if rts else 0.0  # 检索最大耗时
        entry["total_seconds_avg"] = round(sum(tts) / len(tts), 4) if tts else 0.0  # 总平均耗时
        entry["n"] = len(items)                                 # 样本数
        summary[mode] = entry                                   # 记录该模式汇总
    return summary                                              # 返回汇总结果


def _fmt(v):                                                    # 定义数值格式化函数
    """把数值格式化为 4 位小数字符串。"""
    try:                                                        # 尝试格式化
        return "%.4f" % float(v)                                # 返回格式化结果
    except Exception:                                           # 非数值时容错
        return str(v)                                           # 直接转字符串


def render_markdown(summary, records, meta):                    # 定义 Markdown 报告渲染函数
    """生成包含优化前后对比表、逐题明细与性能统计的 Markdown 文本。"""
    lines = []                                                  # 初始化行列表
    lines.append("# 工单九 Graph RAG 优化 —— 评估报告")           # 报告标题
    lines.append("")                                            # 空行
    lines.append("- 工单编号：人工智能NLP-RAG-Graph RAG 优化任务")  # 工单编号
    lines.append("- 生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # 生成时间
    lines.append("- 语料规模：%d 个文本块，%d 个来源文件" % (meta.get("n_chunks", 0),  # 语料规模
                                                             meta.get("n_sources", 0)))  # 来源文件数
    lines.append("- 图谱规模：优化后 %d 实体 / %d 关系；优化前 %d 实体 / %d 关系" % (
        meta.get("graph_opt", {}).get("entities", 0),           # 优化版实体数
        meta.get("graph_opt", {}).get("relations", 0),          # 优化版关系数
        meta.get("graph_base", {}).get("entities", 0),          # 基线版实体数
        meta.get("graph_base", {}).get("relations", 0)))        # 基线版关系数
    lines.append("- 评估问题数：%d" % meta.get("n_questions", 0))  # 问题数
    lines.append("")                                            # 空行

    lines.append("## 一、优化前后指标对比")                        # 一级小标题
    lines.append("")                                            # 空行
    lines.append("| 指标 | 优化前 (Baseline) | 优化后 (Optimized) | 变化 |")  # 表头
    lines.append("| --- | --- | --- | --- |")                    # 表格分隔行
    base, opt = summary.get("baseline", {}), summary.get("optimized", {})  # 取两种模式汇总
    for key, title in METRIC_TITLES.items():                     # 遍历指标
        b, o = base.get(key, 0.0), opt.get(key, 0.0)             # 取两模式数值
        lines.append("| %s | %s | %s | %+.4f |" % (title, _fmt(b), _fmt(o), o - b))  # 追加对比行
    lines.append("| 检索平均耗时(秒) | %s | %s | %+.4f |" % (
        _fmt(base.get("retrieve_seconds_avg", 0.0)),             # 优化前检索平均耗时
        _fmt(opt.get("retrieve_seconds_avg", 0.0)),              # 优化后检索平均耗时
        opt.get("retrieve_seconds_avg", 0.0) - base.get("retrieve_seconds_avg", 0.0)))  # 差值
    lines.append("| 端到端平均耗时(秒) | %s | %s | %+.4f |" % (
        _fmt(base.get("total_seconds_avg", 0.0)),                # 优化前端到端平均耗时
        _fmt(opt.get("total_seconds_avg", 0.0)),                 # 优化后端到端平均耗时
        opt.get("total_seconds_avg", 0.0) - base.get("total_seconds_avg", 0.0)))  # 差值
    lines.append("")                                            # 空行

    lines.append("## 二、验收标准达成情况")                        # 二级标题
    lines.append("")                                            # 空行
    cp = opt.get("context_precision", 0.0)                       # 优化后上下文精度
    cr = opt.get("context_recall", 0.0)                          # 优化后上下文召回
    lines.append("- 上下文精度 ≥ 0.80：%s（实测 %s）" % ("✅ 达标" if cp >= 0.8 else "❌ 未达标", _fmt(cp)))  # 精度判定
    lines.append("- 上下文召回 ≥ 0.90：%s（实测 %s）" % ("✅ 达标" if cr >= 0.9 else "❌ 未达标", _fmt(cr)))  # 召回判定
    rmax = opt.get("retrieve_seconds_max", 0.0)                  # 检索最大耗时
    lines.append("- 检索时间 ≤ 3.00 秒：%s（实测最大 %s 秒）" % (
        "✅ 达标" if rmax <= 3.0 else "❌ 未达标", _fmt(rmax)))    # 时间判定
    lines.append("")                                            # 空行

    lines.append("## 三、结论")                                   # 结论小标题
    lines.append("")                                            # 空行
    bs = base.get("retrieve_seconds_avg", 0.0)                   # 优化前检索平均耗时
    os_ = opt.get("retrieve_seconds_avg", 0.0)                   # 优化后检索平均耗时
    if bs > 0:                                                  # 避免除零
        lines.append("- 检索平均耗时由 %.4f 秒降至 %.4f 秒，降幅 %.1f%%，%s 工单 ≤3 秒要求。" % (
            bs, os_, (bs - os_) / bs * 100.0,                   # 降幅
            "满足" if opt.get("retrieve_seconds_max", 9.9) <= 3.0 else "未完全满足"))  # 达标判定
    worst = min((r for r in records if r.get("mode") == "optimized"),  # 找出优化后精度最低的题
                key=lambda r: r.get("metrics", {}).get("context_precision", 0.0), default=None)
    if worst is not None:                                       # 存在记录时输出定位信息
        lines.append("- 优化后精度最低的是 Q%d（%.4f），该题上下文召回为 %.4f，是拉低整体均值的主要来源。" % (
            worst.get("qid", 0), worst.get("metrics", {}).get("context_precision", 0.0),  # 精度
            worst.get("metrics", {}).get("context_recall", 0.0)))  # 召回
    lines.append("- 未达标原因与改进方向详见同目录《优化方案.md》第三节「瓶颈分析」。" )  # 指向方案文档
    lines.append("")                                            # 空行

    lines.append("## 四、逐题明细")                               # 三级标题
    for mode, label in (("baseline", "优化前"), ("optimized", "优化后")):  # 遍历模式
        lines.append("")                                        # 空行
        lines.append("### %s" % label)                           # 小标题
        for r in [x for x in records if x.get("mode") == mode]:  # 遍历该模式记录
            m = r.get("metrics", {})                             # 取指标
            lines.append("")                                    # 空行
            lines.append("**Q%d（%s）**：%s" % (r.get("qid", 0),     # 题号与模式
                                                r.get("lang", "zh"), r.get("question", "")))  # 语言与问题
            lines.append("")                                    # 空行
            lines.append("- 上下文精度 %s ｜ 上下文召回 %s ｜ 忠实度 %s ｜ 答案相关性 %s" % (
                _fmt(m.get("context_precision")), _fmt(m.get("context_recall")),  # 精度与召回
                _fmt(m.get("faithfulness")), _fmt(m.get("answer_relevancy"))))    # 忠实度与相关性
            lines.append("- 检索耗时 %s 秒 ｜ 端到端 %s 秒" % (
                _fmt(r.get("retrieve_seconds")), _fmt(r.get("timing", {}).get("total"))))  # 耗时
            lines.append("")                                    # 空行
            lines.append("> 回答：%s" % (r.get("answer", "").replace("\n", " ")[:500]))  # 回答摘要
    return "\n".join(lines)                                     # 返回完整 Markdown 文本


def save_markdown(name, content):                               # 定义 Markdown 报告保存函数
    """把 Markdown 文本写入 evaluation/<name>.md。"""
    config.ensure_dirs()                                        # 确保目录存在
    path = os.path.join(config.EVAL_DIR, name + ".md")          # 计算输出路径
    with open(path, "w", encoding="utf-8") as f:                # 以 UTF-8 打开
        f.write(content)                                        # 写入内容
    return path                                                 # 返回输出路径
