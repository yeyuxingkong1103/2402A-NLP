# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：对优化前后两条流水线执行基准测试，统计各阶段耗时分布并输出对比结果

import os                                                       # 导入 os 用于路径拼接
import json                                                     # 导入 json 用于结果落盘
import statistics                                               # 导入 statistics 用于中位数计算
import config                                                   # 导入本工单配置模块
import resources                                                # 导入共享资源加载模块
import baseline_pipeline                                        # 导入优化前流水线
import optimized_pipeline                                       # 导入优化后流水线
from logger import StageTracer, LOGGER                          # 导入阶段追踪器与日志对象

BENCH_JSON = os.path.join(config.DATA_DIR, "benchmark.json")    # 基准测试原始结果
COMPARE_MD = os.path.join(config.EVAL_DIR, "性能对比.md")        # 性能对比报告


def _pct(values, q):                                            # 定义分位数计算函数
    """返回升序序列的 q 分位数（线性插值，q 取 0~1）。"""
    if not values:                                              # 空序列
        return 0.0                                              # 返回 0
    ordered = sorted(values)                                    # 升序排列
    pos = q * (len(ordered) - 1)                                # 插值位置
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)      # 相邻下标
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)  # 线性插值


def _stat(values):                                              # 定义统计摘要函数
    """返回一组耗时的均值、中位数、P95 与最大值。"""
    if not values:                                              # 空序列
        return {"avg": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}  # 返回全 0
    return {"avg": round(statistics.fmean(values), 6),           # 均值
            "median": round(statistics.median(values), 6),       # 中位数
            "p95": round(_pct(values, 0.95), 6),                 # 95 分位
            "max": round(max(values), 6)}                        # 最大值


def run_system(name, pipe, repeats, persist=True):              # 定义单系统基准测试函数
    """对全部测试查询重复执行给定流水线，逐次记录五阶段耗时与检索口径耗时。"""
    records = []                                                # 逐次记录列表
    for rnd in range(repeats):                                  # 重复轮次
        for q in config.QUERIES:                                # 遍历测试查询
            tracer = StageTracer(query=q["query"], persist=persist)  # 创建带 request_id 的追踪器
            resp = pipe.run(q["query"], tracer)                 # 执行一次完整流程
            summary = tracer.summary()                          # 取阶段耗时摘要
            summary.update({"system": name, "query_id": q["id"],  # 补充系统名与查询编号
                            "round": rnd,                       # 记录轮次（用于区分冷启动与稳态）
                            "answer_chars": len(resp["answer"])})  # 记录答案长度
            records.append(summary)                             # 收集记录
        LOGGER.info("%s 第 %d 轮完成，累计 %d 次", name, rnd + 1, len(records))  # 输出进度
    return records                                              # 返回逐次记录


def summarize(records):                                         # 定义汇总统计函数
    """把逐次记录汇总为「各阶段耗时分布 + 检索口径分布 + 超时率」。"""
    stages = {}                                                 # 阶段名 -> 耗时列表
    for s in config.STAGES:                                     # 初始化五个阶段
        stages[s] = [r["stages"].get(s, 0.0) for r in records]  # 收集该阶段逐次耗时
    ret = [r["retrieval_seconds"] for r in records]             # 检索口径耗时列表
    cold = [r for r in records if r["round"] == 0]              # 第一轮（含冷启动）
    steady = [r for r in records if r["round"] > 0]             # 后续轮（稳态）
    return {"stages": {k: _stat(v) for k, v in stages.items()},  # 各阶段统计
            "retrieval": _stat(ret),                            # 检索口径统计
            "first_query": round(records[0]["retrieval_seconds"], 6),  # 首次查询检索耗时（冷启动）
            "cold": _stat([r["retrieval_seconds"] for r in cold]),  # 第一轮检索耗时统计
            "steady": _stat([r["retrieval_seconds"] for r in steady]) if steady  # 稳态检索耗时统计
                      else {"avg": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0},  # 无稳态数据时置零
            "within_budget": sum(1 for r in records if r["within_budget"]),  # 满足 3 秒的次数
            "total": len(records)}                              # 总次数


def compare(base_sum, opt_sum):                                 # 定义对比结论函数
    """生成各阶段与检索口径的对比行，并给出加速比。"""
    rows = []                                                   # 对比行列表
    for s in list(config.STAGES) + ["检索结果返回（阶段1+2+3）"]:  # 五个阶段加检索口径汇总
        if s.startswith("检索结果返回"):                          # 汇总行取检索口径统计
            b, o = base_sum["retrieval"], opt_sum["retrieval"]  # 两侧检索口径统计
        else:                                                   # 单阶段行
            b, o = base_sum["stages"][s], opt_sum["stages"][s]  # 两侧该阶段统计
        rows.append({"item": s, "before_avg": b["avg"], "after_avg": o["avg"],  # 优化前后均值
                     "speedup": round(b["avg"] / o["avg"], 2) if o["avg"] > 1e-6 else 0.0,  # 加速比
                     "before_p95": b["p95"], "after_p95": o["p95"]})  # P95 对比
    return rows                                                 # 返回对比行


def _md_table(headers, rows):                                   # 定义 Markdown 表格渲染函数
    """把表头与数据行渲染为 Markdown 表格。"""
    out = ["| " + " | ".join(headers) + " |",                   # 表头行
           "|" + "|".join(["---"] * len(headers)) + "|"]        # 分隔行
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]  # 数据行
    return "\n".join(out)                                       # 拼接为完整表格


def draw_chart(base_sum, opt_sum, path):                        # 定义耗时对比图绘制函数
    """绘制优化前后各阶段平均耗时对比柱状图。"""
    import matplotlib                                           # 延迟导入绘图库
    matplotlib.use("Agg")                                       # 使用无界面后端
    import matplotlib.pyplot as plt                             # 导入绘图接口
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 中文字体
    plt.rcParams["axes.unicode_minus"] = False                  # 正常显示负号
    labels = list(config.STAGES)                                # 阶段名称
    before = [base_sum["stages"][s]["avg"] for s in labels]     # 优化前各阶段均值
    after = [opt_sum["stages"][s]["avg"] for s in labels]       # 优化后各阶段均值
    idx = range(len(labels))                                    # 横轴下标
    plt.figure(figsize=(12, 6))                                 # 创建画布
    plt.bar([i - 0.2 for i in idx], before, width=0.4, label="优化前")  # 优化前柱体
    plt.bar([i + 0.2 for i in idx], after, width=0.4, label="优化后")   # 优化后柱体
    plt.axhline(config.RETRIEVAL_BUDGET, color="red", linestyle="--",   # 3 秒验收线
                label="检索预算 %.0f 秒" % config.RETRIEVAL_BUDGET)     # 图例文字
    plt.xticks(list(idx), labels, rotation=15)                  # 横轴标签
    plt.ylabel("平均耗时（秒）")                                  # 纵轴标题
    plt.title("RAG 各阶段平均耗时：优化前 vs 优化后")              # 图标题
    plt.legend()                                                # 显示图例
    plt.tight_layout()                                          # 自动调整边距
    plt.savefig(path, dpi=150)                                  # 保存图片
    plt.close()                                                 # 关闭画布
    LOGGER.info("耗时对比图已生成 %s", path)                       # 输出日志


def write_report(base_sum, opt_sum, rows, path):                # 定义报告生成函数
    """生成性能对比 Markdown 报告。"""
    lines = ["# RAG 检索性能优化前后对比", "",                      # 报告标题
             "- 测试集：工单十二的 %d 个验收问题" % len(config.QUERIES),  # 测试集说明
             "- 重复轮次：每查询 %d 次，共 %d 次/系统" % (config.BENCH_REPEATS,
                                                        len(config.QUERIES) * config.BENCH_REPEATS),
             "- 验收标准：检索结果返回时间 ≤%.0f 秒" % config.RETRIEVAL_BUDGET, ""]  # 验收标准
    rows_md = [[r["item"], "%.4f" % r["before_avg"], "%.4f" % r["after_avg"],  # 对比行渲染
                "%.2fx" % r["speedup"], "%.4f" % r["before_p95"], "%.4f" % r["after_p95"]]
               for r in rows]                                   # 逐行渲染
    lines += ["## 一、各阶段平均耗时对比", "",                        # 小节标题
              _md_table(["阶段", "优化前均值(s)", "优化后均值(s)", "加速比",
                         "优化前P95(s)", "优化后P95(s)"], rows_md), ""]  # 阶段对比表
    lines += ["## 二、检索结果返回耗时分布", "",                       # 小节标题
              _md_table(["指标", "优化前", "优化后"], [             # 分布对比表
                  ["首次查询（会话冷启动）", "%.4f" % base_sum["first_query"],
                   "%.4f" % opt_sum["first_query"]],
                  ["第一轮均值", "%.4f" % base_sum["cold"]["avg"], "%.4f" % opt_sum["cold"]["avg"]],
                  ["稳态均值", "%.4f" % base_sum["steady"]["avg"], "%.4f" % opt_sum["steady"]["avg"]],
                  ["稳态中位数", "%.4f" % base_sum["steady"]["median"], "%.4f" % opt_sum["steady"]["median"]],
                  ["整体 P95", "%.4f" % base_sum["retrieval"]["p95"], "%.4f" % opt_sum["retrieval"]["p95"]],
                  ["整体最大值", "%.4f" % base_sum["retrieval"]["max"], "%.4f" % opt_sum["retrieval"]["max"]],
                  ["3 秒内达标率", "%d/%d" % (base_sum["within_budget"], base_sum["total"]),
                   "%d/%d" % (opt_sum["within_budget"], opt_sum["total"])]]), ""]  # 达标率对比
    lines += ["## 三、耗时对比图", "", "- `stage_time.png`：各阶段平均耗时柱状图（含 3 秒预算线）", ""]  # 图说明
    config.ensure_dirs()                                        # 确保输出目录存在
    with open(path, "w", encoding="utf-8") as f:                # 以 UTF-8 打开报告文件
        f.write("\n".join(lines))                               # 写入报告内容
    LOGGER.info("性能对比报告已生成 %s", path)                      # 输出日志


def main(persist=True):                                         # 定义基准测试主函数
    """构建两条流水线并执行基准测试，输出对比报告与图表。"""
    config.ensure_dirs()                                        # 确保目录存在
    chunks = resources.load_chunks()                            # 加载共享语料
    model = resources.load_model()                              # 加载嵌入模型（两条流水线共用）
    base = baseline_pipeline.build(chunks=chunks, model=model)[0]  # 构建优化前流水线（无预热）
    opt = optimized_pipeline.build(chunks=chunks, model=model, warm=True)  # 构建优化后流水线（含预热）
    base_records = run_system("优化前", base, config.BENCH_REPEATS, persist=persist)  # 跑优化前
    opt_records = run_system("优化后", opt, config.BENCH_REPEATS, persist=persist)  # 跑优化后
    base_sum = summarize(base_records)                          # 汇总优化前
    opt_sum = summarize(opt_records)                            # 汇总优化后
    rows = compare(base_sum, opt_sum)                           # 生成对比行
    draw_chart(base_sum, opt_sum, os.path.join(config.EVAL_DIR, "stage_time.png"))  # 生成对比图
    write_report(base_sum, opt_sum, rows, COMPARE_MD)           # 生成对比报告
    data = {"before": base_sum, "after": opt_sum, "compare": rows,  # 汇总结果
            "records": {"before": base_records, "after": opt_records}}  # 保留逐次明细
    with open(BENCH_JSON, "w", encoding="utf-8") as f:          # 以 UTF-8 打开结果文件
        json.dump(data, f, ensure_ascii=False, indent=1)        # 写入基准测试结果
    return data                                                 # 返回结果


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动基准测试
