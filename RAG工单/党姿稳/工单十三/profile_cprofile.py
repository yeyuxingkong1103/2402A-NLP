# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：用 cProfile 对优化前后流水线做函数级性能剖析，输出累计耗时最高的代码路径

import os                                                       # 导入 os 用于路径拼接
import io                                                       # 导入 io 用于捕获剖析输出
import json                                                     # 导入 json 用于结果落盘
import cProfile                                                 # 导入 cProfile 标准库剖析器
import pstats                                                   # 导入 pstats 用于统计剖析结果
import config                                                   # 导入本工单配置模块
import resources                                                # 导入共享资源加载模块
import baseline_pipeline                                        # 导入优化前流水线
import optimized_pipeline                                       # 导入优化后流水线
from logger import StageTracer, LOGGER                          # 导入追踪器与日志对象

TOP_N = 20                                                      # 输出累计耗时最高的函数数量


def profile_pipeline(name, pipe, queries, prof_path):           # 定义单流水线剖析函数
    """对给定流水线跑若干查询做 cProfile 剖析，返回累计耗时 Top-N 统计文本。"""
    profiler = cProfile.Profile()                               # 创建剖析器
    profiler.enable()                                           # 开始采集
    for q in queries:                                           # 遍历待剖析查询
        tracer = StageTracer(query=q["query"], persist=False)   # 创建不落盘的追踪器
        pipe.run(q["query"], tracer)                            # 执行一次完整流程
    profiler.disable()                                          # 停止采集
    profiler.dump_stats(prof_path)                              # 保存原始剖析数据（可用 snakeviz 打开）
    buf = io.StringIO()                                         # 创建内存文本缓冲
    stats = pstats.Stats(profiler, stream=buf)                  # 载入统计数据
    stats.sort_stats("cumulative").print_stats(TOP_N)           # 按累计耗时排序输出前 N 项
    text = buf.getvalue()                                       # 取出文本
    LOGGER.info("%s 剖析完成，数据保存至 %s", name, prof_path)     # 输出日志
    return text                                                 # 返回统计文本


def top_functions(prof_path, n=TOP_N, project_only=False):      # 定义热点函数提取函数
    """从剖析数据中提取累计耗时最高的函数列表；project_only 为真时只保留本工单模块。"""
    stats = pstats.Stats(prof_path)                             # 载入剖析数据
    rows = []                                                   # 结果行
    for (fname, lineno, func), (cc, nc, tt, ct, _) in stats.stats.items():  # 遍历全部函数统计
        # 只看本项目代码：必须是 .py 且所在目录恰好是本工单目录。
        # 内置函数与 C 扩展的 fname 是 "<built-in method ...>" 或 "~"，
        # 若只用 startswith 判断，abspath("~") 会落在当前目录上而被误判为本项目代码。
        if project_only and (not fname.endswith(".py")                     # 排除非 .py（内置与 C 扩展）
                             or os.path.dirname(os.path.abspath(fname)) != config.BASE_DIR):  # 排除其他目录
            continue                                            # 跳过标准库与第三方库
        rows.append({"func": func, "file": os.path.basename(fname),  # 函数名与所在文件
                     "line": lineno, "calls": nc,               # 行号与调用次数
                     "tottime": round(tt, 6), "cumtime": round(ct, 6)})  # 自身耗时与累计耗时
    rows.sort(key=lambda x: -x["cumtime"])                      # 按累计耗时降序
    return rows[:n]                                             # 返回前 N 项


def _fmt_table(rows):                                           # 定义文本表格渲染函数
    """把热点函数列表渲染为等宽文本表格。"""
    out = ["%-38s %20s %8s %10s %10s" % ("函数", "文件:行", "调用次数", "自身(s)", "累计(s)")]  # 表头
    for r in rows:                                              # 遍历热点函数
        out.append("%-38s %20s %8d %10.4f %10.4f" % (r["func"][:38],  # 函数名（截断）
                                                     "%s:%d" % (r["file"][:16], r["line"]),  # 文件与行号
                                                     r["calls"], r["tottime"], r["cumtime"]))  # 三项指标
    return "\n".join(out)                                       # 拼接为文本


def main():                                                     # 定义剖析主函数
    """对优化前后两条流水线各做一次 cProfile 剖析，输出热点函数报告。"""
    config.ensure_dirs()                                        # 确保目录存在
    chunks = resources.load_chunks()                            # 加载语料
    model = resources.load_model()                              # 加载嵌入模型
    probe = [config.QUERIES[13], config.QUERIES[15]]            # 取两条查询做剖析（控制耗时）
    base = baseline_pipeline.build(chunks=chunks, model=model)[0]  # 构建优化前流水线
    opt = optimized_pipeline.build(chunks=chunks, model=model, warm=True)  # 构建优化后流水线
    base_prof = os.path.join(config.DATA_DIR, "before.prof")    # 优化前剖析数据路径
    opt_prof = os.path.join(config.DATA_DIR, "after.prof")      # 优化后剖析数据路径
    profile_pipeline("优化前", base, probe, base_prof)           # 剖析优化前
    profile_pipeline("优化后", opt, probe, opt_prof)             # 剖析优化后
    result = {"probe_queries": [q["id"] for q in probe],        # 参与剖析的查询编号
              "before": top_functions(base_prof),               # 优化前热点函数（全部）
              "after": top_functions(opt_prof),                 # 优化后热点函数（全部）
              "before_project": top_functions(base_prof, project_only=True),  # 优化前（只看本项目）
              "after_project": top_functions(opt_prof, project_only=True),    # 优化后（只看本项目）
              "before_prof": base_prof, "after_prof": opt_prof}  # 原始剖析数据路径
    out = os.path.join(config.DATA_DIR, "cprofile_top.json")    # 汇总结果路径
    with open(out, "w", encoding="utf-8") as f:                 # 以 UTF-8 打开结果文件
        json.dump(result, f, ensure_ascii=False, indent=1)      # 写入汇总结果
    for tag, key in (("优化前", "before_project"), ("优化后", "after_project")):  # 先看本项目自己的代码
        print("\n======== %s 本项目模块热点（累计耗时 Top%d）========" % (tag, TOP_N))  # 输出标题
        print(_fmt_table(result[key]))                          # 打印本项目热点表
    for tag, key in (("优化前", "before"), ("优化后", "after")):  # 再看包含库调用的全量热点
        print("\n======== %s 全量热点（累计耗时 Top%d，含库调用）========" % (tag, TOP_N))  # 输出标题
        print(_fmt_table(result[key]))                          # 打印全量热点表
    print("\n原始剖析数据：%s / %s" % (base_prof, opt_prof))       # 提示原始数据位置
    print("可视化：python -m snakeviz \"%s\"" % base_prof)        # 提示 snakeviz 用法
    LOGGER.info("cProfile 剖析结果已写入 %s", out)                # 输出完成日志
    return result                                               # 返回剖析结果


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动剖析
