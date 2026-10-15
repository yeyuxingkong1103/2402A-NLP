# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：对 16 个测试问题分别执行 RAG 与 LightRAG 的「检索 → 生成 → RAGAS 评估」，并输出对比

import os                                                       # 导入 os 用于路径与文件判断
import json                                                     # 导入 json 用于结果落盘
import time                                                     # 导入 time 用于统计检索耗时
import config                                                   # 导入本工单配置模块
import corpus                                                   # 导入语料分块模块
import embedder                                                 # 导入嵌入编码模块
import rag_retriever                                            # 导入扁平 RAG 检索器
import lightrag_retriever                                       # 导入 LightRAG 检索器
import references                                               # 导入银标参考答案构建模块
import generator                                                # 导入答案生成模块
import evaluator                                                # 导入 RAGAS 风格评估指标
import llm                                                      # 导入 LLM 封装（用于落盘缓存）
from logger import LOGGER                                       # 导入统一日志对象


def _load_graph():                                              # 定义图谱加载函数
    """加载知识图谱；不存在时给出明确提示，避免后续报出难以定位的错误。"""
    if not os.path.isfile(config.GRAPH_FILE):                   # 图谱文件缺失
        raise RuntimeError("知识图谱不存在，请先运行 kg_build.py")  # 抛出明确错误
    with open(config.GRAPH_FILE, "r", encoding="utf-8") as f:   # 以 UTF-8 打开图谱
        return json.load(f)                                     # 反序列化图谱


def _brief(results, limit=3):                                   # 定义检索结果摘要函数
    """把检索结果压缩为便于写入报告的摘要（名次、来源、页码、召回路径、正文前缀）。"""
    out = []                                                    # 初始化摘要列表
    for r in results[:limit]:                                   # 取前若干条
        out.append({"rank": r["rank"] + 1,                      # 名次（从 1 开始，便于阅读）
                    "source": r["source"],                      # 来源说明书
                    "page": r["page"],                          # 来源页码
                    "via": r.get("via", ""),                    # 召回路径（LightRAG 用）
                    "score": r.get("score", 0.0),               # 融合得分
                    "preview": r["text"][:120].replace("\n", " ")})  # 正文前缀
    return out                                                  # 返回摘要列表


def _timed(func, *args, **kwargs):                              # 定义带计时的调用函数
    """执行函数并返回 (结果, 耗时秒数)，用于对比两套系统的检索延迟。"""
    t0 = time.time()                                            # 记录起始时间
    out = func(*args, **kwargs)                                 # 执行目标函数
    return out, time.time() - t0                                # 返回结果与耗时


def _run_system(name, retrieve_fn, questions, refs):            # 定义单系统评估函数
    """对全部问题执行给定系统的检索与生成，并计算四个评估指标。"""
    per_q = []                                                  # 逐题结果列表
    times = []                                                  # 检索耗时列表
    ref_by_id = {r["id"]: r["reference"] for r in refs}         # 问题编号到参考答案的映射
    for q in questions:                                         # 遍历每个测试问题
        (hits, trace), cost = _timed(retrieve_fn, q["question"])  # 检索并计时（LightRAG 带追溯）
        answer = generator.generate(q["question"], hits)        # 生成答案
        reference = ref_by_id.get(q["id"], "")                  # 取该题的参考答案
        metrics = evaluator.evaluate_one(q["question"], answer, hits, reference)  # 计算四指标
        per_q.append({"id": q["id"], "question": q["question"],  # 记录问题编号与原文
                      "answer": answer,                         # 记录生成的答案
                      "reference": reference,                   # 记录参考答案便于对照
                      "retrieved": _brief(hits),                # 记录检索结果摘要
                      "retrieved_count": len(hits),             # 记录检索块数量
                      "retrieve_seconds": round(cost, 3),       # 记录本次检索耗时
                      "trace": trace,                           # 记录检索追溯（关键词、命中实体）
                      "metrics": metrics})                      # 记录四指标与判定明细
        times.append(cost)                                      # 收集耗时
        LOGGER.info("%s 第 %d/%d 题完成", name, len(per_q), len(questions))  # 输出进度
    summary = evaluator.summarize(per_q)                        # 汇总平均指标
    times_sorted = sorted(times)                                # 耗时排序用于分位数
    stats = {"avg": round(sum(times) / len(times), 3) if times else 0.0,  # 平均检索耗时
             "max": round(max(times), 3) if times else 0.0,     # 最大检索耗时
             "p95": round(times_sorted[int(0.95 * (len(times) - 1))], 3) if times else 0.0,  # 95 分位
             "within_budget": sum(1 for t in times if t <= config.RETRIEVE_TIME_BUDGET),  # 达标次数
             "total": len(times)}                               # 总次数
    LOGGER.info("%s 汇总指标：%s", name, json.dumps(summary, ensure_ascii=False))  # 输出汇总
    LOGGER.info("%s 检索耗时：平均 %.3fs / 最大 %.3fs / 3s 内 %d/%d",
                name, stats["avg"], stats["max"], stats["within_budget"], stats["total"])  # 输出耗时
    return {"name": name, "per_question": per_q, "summary": summary, "timing": stats}  # 返回系统结果


def _compare(rag_sum, lr_sum):                                  # 定义指标对比函数
    """对比两个系统的四个指标，返回逐指标差值表与改善结论。"""
    rows = []                                                   # 初始化对比行
    for k in evaluator.METRIC_KEYS:                             # 遍历四个指标
        a = rag_sum.get(k, 0.0)                                 # RAG 取值
        b = lr_sum.get(k, 0.0)                                  # LightRAG 取值
        rows.append({"metric": k, "rag": a, "lightrag": b,      # 记录两侧取值
                     "delta": round(b - a, 4),                  # 绝对变化
                     "better": b > a})                          # 是否优于 RAG
    improved = sum(1 for r in rows if r["better"])              # 统计改善项数
    return {"rows": rows, "improved": improved, "total": len(rows)}  # 返回对比结论


def run():                                                      # 定义评估主函数
    """执行完整评估流程：构建参考答案 → 两套系统逐题检索生成 → 指标计算 → 落盘与打印。"""
    chunks = corpus.build_or_load_chunks()                      # 加载语料块
    model = embedder.load_model()                               # 加载嵌入模型
    base = rag_retriever.build_index(chunks, model)             # 构建扁平 RAG 索引（向量 + BM25）
    refs = references.build_references(chunks, base.bm25)       # 构建银标参考答案
    graph = _load_graph()                                       # 加载知识图谱
    lr_index = lightrag_retriever.build_index(model, graph, base)  # 构建 LightRAG 索引（含实体向量）
    LOGGER.info("图谱规模：%d 实体，%d 关系", len(graph["entities"]), len(graph["relations"]))  # 输出图谱规模

    rag = _run_system("RAG", lambda q: (base.retrieve(model, q), None),  # 评估扁平 RAG
                      config.QUESTIONS, refs)                   # 传入问题与参考答案
    light = _run_system("LightRAG", lambda q: lr_index.retrieve(q, with_trace=True),  # 评估 LightRAG
                        config.QUESTIONS, refs)                 # 传入问题与参考答案
    compare = _compare(rag["summary"], light["summary"])        # 对比两套系统指标

    result = {"meta": {"chunk_total": len(chunks),              # 语料块总数
                       "corpus_chars": sum(len(c["text"]) for c in chunks),  # 语料总字数
                       "entity_total": len(graph["entities"]),  # 实体总数
                       "relation_total": len(graph["relations"]),  # 关系总数
                       "question_total": len(config.QUESTIONS),  # 测试问题数
                       "final_top_k": config.FINAL_TOP_K,       # 最终上下文块数
                       "retrieve_budget": config.RETRIEVE_TIME_BUDGET},  # 检索时间预算
              "references": refs,                               # 参考答案
              "systems": {"RAG": rag, "LightRAG": light},       # 两套系统完整结果
              "compare": compare}                               # 指标对比
    config.ensure_dirs()                                        # 确保目录存在
    with open(config.EVAL_RESULT_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 打开结果文件
        json.dump(result, f, ensure_ascii=False, indent=1)      # 写入评估结果
    llm.save_cache()                                            # 落盘 LLM 磁盘缓存
    _print_report(result)                                       # 打印控制台对比报告
    LOGGER.info("评估结果已落盘 %s", config.EVAL_RESULT_FILE)     # 输出落盘路径
    return result                                               # 返回评估结果


def _print_report(result):                                      # 定义控制台报告打印函数
    """把两套系统的指标与耗时对比打印为对齐的文本表格。"""
    rag, light = result["systems"]["RAG"], result["systems"]["LightRAG"]  # 取两套系统结果
    print("\n================ RAGAS 指标对比（%d 题）================" % result["meta"]["question_total"])  # 表头
    print("%-22s %10s %10s %10s %s" % ("指标", "RAG", "LightRAG", "变化", "结论"))  # 列名
    for r in result["compare"]["rows"]:                         # 遍历对比行
        print("%-22s %10.4f %10.4f %+10.4f %s" % (r["metric"], r["rag"], r["lightrag"],  # 打印指标与两侧取值
                                                  r["delta"], "LightRAG 更优" if r["better"] else "RAG 更优"))  # 打印结论
    print("-" * 62)                                             # 分隔线
    print("LightRAG 更优指标数：%d/%d" % (result["compare"]["improved"], result["compare"]["total"]))  # 汇总
    print("\n================ 检索耗时对比（基线记录，无时限要求）================")  # 耗时表头
    print("%-12s %10s %10s %10s %s" % ("系统", "平均(s)", "最大(s)", "P95(s)", "≤3s 次数"))  # 列名
    for name in ("RAG", "LightRAG"):                            # 遍历两套系统
        t = result["systems"][name]["timing"]                   # 取耗时统计
        print("%-12s %10.3f %10.3f %10.3f %d/%d" % (name, t["avg"], t["max"], t["p95"],  # 打印耗时指标
                                                    t["within_budget"], t["total"]))  # 打印达标次数
    print("=" * 62)                                             # 结束分隔线


if __name__ == "__main__":                                      # 仅在直接运行时执行
    run()                                                       # 启动评估
