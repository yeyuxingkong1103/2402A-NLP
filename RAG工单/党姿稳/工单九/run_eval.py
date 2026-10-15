# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：批量执行『优化前 vs 优化后』问答与 RAGAS 风格评估，并输出 JSON/Markdown 报告
# 使用方式：python run_eval.py

import os                                                       # 导入 os 用于路径与参数处理
import sys                                                      # 导入 sys 用于注册模块路径与命令行参数
import time                                                     # 导入 time 用于统计整体耗时
import json                                                     # 导入 json 用于评估记录落盘

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 把本目录加入模块搜索路径

import config                                                   # 导入本工单配置模块
import data_loader                                              # 导入语料与问题加载
import graph_builder                                            # 导入图谱构建
import evaluator                                                # 导入 RAGAS 风格指标
import report                                                   # 导入报告渲染与保存
import llm                                                      # 导入 LLM 封装（用于缓存落盘）
from build_index import build_index                             # 导入索引构建入口
from pipeline import GraphRAGPipeline                           # 导入问答管线
from timing import LOGGER                                       # 导入统一日志对象


def evaluate_one(pipeline, qa, mode):                           # 定义单题评估函数
    """对一个问题执行指定模式的问答并计算全部指标。"""
    if mode == "optimized":                                     # 优化后模式
        res = pipeline.answer_optimized(qa["question"])         # 执行优化后管线
    else:                                                       # 优化前模式
        res = pipeline.answer_baseline(qa["question"])          # 执行基线管线
    cp, cp_detail = evaluator.context_precision(qa["question"], res["contexts"], qa["reference"])  # 上下文精度
    cr, cr_detail = evaluator.context_recall(qa["question"], res["contexts"], qa["reference"])  # 上下文召回
    faith = evaluator.faithfulness(res["answer"], res["contexts"])  # 忠实度
    relev = _answer_relevancy(qa["question"], res["answer"])    # 答案相关性
    res["qid"] = qa["id"]                                       # 记录题号
    res["reference"] = qa["reference"]                          # 记录参考答案
    res["metrics"] = {"context_precision": round(cp, 4),        # 上下文精度
                      "context_recall": round(cr, 4),           # 上下文召回
                      "faithfulness": round(faith, 4),          # 忠实度
                      "answer_relevancy": round(relev, 4)}      # 答案相关性
    res["details"] = {"context_precision": cp_detail,           # 精度判定明细
                      "context_recall": cr_detail}              # 召回判定明细
    return res                                                  # 返回完整评估记录


def _answer_relevancy(question, answer):                        # 定义答案相关性评估函数
    """判定答案是否切题（0~1），用于补充 RAGAS 的 answer_relevancy 指标。"""
    if not answer or len(answer) < 5:                           # 答案过短视为不相关
        return 0.0                                              # 返回 0
    system = "你是答案质量评估专家。判断给定答案是否直接、准确地回应了用户问题。"  # 系统提示
    user = """问题：%s

答案：%s

请从『是否切题、是否直接回应问题、是否包含实质信息』三方面打分，
只输出 JSON：{"verdicts": [1 或 0]}，1 表示切题且有效，0 表示不切题或空泛。""" % (
        question, answer[:2200])                                # 拼装评估请求（截断答案控制长度：
                                                                # 优化后答案可达 2100 字，截得太短会把结论段切掉）
    # max_tokens 需留足余量：答案较长时模型可能先输出一段判断说明，
    # 若在此被截断则 JSON 不完整、解析失败而被误判为 0 分
    obj = llm.chat_json(system, user, temperature=0.0, max_tokens=800)  # 调用 LLM 判定
    if not isinstance(obj, (dict, list)):                       # 解析失败（模型可能输出了额外解释文字）
        LOGGER.warning("答案相关性首次判定解析失败，强制绕过缓存重试一次")  # 记录告警
        obj = llm.chat_json(system, user, temperature=0.0, max_tokens=800, use_cache=False)  # 无缓存重试（模型偶发格式抖动）
    if isinstance(obj, dict):                                   # 优先按约定的 verdicts 字段取值
        arr = obj.get("verdicts")                               # 取出判定数组
    else:                                                       # 兼容模型直接返回裸数组的情形
        arr = obj                                                # 直接把返回值当作数组
    if isinstance(arr, list) and arr:                           # 判定数组有效
        first = arr[0]                                          # 取第一个判定
        if isinstance(first, dict):                             # 兼容 [{"score": 1}] 这类结构
            first = first.get("score", first.get("verdict", 0))  # 从字典中取出分值
        return 1.0 if str(first).strip() in ("1", "true", "True", "是") else 0.0  # 归一化为 0/1
    LOGGER.warning("答案相关性判定解析失败，原始输出：%r", (obj if obj is None else str(obj)[:120]))  # 记录失败样本
    return 0.0                                                  # 判定无效时返回 0


def main():                                                     # 定义主函数
    t0 = time.time()                                            # 记录整体起始时间
    LOGGER.info("=== 工单九 Graph RAG 优化评估开始 ===")          # 输出开始日志
    retriever, chunks = build_index(use_cache=True)             # 构建或加载检索索引
    questions = data_loader.load_questions()                    # 加载验收问题
    if not questions:                                           # 无问题则直接结束
        LOGGER.error("未加载到验收问题，评估终止")                # 记录错误
        return 1                                                # 返回非零退出码

    LOGGER.info("开始构建知识图谱（优化后，需多次调用 LLM，请耐心等待）")  # 提示长耗时步骤
    graph_opt = graph_builder.build_graph(chunks, retriever.vectors, questions,  # 构建优化版图谱
                                          optimized=True, use_cache=True)  # 启用缓存
    LOGGER.info("开始构建知识图谱（优化前基线）")                  # 提示
    graph_base = graph_builder.build_graph(chunks, retriever.vectors, questions,  # 构建基线版图谱
                                           optimized=False, use_cache=True)  # 启用缓存
    pipeline = GraphRAGPipeline(retriever, graph_opt, graph_base)  # 组装问答管线

    records = []                                                # 评估记录列表
    for qa in questions:                                        # 遍历每个验收问题
        for mode in ("baseline", "optimized"):                  # 先跑基线再跑优化后，便于同机对比
            LOGGER.info("评估中：[%s] Q%d", mode, qa["id"])       # 输出进度日志
            try:                                                # 单题失败不影响整体
                records.append(evaluate_one(pipeline, qa, mode))  # 执行评估并收集记录
            except Exception as e:                              # 捕获异常
                LOGGER.error("Q%d [%s] 评估失败：%r", qa["id"], mode, e)  # 记录错误
        llm.save_cache()                                        # 每题结束落盘 LLM 缓存，避免重复计费

    summary = report.summarize(records)                         # 汇总两种模式的指标
    meta = {"n_chunks": len(chunks),                             # 语料块数
            "n_sources": len(set(c["source"] for c in chunks)),  # 来源文件数
            "n_questions": len(questions),                       # 问题数
            "graph_opt": {"entities": len(graph_opt.get("entities", {})),  # 优化版实体数
                          "relations": len(graph_opt.get("relations", []))},  # 优化版关系数
            "graph_base": {"entities": len(graph_base.get("entities", {})),  # 基线版实体数
                           "relations": len(graph_base.get("relations", []))},  # 基线版关系数
            "elapsed_seconds": round(time.time() - t0, 2)}       # 评估总耗时
    payload = {"meta": meta, "summary": summary,                 # 组装 JSON 报告主体
               "records": records}                              # 附上逐题明细
    json_path = report.save_json("evaluation_report", payload)   # 保存 JSON 报告
    md = report.render_markdown(summary, records, meta)          # 渲染 Markdown 报告
    md_path = report.save_markdown("evaluation_report", md)      # 保存 Markdown 报告
    LOGGER.info("评估完成，耗时 %.1fs", time.time() - t0)         # 输出完成日志
    LOGGER.info("JSON 报告：%s", json_path)                      # 输出 JSON 路径
    LOGGER.info("Markdown 报告：%s", md_path)                    # 输出 Markdown 路径
    print("\n" + md)                                            # 在控制台直接打印报告便于查看
    return 0                                                    # 正常退出


if __name__ == "__main__":                                      # 仅在直接运行时执行
    sys.exit(main())                                            # 以主函数返回值作为退出码
