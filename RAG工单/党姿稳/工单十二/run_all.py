# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：一键串联全部流程——语料分块 → 知识图谱构建 → 双系统检索评估 → 交付物生成

import os                                                       # 导入 os 用于文件存在判断
import sys                                                      # 导入 sys 用于解析命令行参数
import json                                                     # 导入 json 用于读取图谱交给报告模块
import config                                                   # 导入本工单配置模块
import corpus                                                   # 导入语料分块模块
import kg_build                                                 # 导入知识图谱构建模块
import run_eval                                                 # 导入双系统评估模块
import report                                                   # 导入交付物生成模块
from logger import LOGGER                                       # 导入统一日志对象


def main(force=False):                                          # 定义主流程函数
    """按阶段执行完整流水线；force=True 时重建全部缓存（会重新调用 LLM，产生费用）。"""
    config.ensure_dirs()                                        # 确保缓存与报告目录存在
    LOGGER.info("阶段 1/4：PDF 抽页与语料分块")                    # 输出阶段提示
    chunks = corpus.build_or_load_chunks(force=force)           # 抽取并分块（带缓存）
    LOGGER.info("阶段 1/4 完成：%d 块，%d 字", len(chunks), sum(len(c["text"]) for c in chunks))  # 输出规模

    LOGGER.info("阶段 2/4：知识图谱构建（实体关系抽取 + 合并去重）")  # 输出阶段提示
    graph = kg_build.build(force=force)                         # 并发抽取并合并为图谱
    LOGGER.info("阶段 2/4 完成：%d 实体，%d 关系", len(graph["entities"]), len(graph["relations"]))  # 输出规模

    LOGGER.info("阶段 3/4：RAG 与 LightRAG 双系统检索、生成与 RAGAS 评估")  # 输出阶段提示
    result = run_eval.run()                                     # 执行完整评估流程
    LOGGER.info("阶段 3/4 完成：LightRAG 更优指标 %d/%d",
                result["compare"]["improved"], result["compare"]["total"])  # 输出对比结论

    LOGGER.info("阶段 4/4：生成交付物（图谱统计、可视化、检索对比、指标对比）")  # 输出阶段提示
    with open(config.GRAPH_FILE, "r", encoding="utf-8") as f:   # 打开图谱文件
        graph = json.load(f)                                    # 读取图谱（确保为磁盘上的最新版本）
    report.export_all(result, graph)                            # 生成全部交付物
    LOGGER.info("全部流程完成，交付物位于 %s", config.EVAL_DIR)    # 输出完成提示
    return result                                               # 返回评估结果


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main(force="--force" in sys.argv)                           # 支持 --force 强制重建
