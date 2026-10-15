# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：一键构建（或加载）检索索引：语料 → 分块 → 向量 → BM25 → 统一检索器
# 使用方式：python build_index.py            # 建立并落盘全部缓存
#          python build_index.py --rebuild   # 忽略缓存，强制重建

import os                                                       # 导入 os 用于缓存文件判断
import sys                                                      # 导入 sys 用于读取命令行参数与路径注册
import time                                                     # 导入 time 用于统计构建耗时

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 把本目录加入模块搜索路径，支持直接运行

import config                                                   # 导入本工单配置模块
import data_loader                                              # 导入语料与问题加载模块
import chunking                                                 # 导入分块模块
import embedder                                                 # 导入向量化模块
from bm25 import BM25                                           # 导入 BM25 检索器
from retriever import Retriever                                 # 导入统一检索器
from timing import LOGGER                                       # 导入统一日志对象


def build_index(use_cache=True):                                # 定义索引构建入口
    """返回 (Retriever, chunks)；优先复用磁盘缓存，缺失时才重新计算。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    t0 = time.time()                                            # 记录整体起始时间
    docs = data_loader.load_corpus()                            # 加载语料页面
    chunks = chunking.build_or_load_chunks(docs, use_cache=use_cache)  # 分块（带缓存）
    vectors = embedder.build_or_load_embeddings(chunks, use_cache=use_cache)  # 向量化（带缓存）

    bm25 = None                                                 # BM25 索引初值
    if use_cache and os.path.isfile(config.BM25_CACHE_FILE):    # 缓存文件存在时尝试加载
        bm25 = BM25.load(config.BM25_CACHE_FILE)                # 从磁盘恢复索引
    if bm25 is None or bm25.n_docs != len(chunks):              # 索引缺失或与当前分块不一致
        bm25 = BM25()                                           # 新建索引
        bm25.fit([c["text"] for c in chunks])                   # 对全部块正文分词建索引
        try:                                                    # 尝试落盘缓存
            bm25.save(config.BM25_CACHE_FILE)                   # 保存索引用以加速下次启动
        except Exception as e:                                  # 保存失败时容错
            LOGGER.warning("BM25 索引缓存写入失败：%r", e)       # 仅告警不影响运行

    retriever = Retriever(chunks, vectors, bm25=bm25)           # 组装统一检索器
    embedder.warm_up()                                          # 预热嵌入模型：由索引构建阶段承担加载耗时而非首次查询
    retriever.bm25_search("预热")                                # 预热 jieba 分词：首次调用需构建前缀词典（实测约 2.8 秒）
    LOGGER.info("索引就绪：%d 块，向量维度 %d，耗时 %.1fs",       # 输出构建完成日志
                len(chunks), vectors.shape[1], time.time() - t0)  # 填入统计信息
    return retriever, chunks                                    # 返回检索器与分块


def main():                                                     # 定义命令行主函数
    rebuild = "--rebuild" in sys.argv                           # 解析是否强制重建
    retriever, chunks = build_index(use_cache=not rebuild)      # 构建或加载索引
    questions = data_loader.load_questions()                    # 加载验收问题
    LOGGER.info("构建结束：chunks=%d, questions=%d", len(chunks), len(questions))  # 输出最终统计


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 调用主函数
