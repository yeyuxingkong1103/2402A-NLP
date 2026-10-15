# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：加载检索所需的共享资源（语料、向量、BM25、图谱词典、嵌入模型）与冷启动预热

import json                                                     # 导入 json 用于读取语料与图谱
import time                                                     # 导入 time 用于统计加载与预热耗时
import numpy as np                                              # 导入 numpy 用于向量矩阵运算
import config                                                   # 导入本工单配置模块
import bm25 as bm25_mod                                         # 导入 BM25 检索模块
from logger import LOGGER                                       # 导入统一日志对象

_MODEL = None                                                   # 嵌入模型单例（避免重复加载）
_EPS = 1e-9                                                     # 防止除零的极小量


def load_chunks():                                              # 定义语料加载函数
    """读取工单十二产出的语料块缓存（1950 块），返回块列表。"""
    with open(config.CHUNK_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开语料缓存
        return json.load(f)                                     # 反序列化语料块


def load_model():                                               # 定义嵌入模型加载函数
    """按本地路径加载 m3e-base 嵌入模型（单例），返回模型对象。"""
    global _MODEL                                               # 声明使用模块级单例
    if _MODEL is None:                                          # 尚未加载时执行加载
        t0 = time.perf_counter()                                # 记录加载起始时刻
        import torch                                            # 延迟导入 torch，缩短模块导入时间
        from sentence_transformers import SentenceTransformer   # 延迟导入句向量模型类
        torch.set_num_threads(config.EMBED_THREADS)             # 限制 CPU 线程数，避免线程争抢
        model = SentenceTransformer(config.EMBED_MODEL_PATH)    # 按配置路径加载本地模型
        model.max_seq_length = 256                              # 限制最大序列长度，控制编码耗时
        _MODEL = model                                          # 记录单例
        LOGGER.info("嵌入模型加载完成，耗时 %.2f 秒", time.perf_counter() - t0)  # 输出加载耗时
    return _MODEL                                               # 返回模型单例


def load_vectors():                                             # 定义向量加载函数
    """读取语料块向量矩阵（1950×768，float32，未归一化）。"""
    vecs = np.load(config.EMBED_CACHE_FILE).astype(np.float32)  # 读取并转为 float32 节省内存带宽
    LOGGER.info("语料向量加载完成：%s", str(vecs.shape))          # 输出向量规模
    return vecs                                                 # 返回向量矩阵


def load_bm25():                                                # 定义 BM25 加载函数
    """读取工单十二产出的 BM25 倒排索引。"""
    idx = bm25_mod.BM25.load(config.BM25_CACHE_FILE)            # 反序列化 BM25 索引
    LOGGER.info("BM25 索引加载完成：%d 文档", idx.n_docs)         # 输出文档总数
    return idx                                                  # 返回索引对象


def load_entity_names():                                        # 定义图谱实体词典加载函数
    """读取知识图谱中的实体名，构建查询理解的离线词典（替代在线 LLM 关键词抽取）。"""
    with open(config.GRAPH_FILE, "r", encoding="utf-8") as f:   # 以 UTF-8 打开图谱
        graph = json.load(f)                                    # 反序列化图谱
    names = [n for n in graph["entities"] if len(n) >= 2]       # 取长度不少于 2 的实体名
    names.sort(key=len, reverse=True)                           # 按长度降序，保证最长匹配优先
    LOGGER.info("图谱实体词典加载完成：%d 条", len(names))        # 输出词典规模
    return names                                                # 返回实体名列表


def normalize_matrix(vecs):                                     # 定义向量矩阵预归一化函数
    """把向量矩阵按行做 L2 归一化，返回连续内存的 float32 矩阵，供余弦相似度直接点积。"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)         # 逐行计算 L2 范数
    out = vecs / np.maximum(norms, _EPS)                        # 归一化（避免除零）
    return np.ascontiguousarray(out, dtype=np.float32)          # 转为连续内存并统一为 float32


def warm_up(model, entity_names):                               # 定义冷启动预热函数
    """预热嵌入模型编码路径与 jieba 词典，把冷启动开销从首个真实请求中移出。"""
    t0 = time.perf_counter()                                    # 记录预热起始时刻
    model.encode(["预热"], batch_size=1, show_progress_bar=False)  # 触发模型前向与缓存初始化
    bm25_mod.tokenize("预热查询")                                # 触发 jieba 词典加载
    _ = entity_names[:1]                                        # 触碰实体词典（触发列表驻留）
    LOGGER.info("冷启动预热完成，耗时 %.3f 秒", time.perf_counter() - t0)  # 输出预热耗时
