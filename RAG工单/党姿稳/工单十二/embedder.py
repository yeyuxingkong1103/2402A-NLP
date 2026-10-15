# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：加载本地 m3e-base 嵌入模型，提供文本编码与语料块向量缓存能力

import os                                                       # 导入 os 用于缓存文件判断
import numpy as np                                              # 导入 numpy 用于向量矩阵处理
import torch                                                    # 导入 torch 用于设置线程数
import config                                                   # 导入本工单配置模块
from logger import LOGGER                                       # 导入统一日志对象

_MODEL = None                                                   # 模型单例占位（懒加载，避免重复加载）


def load_model(path=None):                                      # 定义模型加载函数
    """加载本地离线嵌入模型（模块级缓存，同一路径只加载一次）。"""
    global _MODEL                                               # 声明使用模块级变量
    path = path or config.EMBED_MODEL_PATH                      # 未指定路径时使用配置中的默认模型
    if _MODEL is None:                                          # 首次调用时才加载
        from sentence_transformers import SentenceTransformer   # 延迟导入，避免模块导入即占用内存
        torch.set_num_threads(config.EMBED_THREADS)             # 限制 CPU 线程数，避免调度开销反超收益
        _MODEL = SentenceTransformer(path)                      # 从本地目录加载模型（离线）
        _MODEL.max_seq_length = 256                             # 截断到 256，匹配语料块长度
        LOGGER.info("加载嵌入模型：%s", path)                    # 输出加载路径
    return _MODEL                                               # 返回模型实例


def encode(model, texts, batch_size=32, is_query=False, show=False):  # 定义文本编码函数
    """把文本编码为归一化向量；查询侧按 m3e 官方约定添加前缀以对齐训练分布。"""
    if is_query:                                                # 查询侧需要加前缀
        texts = [config.EMBED_QUERY_PREFIX + t for t in texts]  # 逐条加上官方推荐前缀
    vecs = model.encode(texts,                                  # 调用模型编码
                        batch_size=batch_size,                  # 指定批大小
                        normalize_embeddings=True,              # 归一化，使点积等价于余弦相似度
                        show_progress_bar=show)                 # 是否显示进度条
    return np.asarray(vecs, dtype=np.float32)                   # 统一转为 float32 矩阵返回


def encode_chunks(model, chunks, cache_path, batch_size=64, force=False):  # 定义语料块编码函数
    """编码全部语料块并按行号缓存；缓存存在时直接读取，避免重复计算。"""
    if not force and os.path.isfile(cache_path):                # 缓存存在且非强制重算
        vecs = np.load(cache_path)                              # 读取向量矩阵
        if vecs.shape[0] == len(chunks):                        # 行数与块数一致才可用
            LOGGER.info("命中向量缓存：%s %s", os.path.basename(cache_path), str(vecs.shape))  # 输出命中信息
            return vecs                                         # 返回缓存的向量矩阵
        LOGGER.warning("向量缓存行数 %d 与块数 %d 不一致，重建", vecs.shape[0], len(chunks))  # 输出告警
    LOGGER.info("开始编码 %d 个语料块（CPU，请耐心等待）", len(chunks))  # 输出开始提示
    texts = [c["text"] for c in chunks]                         # 取出全部块正文
    vecs = encode(model, texts, batch_size=batch_size, show=True)  # 批量编码（文档侧不加前缀）
    config.ensure_dirs()                                        # 确保数据目录存在
    np.save(cache_path, vecs)                                   # 写入缓存供后续复用
    LOGGER.info("向量编码完成并缓存：%s %s", os.path.basename(cache_path), str(vecs.shape))  # 输出完成提示
    return vecs                                                 # 返回向量矩阵
