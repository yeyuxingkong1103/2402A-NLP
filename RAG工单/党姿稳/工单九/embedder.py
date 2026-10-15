# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：加载本地 m3e-base 嵌入模型，把文本块编码为归一化向量并做磁盘缓存

import os                                                       # 导入 os 用于缓存文件判断
import json                                                     # 导入 json 用于缓存元信息读写
import numpy as np                                              # 导入 numpy 用于向量矩阵运算
import config                                                   # 导入本工单配置模块
from timing import LOGGER                                       # 导入统一日志对象

# 延迟导入：sentence_transformers 首次导入较慢，放到函数内按需触发
_MODEL = None                                                   # 嵌入模型单例占位


def get_model():                                                # 定义获取嵌入模型的函数
    """懒加载 m3e-base 模型；单例复用避免重复占用内存与加载时间。"""
    global _MODEL                                               # 声明使用模块级变量
    if _MODEL is None:                                          # 仅首次调用时加载
        import torch                                            # 延迟导入 torch 以控制加载时机
        from sentence_transformers import SentenceTransformer   # 延迟导入以加快无向量需求时的启动
        # 线程数并非越多越快：本机实测 6 条短查询的编码耗时随线程数为
        # 1→1.37s、2→0.85s、4→1.26s、8→2.00s、16→1.45s，用满 16 线程时
        # 线程调度与同步开销反超并行收益。小模型短文本取 4 线程为稳妥值。
        threads = 4                                             # 固定较小线程数，规避 CPU 线程超订
        torch.set_num_threads(threads)                          # 显式设置线程数，避免 Windows 默认值偏大
        LOGGER.info("torch 线程数设置为 %d", threads)             # 输出线程配置日志
        _MODEL = SentenceTransformer(config.EMBED_MODEL_PATH)   # 从本地路径加载模型（无需联网）
        _MODEL.max_seq_length = 512                             # 保持模型原生最大长度，避免截断内容
        LOGGER.info("嵌入模型加载完成：%s", config.EMBED_MODEL_PATH)  # 输出加载成功日志
    return _MODEL                                               # 返回模型实例


def encode_texts(texts, batch_size=32, show_progress=False):    # 定义批量编码函数
    """把文本列表编码为 L2 归一化向量；归一化后内积等价于余弦相似度。"""
    model = get_model()                                         # 获取模型实例
    vecs = model.encode(texts,                                  # 执行编码
                        batch_size=batch_size,                  # 批大小（CPU 下 32 较均衡）
                        show_progress_bar=show_progress,         # 是否显示进度条
                        normalize_embeddings=True,              # 归一化，使后续点积即余弦相似度
                        convert_to_numpy=True)                  # 直接返回 numpy 数组
    return vecs.astype("float32")                               # 统一转为 float32 以节省内存并加速


def build_or_load_embeddings(chunks, use_cache=True):           # 定义带缓存的向量构建入口
    """优先读取 data/embeddings.npy；用块 ID 列表校验一致性，失效则重建。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    vec_path = config.EMBED_CACHE_FILE                          # 向量矩阵缓存路径
    meta_path = config.META_CACHE_FILE                          # 元信息缓存路径
    ids = [c["id"] for c in chunks]                             # 当前块的 ID 序列
    if use_cache and os.path.isfile(vec_path) and os.path.isfile(meta_path):  # 两个缓存文件都存在
        try:                                                    # 尝试读取并校验
            meta = json.load(open(meta_path, "r", encoding="utf-8"))  # 读取缓存元信息
            if meta.get("ids") == ids:                          # 块 ID 序列完全一致说明缓存有效
                vecs = np.load(vec_path)                        # 加载向量矩阵
                if vecs.shape[0] == len(chunks):                # 行数一致才认为可用
                    LOGGER.info("命中向量缓存：%s", vecs.shape)   # 输出缓存命中日志
                    return vecs                                 # 返回缓存的向量矩阵
            LOGGER.info("向量缓存与当前分块不一致，将重建")        # 说明不一致原因
        except Exception as e:                                  # 缓存读取失败时容错
            LOGGER.warning("向量缓存读取失败，将重建：%r", e)      # 输出告警
    LOGGER.info("开始编码 %d 个文本块（首次运行较慢，请耐心等待）", len(chunks))  # 输出开始日志
    vecs = encode_texts([c["text"] for c in chunks], batch_size=32, show_progress=True)  # 批量编码
    try:                                                        # 尝试写入缓存
        np.save(vec_path, vecs)                                 # 保存向量矩阵（.npy）
        json.dump({"ids": ids,                                  # 保存块 ID 序列用于下次校验
                   "dim": int(vecs.shape[1]),                   # 保存向量维度
                   "count": int(vecs.shape[0])},                # 保存向量条数
                  open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)  # 写入元信息文件
        LOGGER.info("向量缓存写入完成：%s", vecs.shape)          # 输出完成日志
    except Exception as e:                                      # 写入失败时容错
        LOGGER.warning("向量缓存写入失败：%r", e)                # 仅告警
    return vecs                                                 # 返回向量矩阵


def encode_query(text):                                         # 定义单条查询编码函数
    """编码用户查询，返回 (1, dim) 的归一化向量。"""
    return encode_texts([text])[0]                              # 复用批量编码并取首行


def warm_up():                                                  # 定义嵌入模型预热函数
    """提前加载模型并跑一次前向，使首个真实查询不承担加载与本轮初始化的开销。

    否则最先执行的那一次检索会额外计入约 10 秒的模型加载时间，
    造成「优化前后检索耗时」的对比失真（首个样本被系统性抬高）。
    """
    get_model().encode(["预热"], normalize_embeddings=True)     # 加载模型并执行一次极短编码
    LOGGER.info("嵌入模型预热完成")                              # 输出预热完成日志
