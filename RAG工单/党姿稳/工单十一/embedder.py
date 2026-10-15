# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：嵌入模型的加载与编码（工单步骤 3「数据集与模型加载」的核心），并缓存语料向量

import os                                                       # 导入 os 用于缓存路径处理
import numpy as np                                              # 导入 numpy 用于向量矩阵运算
import config                                                   # 导入本工单配置模块

_MODEL_CACHE = {}                                               # 已加载模型缓存：路径 -> 模型对象


def set_threads():                                              # 定义线程数设置函数
    """限制 PyTorch 线程数：实测超过 8 后调度开销反超收益，反而变慢。"""
    import torch                                                # 延迟导入 torch（避免无谓启动开销）
    torch.set_num_threads(config.EMBED_THREADS)                 # 设置 CPU 编码线程数


def load_model(path=None):                                      # 定义模型加载函数
    """按路径加载 SentenceTransformer 模型，同一路径只加载一次。"""
    from sentence_transformers import SentenceTransformer       # 延迟导入，加速无关脚本启动
    path = path or config.EMBED_MODEL_PATH                      # 未指定时使用基础模型路径
    if path not in _MODEL_CACHE:                                # 尚未加载该路径的模型
        set_threads()                                           # 先设置线程数
        print("[信息] 加载嵌入模型：%s" % path)                  # 输出加载提示
        _MODEL_CACHE[path] = SentenceTransformer(path)          # 加载并缓存模型对象
    return _MODEL_CACHE[path]                                   # 返回模型对象


def encode(model, texts, batch_size=32, is_query=False, show=False):  # 定义编码函数
    """把文本编码为归一化向量；is_query=True 时加 m3e 官方推荐的查询前缀。"""
    if not texts:                                               # 空输入直接返回空矩阵
        return np.zeros((0, config.EMBED_DIM), dtype="float32")  # 返回形状正确的空矩阵
    if is_query:                                                # 查询侧需要加前缀
        texts = [config.EMBED_QUERY_PREFIX + t for t in texts]  # 逐条加上官方推荐前缀
    vecs = model.encode(texts,                                   # 调用模型编码
                        batch_size=batch_size,                   # 批大小
                        normalize_embeddings=True,               # 归一化，使点积等价余弦相似度
                        show_progress_bar=show,                  # 是否显示进度条
                        convert_to_numpy=True)                   # 返回 numpy 数组
    return np.asarray(vecs, dtype="float32")                    # 统一为 float32 后返回


def encode_chunks(model, chunks, cache_path, batch_size=64):    # 定义语料向量编码函数
    """编码全部语料块并缓存到磁盘；缓存存在且维度匹配时直接复用。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    if os.path.isfile(cache_path):                              # 缓存文件已存在
        try:                                                    # 尝试加载缓存
            vecs = np.load(cache_path)                          # 读取向量矩阵
            if vecs.shape[0] == len(chunks):                    # 行数与当前语料块数一致
                print("[信息] 命中向量缓存：%s %s" % (os.path.basename(cache_path),  # 输出命中提示
                                                    str(vecs.shape)))  # 附带形状信息
                return vecs                                     # 直接复用缓存
            print("[警告] 向量缓存行数不匹配（%d != %d），将重新编码"  # 行数不一致说明语料变了
                  % (vecs.shape[0], len(chunks)))               # 输出告警
        except Exception as e:                                  # 缓存损坏时容错
            print("[警告] 向量缓存读取失败，将重新编码：%r" % e)   # 输出告警
    print("[信息] 开始编码 %d 个语料块（CPU，请耐心等待）" % len(chunks))  # 输出开始提示
    vecs = encode(model, [c["text"] for c in chunks],           # 逐块编码
                  batch_size=batch_size, show=True)             # 显示进度条便于观察
    np.save(cache_path, vecs)                                   # 写入缓存供后续复用
    print("[信息] 向量编码完成并缓存：%s %s" % (os.path.basename(cache_path), str(vecs.shape)))  # 输出完成提示
    return vecs                                                 # 返回向量矩阵
