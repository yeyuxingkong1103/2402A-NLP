# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：封装向量化能力（m3e-base），模型以只读卷挂载；未挂载时自动降级为纯 BM25 检索

import os                                                       # 导入 os 用于模型目录探测
import numpy as np                                              # 导入 numpy 用于向量类型与矩阵运算
import config                                                   # 导入本工单配置模块
from logging_setup import LOGGER                                # 导入统一日志对象

_MODEL = None                                                   # 模型单例
_LOAD_FAILED = False                                            # 加载失败标记，避免反复重试拉长启动时间


def available():                                                # 定义模型可用性判断函数
    """判断模型目录是否已挂载/存在，供启动阶段决定是否启用向量检索。"""
    ok = config.EMBED_ENABLED and os.path.isdir(config.MODEL_PATH)  # 开关开启且目录存在
    if not ok and config.EMBED_ENABLED:                         # 开关开启但目录缺失
        LOGGER.warning("未找到嵌入模型目录 %s，本次降级为 BM25 检索", config.MODEL_PATH)  # 输出告警
    return ok                                                   # 返回可用性


def get_model():                                                # 定义模型获取函数
    """懒加载 SentenceTransformer 模型；失败时返回 None（由调用方降级）。"""
    global _MODEL, _LOAD_FAILED                                 # 声明模块级变量
    if _MODEL is not None or _LOAD_FAILED:                      # 已加载或已失败则直接返回
        return _MODEL                                           # 返回当前状态
    try:                                                        # 尝试加载
        import torch                                            # 延迟导入 torch
        from sentence_transformers import SentenceTransformer   # 延迟导入模型库
        torch.set_num_threads(max(1, (os.cpu_count() or 4)))    # 使用全部可用核心加速 CPU 编码
        _MODEL = SentenceTransformer(config.MODEL_PATH)         # 从挂载目录加载本地模型
        _MODEL.max_seq_length = 512                             # 保持原生最大长度
        LOGGER.info("嵌入模型加载完成：%s", config.MODEL_PATH)   # 输出成功日志
    except Exception as e:                                      # 加载失败
        _LOAD_FAILED = True                                     # 记录失败标记
        LOGGER.warning("嵌入模型加载失败，降级为 BM25 检索：%r", e)  # 输出告警
    return _MODEL                                               # 返回模型或 None


def encode(texts, batch_size=32):                               # 定义批量编码函数
    """编码文本为 L2 归一化向量；模型不可用时抛出异常由上层降级。"""
    model = get_model()                                         # 获取模型
    if model is None:                                           # 模型不可用
        raise RuntimeError("嵌入模型不可用")                      # 抛出异常交由上层处理
    vecs = model.encode(texts, batch_size=batch_size,           # 执行编码
                        normalize_embeddings=True,              # 归一化使点积等于余弦相似度
                        convert_to_numpy=True,                  # 返回 numpy 数组
                        show_progress_bar=False)                # 服务端不显示进度条
    return vecs.astype(np.float32)                              # 统一为 float32


def encode_query(text):                                         # 定义查询编码函数
    """编码单条查询，返回一维向量。"""
    return encode([text])[0]                                    # 复用批量编码取首行
