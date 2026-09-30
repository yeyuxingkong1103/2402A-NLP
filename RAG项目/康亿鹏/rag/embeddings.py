"""BGE-M3 Embedding 工厂：基于 HuggingFaceEmbeddings（sentence-transformers 加载本地模型，dense 向量，1024 维）。"""  # 模块说明
import logging  # 运行日志
import time  # 计时（统计模型加载耗时）

from langchain_community.embeddings import HuggingFaceEmbeddings  # LangChain 框架封装（底层 sentence-transformers）

import config  # 全局配置

logger = logging.getLogger("rag.embedding")  # 本模块日志器


def get_bge_m3_embeddings(model_name=None, device=None, batch_size=16):  # 工厂函数：创建一个 BGE-M3 嵌入实例
    """创建本地 BGE-M3 Embedding（dense 输出，1024 维，向量归一化配合 COSINE）。"""
    model_name = model_name or config.bge_m3_model  # 未传模型路径则用配置中的默认值
    device = device or config.device or None  # 设备优先级：显式传参 > 配置项 > None（自动检测 cuda/cpu）
    model_kwargs = {}  # 模型加载参数
    if device:  # 仅在确定设备时传入
        model_kwargs["device"] = device  # 否则留空，交给 sentence-transformers 自动选择
    logger.info("加载 BGE-M3 模型 model=%s device=%s", model_name, device or "auto")  # 加载很慢，先记一条便于定位首问延迟
    started = time.perf_counter()  # 计时起点
    embeddings = HuggingFaceEmbeddings(  # 返回 LangChain 标准 Embedding 实例，可直接交给 Milvus
        model_name=model_name,  # 本地模型目录
        model_kwargs=model_kwargs,  # 设备参数
        encode_kwargs={  # 编码参数
            "batch_size": batch_size,  # 批次大小
            "normalize_embeddings": True,  # 归一化向量，配合 COSINE 度量
        },
    )
    logger.info("BGE-M3 加载完成 耗时=%.2fs", time.perf_counter() - started)  # 加载耗时
    return embeddings
