"""BGE 重排：用 cross-encoder 对候选片段按与问题的相关性重新打分排序，取 top_n。"""  # 模块说明
import logging  # 运行日志
import time  # 计时（统计模型加载耗时）
from typing import Any, Dict, List, Sequence  # 类型注解

from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # 交叉编码器封装（底层 sentence-transformers.CrossEncoder）
from langchain_core.documents import Document  # LangChain 文档对象

import config  # 全局配置

logger = logging.getLogger("rag.reranker")  # 本模块日志器

# 模型进程内缓存，避免重复加载
model_cache: Dict[str, Any] = {}  # 模型名 -> 已加载的模型实例


def get_reranker() -> HuggingFaceCrossEncoder:  # 懒加载 BGE cross-encoder（进程内单例）
    """首次调用时加载本地 BGE 重排模型，之后直接复用缓存。"""
    model_name = config.bge_reranker_model  # 重排模型本地路径
    if model_name not in model_cache:
        device = config.device or None  # 空串转 None -> 自动检测设备
        model_kwargs: Dict[str, Any] = {"max_length": 512}  # bge-reranker-large 基于 XLM-RoBERTa，上限 512
        if device:  # 仅在显式指定设备时传入
            model_kwargs["device"] = device
        logger.info("加载 BGE 重排模型 model=%s device=%s", model_name, device or "auto")  # 加载较慢，先记一条
        started = time.perf_counter()  # 计时起点
        model_cache[model_name] = HuggingFaceCrossEncoder(  # 加载模型并放入缓存
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        logger.info("BGE 重排模型加载完成 耗时=%.2fs", time.perf_counter() - started)  # 加载耗时
    return model_cache[model_name]


def rerank_documents(  # 普通重排函数：候选文档 -> 按相关性重排 -> top_n
    documents: Sequence[Document],  # 待重排的候选文档（混合召回结果）
    query: str,  # 用户查询
    top_n: int = config.rerank_top_n,  # 保留条数
) -> List[Document]:
    """用 BGE cross-encoder 给每个 (查询, 文档) 对打分，返回分数最高的 top_n 条。"""
    if not documents:  # 空输入直接返回
        return []
    pairs = [[query, doc.page_content] for doc in documents]  # 构造打分对列表
    scores = get_reranker().score(pairs)  # 批量计算每对的相关性分数
    if isinstance(scores, (int, float)):  # 只有一条时可能返回标量
        scores = [scores]
    scores = [float(s) for s in scores]  # 兼容 numpy 数组，统一转 Python float
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)  # 分数从高到低排序 ranked=[(Document, score)]
    results: List[Document] = []
    for doc, score in ranked[:top_n]:  # 只取分数最高的 top_n 条
        doc.metadata = {**doc.metadata, "rerank_score": score}  # 得分写进元数据供展示/排查
        results.append(doc)
    return results
