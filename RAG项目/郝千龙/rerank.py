# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8，保证文件内中文字符串与注释正常解析
"""可选 BGE-rerank；未启用时按混合得分截断。"""  # 模块文档字符串：说明重排为可选项，未启用则沿用混合得分排序
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，使 list[int] | None 等注解在低版本 Python 也可用

from config import RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_K  # 从全局配置读取：是否启用重排、模型名、返回条数 top_k
from logger import log  # 导入项目统一日志器，用于记录模型加载与降级告警

_reranker = None  # 模型单例占位：懒加载，避免模块导入即加载大模型


def rerank(query: str, documents: list[str], top_k: int | None = None) -> list[int]:
    """返回 documents 的下标，按相关性从高到低。"""  # 文档字符串：返回的是下标列表，调用方据此重排原文
    k = top_k or RERANK_TOP_K  # 优先用调用方传入的 top_k，否则取配置默认值
    if not documents:  # 空文档列表直接返回空结果，避免后续计算报错
        return []
    if not RERANK_ENABLED:  # 未启用重排：跳过模型调用以节省开销
        # 未启用重排：保持原序（混合检索已按分数排序），只截断到 top_k
        return list(range(min(k, len(documents))))  # 返回前 k 个下标并保持原序（混合检索本身已按得分排序）
    global _reranker  # 声明使用全局 _reranker，便于首次加载时为其赋值
    try:
        from FlagEmbedding import FlagReranker  # 延迟导入：仅启用重排时才加载依赖

        if _reranker is None:  # 懒加载：首次调用才加载模型，避免启动即加载大模型
            log.info("loading rerank model %s", RERANK_MODEL)  # 记录加载日志，便于排查耗时
            _reranker = FlagReranker(RERANK_MODEL, use_fp16=False)  # 实例化交叉编码器；use_fp16=False 保证在 CPU/兼容环境下的精度与稳定性
        # 交叉编码器逐对打分：[query, doc] → 相关性分数
        pairs = [[query, doc] for doc in documents]  # 构造 [query, doc] 文本对列表，供 Cross-Encoder 拼接送入模型（区别于双塔的分别编码）
        scores = _reranker.compute_score(pairs, normalize=True)  # 计算每对相关性分数；normalize=True 用 sigmoid 归一化到 0~1，便于跨场景比较与阈值过滤
        if isinstance(scores, float):  # 当只有一对时 compute_score 返回标量 float，需统一为列表以支持下标访问
            scores = [scores]  # 包装成单元素列表，保证后续 scores[i] 不报错
        ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)  # 按分数降序得到下标排序（reverse=True 表示从高到低）
        return ranked[:k]  # 截断到 top_k，返回最相关的前 k 个下标
    except Exception as exc:
        # 模型加载失败等异常：降级为原序截断，保证主流程可用
        log.warning("rerank unavailable: %s", exc)  # 记录告警，便于定位依赖缺失或模型问题
        return list(range(min(k, len(documents))))  # 降级：保持原序截断，保证 RAG 主流程继续可用


# =====================================================================
# 知识点说明（Transformer / 微调 / RAG）
# ---------------------------------------------------------------------
# 1. Transformer：BGE-reranker 是交叉编码器（Cross-Encoder），把
#    [Query, Doc] 拼成一个序列送入 Transformer，用打分头输出相关性分数，
#    能捕捉词级交互，精度高于双塔，但必须逐对计算、较慢。
# 2. 与双塔的区别：Bi-Encoder（见 embeddings.py）可离线建索引、在线快；
#    Cross-Encoder 不能预建索引、精度高。所以工程上"先粗排后精排"：
#    向量/BM25 先召回 top_k 的数倍候选，再用 rerank 精排出 top_k。
# 3. 微调：reranker 常用 pointwise 交叉熵损失微调，数据形如
#    (query, positive_doc, negative_doc)；同样可套 LoRA/PEFT。
# 4. 在 RAG 中的位置：属于"精排（重排序）"环节，把最相关的资料排到
#    提示词前面，可显著提升生成质量（上下文精确率）。
# =====================================================================
