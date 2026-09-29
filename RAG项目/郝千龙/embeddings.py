# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8，保证文件内中文字符串与注释正常解析
"""可选 BGE-m3 向量化；未启用时跳过。"""  # 模块文档字符串：概述本模块职责（可选启用，未启用则跳过向量化）
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，使 list[str] | None 等泛型注解在低版本 Python 也可用

from config import EMBEDDING_ENABLED, EMBEDDING_MODEL  # 从全局配置读取：是否启用向量化、所用模型名
from logger import log  # 导入项目统一日志器，便于记录模型加载进度与异常信息

_model = None  # 模型单例占位：懒加载，避免在模块导入时就加载大模型（节省启动时间与内存）


def embedding_dim() -> int:
    """返回向量维度：BGE-m3 为 1024，其他默认 768。"""  # 函数文档字符串：说明返回的向量维度判断规则
    if "m3" in EMBEDDING_MODEL.lower():  # 通过模型名是否包含 "m3" 判断是否为 BGE-m3（大小写不敏感）
        return 1024  # BGE-m3 的稠密向量维度为 1024（与训练配置一致，需与 Milvus 集合维度对齐）
    return 768  # 其他常见 Transformer 编码器（如 BERT-base）默认输出 768 维


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """批量文本 → 归一化向量；未启用或加载失败返回 None（调用方自动降级）。"""  # 文档字符串：说明返回 None 用于触发调用方降级
    global _model  # 声明使用全局 _model，以便在首次调用时为其赋值
    if not EMBEDDING_ENABLED:  # 配置未开启向量化时直接返回 None，跳过模型加载
        return None  # 返回 None 让调用方走降级路径（例如仅用 BM25 检索）
    try:
        from sentence_transformers import SentenceTransformer  # 延迟导入：仅在实际需要时加载依赖，避免无用导入导致启动报错

        if _model is None:  # 懒加载：首次调用才加载模型，避免模块导入即加载大模型
            log.info("loading embedding model %s", EMBEDDING_MODEL)  # 记录加载日志，便于排查启动耗时与模型路径
            _model = SentenceTransformer(EMBEDDING_MODEL)  # 实例化模型（首次会下载/读取权重到内存）
        # normalize=True：归一化后内积即余弦相似度（配合 Milvus COSINE 度量）
        vectors = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # 批量编码并对向量做 L2 归一化；关闭进度条减少日志噪音
        return [v.tolist() for v in vectors]  # 把 numpy 数组逐个转成 Python 列表，便于序列化入库
    except Exception as exc:  # 捕获依赖缺失、模型加载失败等异常
        log.warning("embedding unavailable: %s", exc)  # 记录告警，方便定位问题
        return None  # 返回 None 触发调用方降级，保证 RAG 主流程不中断


# =====================================================================
# 知识点说明（Transformer / 微调 / RAG）
# ---------------------------------------------------------------------
# 1. Transformer：BGE-m3 是基于 XLM-RoBERTa（Transformer 编码器/Encoder）
#    的双向稠密检索模型。Transformer 的核心是自注意力（Self-Attention）
#    与多头注意力（Multi-Head Attention），能把整句编码为上下文相关的向量。
# 2. 双塔结构（Bi-Encoder）：文档离线编码成向量入库（本文件 embed_texts），
#    查询时在线编码 Query，用余弦相似度做近邻检索，速度快、可配合 Milvus。
# 3. 微调：BGE-m3 可用对比学习（InfoNCE）微调——正例为匹配句对，负例用
#    BM25 召回的"难负例"（hard negatives），工具为 FlagEmbedding 的
#    finetune 脚本；也可用 LoRA（PEFT）做参数高效微调，只训练少量低秩
#    附加矩阵，省显存、不易灾难性遗忘。
# 4. 在 RAG 中的位置：属于"索引/召回"环节的向量化层，embed_texts 的
#    质量直接决定向量召回的准确率（召回率上限）。
# =====================================================================
