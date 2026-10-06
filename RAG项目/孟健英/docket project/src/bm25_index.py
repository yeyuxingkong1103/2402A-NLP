# -*- coding: utf-8 -*-
"""BM25 关键词检索索引。

用 jieba 中文分词 + rank_bm25 实现，与 Milvus 向量检索并列构成混合检索的两路召回。
"""
from typing import List, Tuple  # 类型注解：List 存分词后的词列表，Tuple 存 (下标, 分数) 二元组

import jieba  # jieba 中文分词库：按词切中文句，避免按字切把语义拆散；BM25 计算的是词级别的词频与逆文档频率
from rank_bm25 import BM25Okapi  # Okapi BM25 经典实现：TF-IDF 的概率化升级版，对长文本与短查询更稳健


class BM25Index:
    """BM25 轻量索引（进程内，重启需重建）。"""

    def __init__(self):
        self._bm25 = None  # BM25Okapi 实例：进程内内存对象，服务重启即失效，需要持久化——多数项目干脆在启动时从知识库重建
        self._records = []  # 与 BM25 下标严格一一对应的原始片段列表，检索命中后拿数据用

    def build(self, records: List[dict]) -> None:  # 全量构建：知识库更新后重建索引
        """从知识片段构建索引，records 为 [{text, source, page}, ...]。"""
        self._records = records  # 保存原始记录，保证 search 返回的下标能到这里取回来源与页码
        if not records:  # 空库时索引置空，避免后续 search 报错
            self._bm25 = None
            return
        self._bm25 = BM25Okapi([list(jieba.cut(r["text"])) for r in records])  # 每条记录用 jieba 分词成 list：BM25 按词列表计算；jieba.cut 返回生成器，list() 转成列表——BM25 擅长字面精确匹配（药名/术语），和向量语义召回互补

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:  # 查询：分词后打分，取 Top-K
        """返回 [(记录下标, BM25分数), ...]。"""
        if self._bm25 is None:  # 索引未初始化时直接返回空结果
            return []
        scores = self._bm25.get_scores(list(jieba.cut(query)))  # 查询也走 jieba 分词：分词器必须保持一致，否则词粒度不对齐，BM25 打分失真
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]  # 按分数降序取前 top_k：enumerate 同时拿到下标与分数，避免分数与记录错位

    @property
    def records(self) -> List[dict]:  # 对外暴露原始记录，供检索后取文本与来源
        """原始片段（与检索下标一一对应）。"""
        return self._records

    def add(self, new_records: List[dict]) -> None:  # 增量更新：合并旧数据与新数据后重建
        """增量添加（重建整个索引，轻量场景够用）。"""
        self.build(self._records + new_records)  # 把新记录拼到已有记录末尾再重建；数据量不大时全量重建成本极低，最简单、最可靠