# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：检索层——中文分词、BM25 全文检索、向量检索与 RRF 混合融合

import re                                                       # 导入正则用于标点清洗
import math                                                     # 导入数学库用于 BM25 计算
import jieba                                                    # 导入结巴分词
import numpy as np                                              # 导入 numpy 用于向量点积
import config                                                   # 导入本工单配置模块
from logging_setup import LOGGER                                # 导入统一日志对象

jieba.setLogLevel(20)                                           # 屏蔽 jieba 的调试日志
_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》【】·,.;:?!\"'()\[\]{}<>|/\\~`@#$%^&*+=_-]+")  # 标点与空白


def tokenize(text):                                             # 定义分词函数
    """中英混合分词：小写化、去标点、过滤单字中文噪音。"""
    text = _PUNCT.sub(" ", (text or "").lower())                # 统一小写并把标点替换为空格
    tokens = []                                                 # 词元列表
    for tok in jieba.lcut(text):                                # 使用精确模式切词
        tok = tok.strip()                                       # 去除空白
        if not tok:                                             # 跳过空词元
            continue                                            # 继续
        if len(tok) == 1 and not re.match(r"[a-z0-9]", tok):    # 过滤单个中文字符
            continue                                            # 继续
        tokens.append(tok)                                      # 保留有效词元
    return tokens                                               # 返回词元列表


class BM25(object):                                             # 定义 BM25 检索器
    """标准 Okapi BM25 实现，支持序列化以便缓存到数据卷。"""

    def __init__(self, k1=1.5, b=0.75):                         # 构造方法
        self.k1 = k1                                            # 词频饱和参数
        self.b = b                                              # 长度归一化参数
        self.df = {}                                            # 词 -> 文档频次
        self.doc_len = []                                       # 各文档词元数
        self.avg_len = 0.0                                      # 平均文档长度
        self.n_docs = 0                                         # 文档总数
        self.inverted = {}                                      # 词 -> {文档下标: 词频}

    def fit(self, texts):                                       # 定义索引构建方法
        """对文本列表构建倒排索引与统计量。"""
        self.n_docs = len(texts)                                # 记录文档总数
        for i, text in enumerate(texts):                        # 遍历文档
            toks = tokenize(text)                               # 分词
            self.doc_len.append(len(toks))                      # 记录长度
            for t in set(toks):                                 # 遍历去重词（统计文档频次）
                self.df[t] = self.df.get(t, 0) + 1              # 文档频次加一
            for t in toks:                                      # 遍历全部词（统计词频）
                self.inverted.setdefault(t, {})                 # 初始化倒排项
                self.inverted[t][i] = self.inverted[t].get(i, 0) + 1  # 累加词频
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0  # 计算平均长度
        LOGGER.info("BM25 索引构建完成：%d 文档 / %d 词", self.n_docs, len(self.df))  # 输出日志

    def search(self, query, top_k=None):                        # 定义检索方法
        """返回 [(文档下标, 分数), ...]，按分数降序。"""
        top_k = top_k or config.BM25_TOP_K                      # 取默认条数
        scores = {}                                             # 下标 -> 分数
        for t in tokenize(query):                               # 遍历查询词
            postings = self.inverted.get(t)                     # 取倒排列表
            if not postings:                                    # 词不在词表
                continue                                        # 跳过
            df = self.df.get(t, 0)                              # 文档频次
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))  # 平滑 IDF
            for idx, tf in postings.items():                    # 遍历包含该词的文档
                dl = self.doc_len[idx] or 1                     # 文档长度（防除零）
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1.0))  # BM25 分母
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (self.k1 + 1) / denom  # 累加得分
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 降序排序
        return ranked[:top_k]                                   # 返回前 top_k


class Retriever(object):                                        # 定义统一检索器
    """封装 BM25 与向量两路召回，并支持 RRF 融合。"""

    def __init__(self, chunks, vectors=None, bm25=None):        # 构造方法
        self.chunks = chunks                                    # 文本块列表
        self.vectors = vectors                                  # 归一化向量矩阵（可为 None）
        self.bm25 = bm25 or _fit_bm25(chunks)                   # BM25 索引（可外部注入缓存）
        self.embed_enabled = vectors is not None and config.EMBED_ENABLED  # 是否具备向量检索能力

    def vector_search(self, query_vec, top_k=None):             # 定义向量检索方法
        """按余弦相似度（归一化向量点积）返回 top-k。"""
        top_k = top_k or config.VECTOR_TOP_K                    # 取默认条数
        if self.vectors is None or self.vectors.shape[0] == 0:  # 无向量索引时返回空
            return []                                           # 返回空列表
        scores = self.vectors.dot(query_vec)                    # 点积计算相似度
        k = min(top_k, scores.shape[0])                         # 防止越界
        idx = np.argpartition(-scores, k - 1)[:k]               # 快速取前 k 个候选
        idx = idx[np.argsort(-scores[idx])]                     # 候选按分数降序
        return [(int(i), float(scores[i])) for i in idx]        # 返回 (下标, 分数)

    def retrieve(self, query, query_vec=None, top_k=None, mode="hybrid"):  # 定义检索入口
        """按模式执行检索：bm25 / vector / hybrid，返回文本块列表。"""
        top_k = top_k or config.FINAL_TOP_K                     # 最终条数
        if mode == "bm25" or not self.embed_enabled:            # BM25 模式或未启用向量
            hits = self.bm25.search(query, top_k=top_k)         # 仅 BM25 检索
            return [self.chunks[i] for i, _ in hits if i < len(self.chunks)]  # 映射为块
        if mode == "vector":                                    # 纯向量模式
            hits = self.vector_search(query_vec, top_k=top_k)   # 仅向量检索
            return [self.chunks[i] for i, _ in hits if i < len(self.chunks)]  # 映射为块
        rrf = {}                                                # 混合模式：RRF 融合打分
        for rank, (i, _s) in enumerate(self.bm25.search(query, top_k=config.BM25_TOP_K)):  # 全文召回
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)  # 累加 RRF 贡献
        for rank, (i, _s) in enumerate(self.vector_search(query_vec, top_k=config.VECTOR_TOP_K)):  # 向量召回
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)  # 累加 RRF 贡献
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]  # 融合后取前 top_k
        return [self.chunks[i] for i, _ in ranked if i < len(self.chunks)]  # 映射为块列表


def _fit_bm25(chunks):                                          # 定义 BM25 构建辅助函数
    """对文本块列表构建 BM25 索引。"""
    bm = BM25()                                                 # 实例化
    bm.fit([c["text"] for c in chunks])                         # 构建索引
    return bm                                                   # 返回索引对象
