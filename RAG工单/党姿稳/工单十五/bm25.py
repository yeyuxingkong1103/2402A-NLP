# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：中文分词与 BM25 全文检索实现，与向量检索构成混合召回的两路之一。

import math                                                      # IDF 对数运算
import re                                                        # 标点清洗
import jieba                                                     # 中文分词
from logger import get_logger                                    # 日志

jieba.setLogLevel(20)                                            # 关闭 jieba 调试日志

logger = get_logger("rag15.bm25")

# 标点与空白：统一替换为空格，避免标点粘连影响分词
_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》〈〉【】·,.;:?!\"'()\[\]{}<>|/\\~`@#$%^&*+=_-]+")


def tokenize(text):
    """中英混合分词：英文数字保留原形，中文用 jieba 切分，过滤标点与单字噪音。"""
    text = text.lower()                                          # 英文统一小写
    text = _PUNCT.sub(" ", text)                                 # 标点转空格
    tokens = []                                                  # 词元结果
    for tok in jieba.lcut(text):                                 # jieba 精确模式
        tok = tok.strip()                                        # 去空白
        if not tok:                                              # 空词跳过
            continue
        if len(tok) == 1 and not re.match(r"[a-z0-9]", tok):      # 过滤单个汉字（信息量过低）
            continue
        tokens.append(tok)                                       # 保留
    return tokens


class BM25(object):
    """标准 BM25（Okapi）：构建倒排索引，返回 top-k 结果。"""

    def __init__(self, k1=1.5, b=0.75):                           # 构造
        self.k1 = k1                                             # 词频饱和参数
        self.b = b                                               # 长度归一化参数
        self.df = {}                                             # 词 -> 文档频次
        self.doc_len = []                                        # 各文档词元数
        self.avg_len = 0.0                                       # 平均文档长度
        self.n_docs = 0                                          # 文档总数
        self.inverted = {}                                       # 词 -> {文档下标: 词频}

    def fit(self, texts):
        """对文本列表建立倒排索引与统计量。"""
        self.n_docs = len(texts)                                 # 文档总数
        for i, text in enumerate(texts):                         # 逐文档
            toks = tokenize(text)                                # 分词
            self.doc_len.append(len(toks))                       # 记录长度
            seen = set()                                         # 本文档出现过的词
            for t in toks:                                       # 逐词
                self.inverted.setdefault(t, {})                  # 初始化倒排项
                self.inverted[t][i] = self.inverted[t].get(i, 0) + 1  # 累计词频
                seen.add(t)                                      # 记录
            for t in seen:                                       # 文档频次（每文档只计一次）
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0  # 平均长度
        logger.info("BM25 索引构建完成：%d 文档，%d 词", self.n_docs, len(self.df))

    def _idf(self, term):
        """带平滑的 IDF，避免负值。"""
        df = self.df.get(term, 0)                                # 文档频次
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query, top_k=10):
        """返回 [(文档下标, 分数)] 按分数降序。"""
        q_toks = tokenize(query)                                 # 查询分词
        scores = {}                                              # 文档 -> 得分
        for t in q_toks:                                         # 逐查询词
            if t not in self.inverted:                           # 未登录词跳过
                continue
            idf = self._idf(t)                                   # 该词 IDF
            for doc_i, tf in self.inverted[t].items():            # 命中文档
                dl = self.doc_len[doc_i] or 1                     # 文档长度
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                scores[doc_i] = scores.get(doc_i, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])     # 降序排序
        return ranked[:top_k]                                    # 取前 k

    def state(self):
        """导出可序列化状态，供 pickle 缓存。"""
        return {"k1": self.k1, "b": self.b, "df": self.df,       # 参数与词表
                "doc_len": self.doc_len, "avg_len": self.avg_len,
                "n_docs": self.n_docs, "inverted": self.inverted}

    def load_state(self, st):
        """从缓存恢复状态，跳过重复建索引。"""
        self.k1, self.b = st["k1"], st["b"]                      # 恢复参数
        self.df, self.doc_len = st["df"], st["doc_len"]          # 恢复词表与长度
        self.avg_len, self.n_docs = st["avg_len"], st["n_docs"]  # 恢复统计量
        self.inverted = st["inverted"]                           # 恢复倒排
        return self
