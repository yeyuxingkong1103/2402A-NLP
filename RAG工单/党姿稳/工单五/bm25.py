# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
BM25 关键词检索模块（倒排索引实现）。
中文无空格，采用"字符 unigram + bigram"切分作为词项，无需外部分词器。
"""
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')


def tokenize(text):
    """
    中英混合分词：英文/数字按词，中文以 2-gram 为主、1-gram 为辅。
    单个汉字信息量低、噪声大（如"是""哪""个"），故只保留 1-gram 到 lower-idf 场景，
    这里以 2-gram 为词项单位，显著提升关键词检索区分度。
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    zh = re.findall(r'[一-鿿]', text)
    tokens.extend(zh[i] + zh[i + 1] for i in range(len(zh) - 1))  # 双字为主
    return tokens


class BM25:
    """标准 BM25 检索器：倒排索引 + 文档长度归一化"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.inverted = defaultdict(dict)   # term -> {doc_id: tf}
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.inverted[term][i] = tf
        self.n_docs = len(docs)

    def _idf(self, term):
        df = len(self.inverted.get(term, {}))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query, top_n=20):
        """返回 [(doc_id, score), ...] 按分数降序"""
        scores = defaultdict(float)
        for term in tokenize(query):
            if term not in self.inverted:
                continue
            idf = self._idf(term)
            for doc_id, tf in self.inverted[term].items():
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


if __name__ == "__main__":
    corpus = [
        "武汉兴图新科电子股份有限公司注册资本为5,520.00万元",
        "电子信息行业的上游涉及电子元器件制造企业",
        "军用领域的收入占主营业务收入比重",
    ]
    bm = BM25(corpus)
    for did, s in bm.search("注册资本是多少"):
        print(did, round(s, 4), corpus[did][:30])
