# -*- coding: utf-8 -*-
"""BM25 稀疏向量编码：客户端分词统计，产出 Milvus 稀疏向量（{词索引: 权重}）。"""
import math
# 解析：数学模块（log 计算 IDF）


def _default_tokenizer(text: str) -> list[str]:
    # 解析：默认分词器
    import jieba  # 延迟导入：单测可注入自定义分词器
    # 解析：jieba 延迟导入（单测注入假分词器时无需加载词典）

    return list(jieba.lcut(text))
    # 解析：精确模式分词并转列表（中文按词切分）


class BM25Encoder:
    """先 fit 语料建立词表/IDF，再编码文档或查询（查询须与语料用同一词表）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        # 解析：构造——BM25 超参与分词器
        self.k1 = k1
        # 解析：词频饱和度参数（标准 BM25 取值 1.5）
        self.b = b
        # 解析：文档长度归一化参数（标准取值 0.75）
        self.tokenizer = tokenizer or _default_tokenizer
        # 解析：分词器（默认 jieba，测试可注入）
        self.vocab: dict[str, int] = {}
        # 解析：词表（词 → 向量索引）
        self.idf: dict[str, float] = {}
        # 解析：逆文档频率（词 → IDF 值）
        self.doc_len: list[int] = []
        # 解析：各文档词数（算平均长度用）
        self.avg_len: float = 0.0
        # 解析：语料平均文档长度

    def fit(self, texts: list[str]) -> None:
        """重建词表与 IDF（语料变更后必须重新 fit）。"""
        tokenized = [self.tokenizer(t) for t in texts]
        # 解析：全部文档分词
        doc_freq: dict[str, int] = {}
        # 解析：文档频率统计（词 → 出现文档数）
        self.doc_len = []
        # 解析：重置文档长度列表
        for tokens in tokenized:
            # 解析：逐文档统计
            self.doc_len.append(len(tokens))
            # 解析：记录该文档词数
            for term in set(tokens):
                # 解析：去重后逐词统计（同一文档内重复词只计一次文档频率）
                doc_freq[term] = doc_freq.get(term, 0) + 1
                # 解析：文档频率 +1

        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        # 解析：按字母序给每个词分配索引（稳定可复现）
        n = max(len(texts), 1)
        # 解析：文档总数（至少 1，防除零）
        self.avg_len = sum(self.doc_len) / n if self.doc_len else 0.0
        # 解析：平均文档长度
        self.idf = {
            # 解析：逐词计算 IDF
            term: math.log((n - df + 0.5) / (df + 0.5)) + 1.0
            # 解析：BM25 标准 IDF 公式（+1 保证非负）
            for term, df in doc_freq.items()
            # 解析：遍历全部词
        }

    def _encode_row(self, tokens: list[str], doc_length: int) -> dict[int, float]:
        # 解析：编码单个文本为稀疏向量行
        term_freq: dict[str, int] = {}
        # 解析：词频统计
        for term in tokens:
            # 解析：逐词计数
            term_freq[term] = term_freq.get(term, 0) + 1
            # 解析：词频 +1

        row: dict[int, float] = {}
        # 解析：稀疏向量行（词索引 → 权重）
        for term, tf in term_freq.items():
            # 解析：逐词计算权重
            idx = self.vocab.get(term)
            # 解析：查词表索引
            if idx is None:
                # 解析：词不在语料词表
                continue  # 未登录词丢弃
                # 解析：跳过（查询里的新词无法参与匹配）
            norm = tf * (self.k1 + 1) / (
                # 解析：BM25 词频归一化公式
                tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_len, 1e-9))
                # 解析：长度归一化（max 防除零）
            )
            weight = self.idf[term] * norm
            # 解析：最终权重 = IDF × 归一化词频
            if weight > 0:
                # 解析：权重为正才写入
                row[idx] = weight
                # 解析：写入稀疏向量
        return row
        # 解析：返回该文本的稀疏向量

    def encode_texts(self, texts: list[str]) -> list[dict[int, float]]:
        """编码语料文档，返回按词表索引的稀疏向量行。"""
        return [
            # 解析：逐文档编码
            self._encode_row(self.tokenizer(t), len(self.tokenizer(t)))
            # 解析：分词后编码（分词调两次——先分词算长度再编码，简单可靠）
            for t in texts
            # 解析：遍历全部文档
        ]

    def encode_query(self, query: str) -> dict[int, float]:
        """编码查询（须先 fit 语料）。"""
        tokens = self.tokenizer(query)
        # 解析：查询分词
        return self._encode_row(tokens, len(tokens))
        # 解析：按相同词表编码——保证查询与文档索引空间一致
