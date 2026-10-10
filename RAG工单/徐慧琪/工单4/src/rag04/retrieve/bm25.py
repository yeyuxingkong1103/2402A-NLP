# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""BM25 稀疏检索。jieba 分词 + rank_bm25，支持持久化。"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from rag04.schema import Chunk

_ASCII = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]*")
_WS = re.compile(r"\s+")

# 招股书领域词典：避免关键实体被切碎
_DOMAIN_WORDS = [
    "销售部", "大客户销售部", "渠道销售部", "电话及网络销售部", "国际贸易部",
    "销售处", "组织结构图", "招股说明书", "招股意向书",
    "武汉力源信息技术股份有限公司", "武汉兴图新科电子股份有限公司",
    "募集资金", "注册资本", "法定代表人", "关联方", "军用领域",
    "主营业务收入", "补充流动资金", "IC市场", "应用结构",
]
for _w in _DOMAIN_WORDS:
    jieba.add_word(_w, freq=10_000)


def tokenize_zh(text: str) -> list[str]:
    """中文分词 + 英文/数字词元保留，统一小写。

    用 jieba 搜索模式而非精确模式：领域词典中的长实体既保留整词（整名精确命中），
    也拆出子词（如「武汉力源信息技术股份有限公司」→ 武汉/力源/信息/技术/…），
    避免长实体只产出一个词元而丢失部分匹配能力。
    """
    if not text:
        return []
    toks: list[str] = []
    for piece in jieba.lcut_for_search(text):
        p = _WS.sub("", piece)
        if not p:
            continue
        if _ASCII.fullmatch(p):
            toks.append(p.lower())
        elif re.search(r"[一-鿿0-9]", p):
            for sub in _ASCII.findall(p):
                toks.append(sub.lower())
            zh = re.sub(r"[^一-鿿0-9]", "", p)
            if zh:
                toks.append(zh)
    return toks


class BM25Index:
    """稀疏索引。空语料时 search 返回空列表而非抛错。"""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._doc_tokens: list[set[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self._bm25 = None
            self._doc_tokens = []
            return
        tokenized = [tokenize_zh(c.text) for c in self.chunks]
        self._doc_tokens = [set(t) for t in tokenized]
        corpus = [t or ["<empty>"] for t in tokenized]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 30,
               exclude_ids: set[str] | frozenset[str] | None = None
               ) -> list[tuple[str, float]]:
        """BM25 检索。``exclude_ids``（RC1 样板集合）在截 top-k **之前**生效。

        必须在截断前排除：页眉块可占满 top-30（实测 id34 为 29/30），
        先截断再排除会把真实候选一并截掉；`get_scores` 本就对全库打分，
        此处过滤是零额外开销的。
        """
        if self._bm25 is None or not self.chunks:
            return []
        toks = tokenize_zh(query)
        if not toks:
            return []
        qset = set(toks)
        scores = self._bm25.get_scores(toks)
        # 命中判定用「块含至少一个查询词元」而非「分数 > 0」：BM25Okapi 的 IDF 在
        # 词项出现于恰好半数文档时为 0，按分数过滤会漏掉真实关键词命中；
        # 分数仍按原始 BM25 值降序排列（可能为 0 或负，属病态 idf 场景）。
        pairs = [(self.chunks[i].chunk_id, float(scores[i]))
                 for i in range(len(self.chunks)) if qset & self._doc_tokens[i]
                 and (exclude_ids is None or self.chunks[i].chunk_id not in exclude_ids)]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self._bm25,
                         "doc_tokens": self._doc_tokens}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        inst = cls()
        inst.chunks = d["chunks"]
        inst._bm25 = d["bm25"]
        inst._doc_tokens = d.get("doc_tokens")
        if not inst._doc_tokens:
            inst._doc_tokens = [set(tokenize_zh(c.text)) for c in inst.chunks]
        return inst
