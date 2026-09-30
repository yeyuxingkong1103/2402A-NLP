# -*- coding: utf-8 -*-
"""retrieval/corpus.py —— 语料加载与 BM25 索引缓存。

在链路中的位置：
    retrieval 包的内存数据层：两路召回都依赖它把 Milvus 里的全量 chunk 读进内存，
    供 BM25 打分与"下标 <-> 记录"的互相映射使用。

为什么必须缓存：
    BM25 打分要遍历整个语料，每次提问都重新拉全量数据并重新分词建索引，
    响应时间无法接受。所以进程启动后第一次检索时建一次索引。

缓存何时失效：
    上传/删除文档后由 backend/server.py 调 invalidate_cache() 主动清空 ——
    不这么做，新入库的内容在检索侧"看不见"。
"""
from __future__ import annotations

import threading
from typing import Any

try:
    from ..pipeline import COLLECTION
    from ..vector_store import query_vectors
except ImportError:
    from pipeline import COLLECTION
    from vector_store import query_vectors

from .bm25 import BM25

# 语料与 BM25 索引的进程内缓存。
# 为什么需要缓存：BM25 打分要遍历整个语料，每次提问都重新从 Milvus 拉全量数据
# 再重新分词建索引，响应时间无法接受。所以进程启动后第一次检索时建一次索引。
# 数据变了怎么办：上传/删除文档后由 server.py 调 invalidate_cache() 主动失效。
_cache_lock = threading.Lock()  # RLock 的替代：这里用普通 Lock，配合双检锁避免重复建索引
_corpus: list[dict[str, Any]] | None = None
_index: dict[tuple[Any, ...], int] | None = None
_bm25: "BM25" | None = None

def invalidate_cache() -> None:
    """使语料与 BM25 索引缓存失效，下次检索时重建。

    调用时机：上传新文档、删除文档之后（由 server.py 触发）。
    不这么做的话，新入库的内容在对端进程里"看不见" —— 用户会以为上传失败。
    """
    global _corpus, _index, _bm25
    with _cache_lock:
        _corpus = _index = _bm25 = None

def load_corpus() -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], int]]:
    """加载全量语料并建立"内容特征 -> 下标"的映射。

    返回：
        (corpus, index_map)
        corpus:    每个 chunk 归一化成一个 dict（idx/page/source/section/text/...）
        index_map: (source, page, section, text) -> corpus 下标

    为什么要 index_map：
        向量检索返回的是 Milvus 的点，BM25 需要的是语料下标，两路结果要融合就必须
        能互相映射。用 (来源, 页码, 章节, 正文) 四元组作为身份标识，
        比用 Milvus 主键更稳 —— 因为 BM25 路的数据完全来自本地语料、没有主键概念。
        四元组带上了 text，实际等价于"内容相同的片段视为同一条"。
    """
    global _corpus, _index
    with _cache_lock:
        if _corpus is None:
            # limit 给到 10 万：这是"取出全量"的意思，当前知识库规模远小于此
            rows = query_vectors(COLLECTION, limit=100000)
            _corpus = []
            _index = {}
            for index, row in enumerate(rows):
                payload = row["payload"]
                item = {
                    "idx": index,
                    "page": int(payload.get("page", -1)),
                    "source": str(payload.get("source", "")),
                    "section": payload.get("section", ""),
                    "text": payload.get("text", ""),
                    "semantic_type": payload.get("semantic_type", ""),
                    "important_kwd": payload.get("important_kwd", []),
                }
                _corpus.append(item)
                # setdefault：同一内容出现多次时保留首次出现的下标，避免后来的覆盖先前的
                _index.setdefault((item["source"], item["page"], item["section"], item["text"]), index)
    return _corpus, _index

def bm25_index() -> BM25:
    """惰性构建并复用 BM25 索引。

    返回：
        进程级共享的 BM25 实例。

    双检锁（先无锁判空，再加锁判空）：
        没有第一层判断的话，每次检索都要抢锁，多请求时全排在锁上等；
        没有第二层判断的话，两个请求同时发现 _bm25 为空，会各建一遍索引。
    """
    global _bm25
    if _bm25 is None:
        corpus, _ = load_corpus()
        with _cache_lock:
            if _bm25 is None:
                _bm25 = BM25([item["text"] for item in corpus])
    return _bm25
