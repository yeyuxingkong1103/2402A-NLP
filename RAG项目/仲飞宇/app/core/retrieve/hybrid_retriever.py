"""混合检索 + 多路召回 + RRF 融合。

两路召回：
1. 稠密向量：query 向量化 -> Milvus ANN（IP 度量 + 归一化向量 = 余弦相似度）
2. 关键词  ：BM25（Python 侧 rank_bm25 + jieba 分词），对角色内全部 chunk 建索引

融合：RRF（Reciprocal Rank Fusion）对两路召回按排名融合去重。
生产可将 BM25 迁入 Milvus 稀疏向量（2.5+ 的 BM25 function），此处保持一致接口。

**索引缓存是每进程的**（读这块代码前必须先知道）：
BM25 索引建在本进程内存里（_bm25_cache），没有 TTL、不落盘、也不跨进程共享；稠密一路
每次都现查 Milvus，没有缓存。于是入库之后**必须由处理该请求的那个进程**调
invalidate(role_id)，否则新文档在关键词一路看不见（见 app/api/knowledge.py 上传后的调用）。
多 worker（nginx 负载均衡）下这点尤其要留神：只有收到上传的那个 worker 会失效，
其余 worker 会继续用旧索引直到进程重启或者那次请求落到它头上才重建——表现出来是
"同一份资料有时召得回、有时召不回"，而两路召回不再对齐这件事本身没有任何日志。
"""
from __future__ import annotations

import threading

import jieba
from rank_bm25 import BM25Okapi

from ..config import Settings
from ..embedding import EmbeddingClient
from ..logging_config import get_logger
from ..store.milvus_store import MilvusStore
from .query_rewrite import QueryRewriter

log = get_logger("retrieve")


def tokenize(text: str) -> list[str]:
    """中文分词 + 英文小写，供 BM25 使用。

    语料和 query **必须**走同一个函数：BM25 的打分靠词表精确匹配，而 BM25Index.search
    里"有没有真的共享词"那一步是拿两边的 token 集合求交集（q_tokens & set(toks)）。
    换个分词方式（比如 query 侧用别的切法）不会报错，只会让交集恒为空、关键词一路静默归零。

    jieba 是懒加载的：首次调用才构建前缀词典，明显慢于后续调用（BM25 首次建索引慢有它一份）。
    """
    text = text.lower()
    return [t for t in jieba.lcut(text) if t.strip()]


def rrf_fusion(*ranked_lists: list, k: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion：多路召回的 id 列表 -> [(id, 融合分数)]，降序。

    入参是**若干"已按相关度排好序的 id 列表"**，不看分数本身：每条列表里第 rank 名贡献
    1/(k+rank+1)，同一 id 在多路里出现就把贡献加起来。这正是它能把两路凑到一起的原因——
    稠密路给的是余弦相似度（0~1 的连续值），BM25 给的是没有上界、还会为负的分数（见
    BM25Index.search），两者量纲完全不同，直接相加或归一化都需要再来一套调参；
    只取排名就绕开了整件事，也让"多查询改写"这种路数不定（1~4 条 query × 2 路）的场景
    天然可扩展。

    k=60 是 RRF 原论文的取值：它的作用是压平头部——k 越小，第 1 名越占绝对优势（k=0 时
    第 1 名得分 1、第 2 名 0.5）；k=60 让第 1 名与第 2 名的差距只有约 1.6%，于是"被多路
    同时召回"比"在某一路排第一"更能决定排序，符合混合检索的意图。

    返回值里的分数只是**排名融合分**（量级很小，大约 0.01~0.13），只能用来排序：
    既不是相似度、也不能和 score_threshold 这类阈值比（reranker.py 里也标注了这一点）。
    """
    score: dict = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)


class BM25Index:
    """对角色内 chunk 建立 BM25 关键词索引。"""

    def __init__(self, items: list[dict]):
        # items: [{id, text, title, source, chunk_index}, ...]（来自 milvus.list_texts，
        # 单角色上限 10000 条；语料顶到上限时那一路只是"覆盖不全"，不会报错，见 list_texts）
        # 整份语料的原文和分词结果都常驻内存，且和 HybridRetriever 同生命周期——这就是
        # 缓存不能无限增长的原因（见 _get_bm25 的说明）。
        self.items = items
        self.corpus = [tokenize(it["text"]) for it in items]
        # 全部 chunk 分词后为空时（corpus 非空但每条都是空列表），BM25Okapi 的
        # avgdl = 0 会在打分时除零，这里直接退化为「无关键词召回」。
        ok = bool(self.corpus) and any(self.corpus)
        self.bm25 = BM25Okapi(self.corpus) if ok else None

    def search(self, query: str, top_k: int) -> list[dict]:
        """关键词召回，最多 top_k 条（实际可能更少：只有真的共享词条目的才在候选里）。

        注意这一路**不设分数阈值**，和稠密路的 score_threshold 不对称，是有意的：
        上面已经说明 BM25 的分数可能为 0 甚至为负，能过滤的只有"有没有共享词"。
        想给两路加统一阈值只能加在 RRF 融合分上，而那又是个排名分，不能当相似度用。
        """
        if self.bm25 is None:
            return []
        q_tokens = set(tokenize(query))
        scores = self.bm25.get_scores(tokenize(query))

        # 先用「有没有真的共享词」筛，再按分数排序截断。
        #
        # 不能先排序截断、再拿分数过滤：rank_bm25 的 BM25Okapi 用平滑 IDF
        # ln((N-n+0.5)/(n+0.5))，分数为零**不代表词没出现**——
        #   · df == N/2 时 IDF 恰为 ln(1) = 0，真的命中了也是 0 分；
        #   · df > N/2 时命中项为负，而未命中的项恰好是 0.0，降序排在命中项前面，
        #     先切片就把命中的挤出去了。
        # 实测（7 块的高血压语料）：`盐` 出现在 4 块里，top_k=5 只返回 1 条；
        # top_k 再小一点就整路归零，而稠密路察觉不到——两路召回不对齐且无任何日志。
        matched = [i for i, toks in enumerate(self.corpus) if q_tokens & set(toks)]
        ranked = sorted(matched, key=lambda i: scores[i], reverse=True)
        return [{**self.items[i], "score": float(scores[i])} for i in ranked[:top_k]]


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedding: EmbeddingClient,
        milvus: MilvusStore,
        rewriter: QueryRewriter | None = None,
    ):
        self.settings = settings
        self.embedding = embedding
        self.milvus = milvus
        self.rewriter = rewriter or QueryRewriter(settings)
        # 每角色一份 BM25 索引，键是 role_id、**没有上限也没有淘汰**：条目里装着该角色
        # 全量语料的原文与分词结果。角色可以由 POST /roles 动态创建，所以严格说这个字典
        # 能一直涨——当前预设角色只有十来个、语料量级也小，才没做 LRU（对比 memory.py
        # 的 MAX_SESSIONS：那边 session_id 完全由客户端控制，所以必须兜底）。
        self._bm25_cache: dict[str, BM25Index] = {}
        # 同步路由跑在线程池里，多个请求并发首次访问同一角色时都会看到缓存未命中、
        # 各自去建一份 BM25 索引（结果一致但重复劳动）。串行化构建。
        self._bm25_lock = threading.Lock()

    def invalidate(self, role_id: str | None = None) -> None:
        """入库后失效 BM25 缓存。

        这里是**唯一**的失效入口：没有 TTL、没有基于 Milvus 版本号的自动失效，漏调就是
        关键词一路长期停留在旧语料上（不报错、日志也没有痕迹）。role_id 传 None 清空全部。
        """
        with self._bm25_lock:
            if role_id:
                self._bm25_cache.pop(role_id, None)
            else:
                self._bm25_cache.clear()

    def _get_bm25(self, role_id: str) -> BM25Index:
        """取角色的 BM25 索引，没有就现建。

        建索引是**同步**的：要拉全量语料（list_texts）+ 逐条 jieba 分词，所以某个角色
        的第一轮对话会明显慢一截，之后才走缓存。缓存是进程内的，多 worker 下每个 worker
        都要各自付一遍这份冷启动成本——这也是"重启后第一次检索慢"的常见来源。
        """
        if role_id not in self._bm25_cache:
            with self._bm25_lock:
                if role_id not in self._bm25_cache:  # 等锁期间可能已被别的线程建好
                    items = self.milvus.list_texts(role_id)
                    self._bm25_cache[role_id] = BM25Index(items)
        return self._bm25_cache[role_id]

    def retrieve(self, query: str, role_id: str, top_k: int | None = None) -> list[dict]:
        """混合检索，返回按 RRF 融合分降序、**已经去重**的候选（不是 top_k 条）。

        三个容易踩的契约：
        · top_k 是**每条 query、每一路各自**的召回深度，不是返回条数上限。开了 query 改写
          时条数可达 top_k 的数倍（实测 rerank_pool=20 + 4 条 query → 80 条），
          所以调用方要自己截断——pipeline.retrieve 就是按精排池宽截的那一刀（见 pipeline.py）。
        · score_threshold 只作用在稠密一路；BM25 那一路不设阈值（原因见 BM25Index.search）。
        · 每条结果的 "score" 是**融合后的排名分**（会覆盖掉原来的向量分），想看单路原始分
          得读 "dense_score" / "bm25_score"：同一 chunk 被多条 query 命中时取各路的**最大值**，
          没命中的那路是 None。三个字段量纲互不相同，别混着比。
        """
        top_k = top_k or self.settings.top_k
        # Query 改写/扩写：多查询各自召回后 RRF 融合（关闭时就是单查询，退化为原行为）
        queries = self.rewriter.rewrite(query)
        bm25 = self._get_bm25(role_id)

        merged: dict[str, dict] = {}
        dense_lists: list[list[str]] = []
        bm25_lists: list[list[str]] = []

        for q in queries:
            # 1) 稠密召回 + 余弦阈值过滤
            qv = self.embedding.embed_query(q)
            dense = [
                d for d in self.milvus.search(role_id, qv, top_k)
                if d["score"] >= self.settings.score_threshold
            ]
            for h in dense:
                cur = merged.setdefault(h["id"], {**h, "dense_score": None, "bm25_score": None})
                cur["dense_score"] = h["score"] if cur["dense_score"] is None else max(cur["dense_score"], h["score"])
            dense_lists.append([d["id"] for d in dense])

            # 2) BM25 关键词召回
            hits = bm25.search(q, top_k)
            for h in hits:
                cur = merged.setdefault(h["id"], {**h, "dense_score": None, "bm25_score": None})
                cur["bm25_score"] = h["score"] if cur["bm25_score"] is None else max(cur["bm25_score"], h["score"])
            bm25_lists.append([b["id"] for b in hits])

        # 3) 多查询多路的 id 列表一起 RRF 融合排序
        # 每条 query 贡献两条列表（稠密、BM25），同一 query 两路都命中的 chunk 会拿到两份
        # 1/(k+rank+1)——"两路共识"就是这样被顶到前面的，这是混合检索想要的效果，不是重复计分。
        # 某个 query 一路没召回就是空列表，rrf_fusion 对空列表无副作用（没有任何 id 可加），
        # 所以不用先过滤；查询数变化（改写开关、模型少给一条）也不会打乱已排好的相对次序。
        ranked = rrf_fusion(*dense_lists, *bm25_lists)
        results = []
        for cid, fusion_score in ranked:
            item = merged[cid]
            item["score"] = fusion_score
            results.append(item)
        return results
