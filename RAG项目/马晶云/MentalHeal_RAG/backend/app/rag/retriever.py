from functools import lru_cache
import json
from pathlib import Path
import re
import threading
from typing import Any
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from app.core.config import Settings, get_settings


# 这个类就是整个 RAG 检索部分的核心。
# 它做的事情可以概括成：向量召回 + BM25 召回 + rerank 精排 + 去重筛选。
class KnowledgeRetriever:
    # 这里是一些心理健康场景里的常见关键词。
    # 注意：这些词不是拿来生成回答的，而是辅助计算“字面/主题匹配分”。
    _TOPIC_KEYWORDS = {
        "sleep": (
            "睡眠",
            "失眠",
            "入睡",
            "早醒",
            "夜间醒",
            "睡前",
            "作息",
            "熬夜",
            "午睡",
        ),
        "anxiety": (
            "焦虑",
            "紧张",
            "呼吸",
            "平复",
            "接地",
            "放松",
            "担心",
            "恐慌",
        ),
        "crisis": (
            "自杀",
            "自伤",
            "不想活",
            "危险",
            "急救",
            "陪伴",
            "危机",
            "伤害自己",
        ),
        "stress": (
            "压力",
            "缓解",
            "运动",
            "规律作息",
            "交流",
            "支持",
            "应对",
        ),
        "diagnosis": (
            "抑郁",
            "诊断",
            "症状",
            "专业人员",
            "心理咨询",
        ),
    }
    # 把上面所有主题关键词摊平成一个不重复的元组。
    # 比如后面判断 query 里有没有“失眠”“焦虑”时，就不用一组一组遍历。
    _LEXICAL_TERMS = tuple(
        # dict.fromkeys 有一个小技巧：可以在保持顺序的同时去重。
        dict.fromkeys(
            # term 是具体的关键词。
            term
            # keywords 是某一类主题的关键词列表，比如 sleep 里的那些词。
            for keywords in _TOPIC_KEYWORDS.values()
            # 从每一组 keywords 里再逐个取出 term。
            for term in keywords
        )
    )

    # settings 里保存了模型路径、Milvus 地址、检索 top_k、各种权重等配置。
    def __init__(self, settings: Settings) -> None:
        # 先检查 embedding 模型是否存在。
        # 如果模型路径错了，直接在启动阶段报清楚，不要等用户请求时才失败。
        if not Path(settings.embedding_model_name).exists():
            raise FileNotFoundError(f"BGE-m3 model not found: {settings.embedding_model_name}")
        # 再检查 rerank 模型是否存在。
        # rerank 模型负责精排，没有它后面的重排序步骤就跑不了。
        if not Path(settings.rerank_model_name).exists():
            raise FileNotFoundError(
                f"BGE-reranker model not found: {settings.rerank_model_name}"
            )

        # 保存配置，后面每个方法都可以从 self.settings 里取参数。
        self.settings = settings

        # embedder：把用户问题转成向量，用于向量数据库检索。
        # device="cpu" 表示这里用 CPU 跑模型。
        self.embedder = SentenceTransformer(settings.embedding_model_name, device="cpu")
        # reranker：输入“问题 + 候选文本”，输出更精细的相关性分数。
        self.reranker = CrossEncoder(settings.rerank_model_name, device="cpu")
        # 连接 Milvus。Milvus 里存的是知识库 chunk 的向量和元数据。
        self.client = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
        )

        # BM25 是传统关键词检索。
        # 这里先设成 None，等第一次检索时再构建，避免程序启动太慢。
        self._bm25: BM25Okapi | None = None
        # 这里保存 BM25 索引对应的 chunk 信息，比如 chunk_id、页码、文本等。
        self._bm25_items: list[dict[str, Any]] = []
        # signature 用来判断本地 chunk 文件有没有变化。
        # 如果文件没变，就不用重复构建 BM25 索引。
        self._bm25_signature: tuple[tuple[str, int, int], ...] = ()
        # 多个请求同时进来时，锁可以避免大家一起重建索引。
        self._bm25_lock = threading.Lock()

    # 这是对外最重要的方法：输入用户问题，返回最相关的知识库片段。
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        # 第一步：把用户问题转成向量。
        # 向量可以理解成“语义坐标”，意思相近的文本向量距离也会更近。
        vector = self.embedder.encode(
            # query 是用户输入的问题。
            query,
            # 归一化后更适合用余弦相似度比较。
            normalize_embeddings=True,
            # 转成 numpy，方便模型内部处理；下一行会再转成 list 给 Milvus。
            convert_to_numpy=True,
            # 接口调用时不需要显示进度条。
            show_progress_bar=False,
        ).tolist()

        # 第二步：用问题向量去 Milvus 里搜索相似 chunk。
        # 这一步是“向量召回”，先找出一批可能相关的候选片段。
        hits = self.client.search(
            # collection_name 指定知识库向量存在哪个集合里。
            collection_name=self.settings.milvus_collection_knowledge,
            # Milvus 支持一次查多个向量，所以这里要用列表包一层。
            data=[vector],
            # anns_field 是 Milvus 里保存向量的字段名。
            anns_field="vector",
            # COSINE 表示用余弦相似度；ef 越大通常召回越准，但也可能更慢。
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            # rag_top_k 表示向量检索阶段先取多少个候选。
            limit=self.settings.rag_top_k,
            # output_fields 表示除了相似度分数，还要把这些字段带回来。
            output_fields=[
                "chunk_id",
                "document_id",
                "source_file",
                "page_start",
                "page_end",
                "text",
            ],
        )

        # Milvus 返回结果是嵌套结构。因为这里只查了一个 query，所以用 hits[0]。
        # _normalize_hit 会把每条结果整理成统一格式，后面处理更方便。
        candidates = [self._normalize_hit(hit) for hit in hits[0]]
        # 过滤掉相似度太低的候选。
        # 这样可以减少后面 BM25 合并、rerank、综合打分的无效工作。
        candidates = [
            item
            for item in candidates
            if item["score"] >= self.settings.rag_score_threshold
        ]

        # 第三步：合并 BM25 关键词检索结果。
        # 向量检索擅长找“意思相近”的内容，BM25 擅长找“关键词命中”的内容，两者互补。
        candidates = self._merge_bm25_candidates(query, candidates)
        # 第四步：先筛出一小批候选交给 reranker。
        # reranker 更准但更慢，所以不能把所有候选都塞进去。
        candidates = self._select_rerank_candidates(query, candidates)
        # 如果前面所有检索都没找到合适内容，就直接返回空列表。
        if not candidates:
            return []

        # 第五步：reranker 精排。
        # 这里会把每个候选片段和 query 组成一对，让模型判断它们有多相关。
        scores = self.reranker.predict([(query, item["text"]) for item in candidates])
        # 把 reranker 的分数写回对应的 candidate。
        # zip 会把 candidates 和 scores 一一配对。
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        # 第六步：计算最终综合分。
        # 最终分数不是只看一个指标，而是综合向量分、rerank 分、关键词分和 BM25 分。
        self._score_candidates(query, candidates)
        # top_k 如果调用方传了，就按调用方要求；否则用配置里的默认值。
        # 这里最多限制到 20，避免一次返回太多内容。
        result_count = min(top_k or self.settings.rag_rerank_top_k, 20)

        # 第七步：最终选择返回结果。
        # 除了按分数排序，还会限制同一文档、同一页的数量，避免结果重复。
        return self._select_candidates(
            candidates,
            result_count,
            max_chunks_per_document=self.settings.rag_max_chunks_per_document,
            max_chunks_per_page=self.settings.rag_max_chunks_per_page,
        )

    # 这个方法负责把 BM25 检索结果和向量检索结果合并。
    def _merge_bm25_candidates(
        self,
        query: str,
        vector_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # 确保 BM25 索引可用。
        # 如果本地 chunk 文件变了，这个方法内部也会自动重建索引。
        self._ensure_bm25_index()
        # 如果 BM25 没有成功建立，或者没有可检索的 chunk，就直接返回向量候选。
        if self._bm25 is None or not self._bm25_items:
            return vector_candidates

        # BM25 不能直接吃原始句子，需要先把 query 切成 token。
        tokenized_query = self._tokenize(query)
        # raw_scores 是每个 chunk 对这个 query 的 BM25 原始分数。
        raw_scores = self._bm25.get_scores(tokenized_query)

        # BM25 阶段最多取多少个候选。
        # 这里取 bm25_top_k 和 rag_top_k 中较大的值，避免 BM25 召回太少。
        limit = max(self.settings.bm25_top_k, self.settings.rag_top_k)
        # 按 BM25 分数从高到低排列 chunk 的下标，然后截取前 limit 个。
        ranked_indexes = sorted(
            range(len(raw_scores)), key=lambda index: raw_scores[index], reverse=True
        )[:limit]
        # maximum 是这批 BM25 候选里的最高分，用来做归一化。
        maximum = max((float(raw_scores[index]) for index in ranked_indexes), default=0.0)

        # 把 BM25 命中的结果整理成：chunk_id -> (chunk信息, 原始分, 最高分)。
        # 后面用 chunk_id 合并，可以避免同一个片段重复出现。
        bm25_by_id = {
            self._bm25_items[index]["chunk_id"]: (
                self._bm25_items[index],
                float(raw_scores[index]),
                maximum,
            )
            for index in ranked_indexes
            # BM25 分数为 0 基本代表没命中关键词，没必要合并进来。
            if raw_scores[index] > 0
        }

        # 先把向量候选放进字典。
        # key 是 chunk_id，这样 BM25 如果命中了同一个 chunk，就只更新分数，不重复添加。
        merged = {item["chunk_id"]: item for item in vector_candidates}
        # 遍历 BM25 找到的候选，逐个合并。
        for chunk_id, (item, score, maximum_score) in bm25_by_id.items():
            # 先看这个 chunk 是否已经被向量检索找到了。
            candidate = merged.get(chunk_id)
            # 如果没有，就说明这是 BM25 独立补召回的结果。
            if candidate is None:
                # 复制一份 item，避免修改 _bm25_items 里的原始数据。
                candidate = dict(item)
                # 因为不是向量检索召回的，所以向量分先设为 0。
                candidate["score"] = 0.0
                # 放进合并后的候选池。
                merged[chunk_id] = candidate

            # 保存 BM25 原始分，方便调试或展示。
            candidate["bm25_score"] = score
            # 保存 BM25 归一化分。
            # 归一化后它才能和向量分、rerank 分一起做加权计算。
            candidate["bm25_normalized"] = (
                0.5 + 0.5 * score / maximum_score if maximum_score > 0 else 0.0
            )

        # 对没有 BM25 分的向量候选补默认值。
        # 这样后面统一取 bm25_score 时，不会出现 KeyError。
        for candidate in merged.values():
            candidate.setdefault("bm25_score", 0.0)
            candidate.setdefault("bm25_normalized", 0.0)
        # merged 是字典，返回前转成列表。
        return list(merged.values())

    # 这个方法是在 rerank 之前做一次“预筛选”。
    def _select_rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # 计算最多送多少个候选给 reranker。
        # max(1, ...) 保证只要有候选，至少会保留 1 个。
        limit = max(1, min(self.settings.rag_rerank_candidates, len(candidates)))
        # 收集向量分数，后面归一化时要用整批分数的最大最小值。
        vector_scores = [float(item.get("score", 0.0)) for item in candidates]

        # 这三个 weight 控制预筛选时各类分数的重要程度。
        vector_weight = self.settings.rag_vector_weight
        lexical_weight = self.settings.rag_lexical_weight
        # getattr 是为了兼容旧配置：如果没有 rag_bm25_weight，就默认 0。
        bm25_weight = getattr(self.settings, "rag_bm25_weight", 0.0)
        # 总权重用于做加权平均。
        total_weight = vector_weight + lexical_weight + bm25_weight
        # 如果配置错了，总权重小于等于 0，就兜底成 1，避免除以 0。
        if total_weight <= 0:
            total_weight = 1.0

        # ranked 里保存的是 (预筛选分数, 候选片段)。
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in candidates:
            # 先把向量分归一化，否则它和其他分数不一定在同一个范围。
            normalized_vector = self._normalize_score(float(item.get("score", 0.0)), vector_scores)
            # lexical_score 是字面关键词匹配分。
            lexical_score = self._lexical_score(query, str(item.get("text", "")))
            # bm25_score 是 BM25 的归一化分，没有的话默认 0。
            bm25_score = float(item.get("bm25_normalized", 0.0))
            # 用比较便宜的几个分数先算一个 preliminary_score。
            # 注意这里还没用 rerank 分，因为 rerank 还没开始跑。
            preliminary_score = (
                vector_weight * normalized_vector
                + lexical_weight * lexical_score
                + bm25_weight * bm25_score
            ) / total_weight
            ranked.append((preliminary_score, item))

        # 按预筛选分数从高到低排序。
        # 如果预筛选分一样，再用向量分、BM25 分、chunk_id 继续比较，让排序稳定。
        ranked.sort(
            key=lambda pair: (
                pair[0],
                float(pair[1].get("score", 0.0)),
                float(pair[1].get("bm25_normalized", 0.0)),
                str(pair[1].get("chunk_id", "")),
            ),
            reverse=True,
        )
        # 只返回前 limit 个，交给 reranker 精排。
        return [item for _, item in ranked[:limit]]

    # 这个方法负责准备 BM25 索引。
    # 它会读取本地 chunk json 文件，把里面的文本建立成关键词检索索引。
    def _ensure_bm25_index(self) -> None:
        # 找出 chunks_data_dir 目录下所有 json 文件。
        # 这些文件通常是文档切块后的结果。
        chunk_files = sorted(Path(self.settings.chunks_data_dir).glob("*.json"))
        # signature 是“文件状态指纹”：路径 + 修改时间 + 文件大小。
        # 只要文件内容变化，修改时间或大小通常就会变。
        signature = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in chunk_files
        )
        # 如果这次文件指纹和上次一样，说明索引还是新的，直接返回。
        if signature == self._bm25_signature:
            return

        # 文件有变化时才进入锁，准备重建 BM25。
        with self._bm25_lock:
            # 加锁后再检查一遍，防止别的线程已经抢先重建好了。
            if signature == self._bm25_signature:
                return

            # items 用来保存所有可被 BM25 检索的 chunk。
            items: list[dict[str, Any]] = []
            # 逐个读取 chunk 文件。
            for chunk_file in chunk_files:
                try:
                    # 每个 chunk_file 是 json，里面一般包含 document_id、source_path、chunks 等字段。
                    payload = json.loads(chunk_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # 如果某个文件读不了或格式坏了，就跳过。
                    # 这样不会因为一个坏文件导致整个检索器不可用。
                    continue

                # source_file 只保留文件名，不保留完整路径。
                source_file = Path(str(payload.get("source_path", ""))).name
                # document_id 用来标记这个 chunk 属于哪篇文档。
                document_id = str(payload.get("document_id", ""))
                # 遍历当前文档里的所有 chunk。
                for chunk in payload.get("chunks", []):
                    # text 是 chunk 的正文；strip 去掉前后空白。
                    text = str(chunk.get("text", "")).strip()
                    # chunk_id 是 chunk 的唯一标识。
                    chunk_id = str(chunk.get("chunk_id", "")).strip()
                    # 没有正文或没有 id 的 chunk 没法检索，直接跳过。
                    if not text or not chunk_id:
                        continue

                    # 整理成和 Milvus 返回结果接近的格式。
                    # 这样后面合并 BM25 和向量结果时，字段能对得上。
                    items.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id or None,
                            "title": source_file or "未知来源",
                            "page_start": int(chunk.get("page_start", 0)),
                            "page_end": int(chunk.get("page_end", 0)),
                            "text": text,
                            # 本地 BM25 结果本身没有向量分，所以先设为 0。
                            "score": 0.0,
                        }
                    )

            # 保存原始 item，后面 BM25 命中后还要通过下标拿回 chunk 信息。
            self._bm25_items = items
            # 建 BM25 索引前，需要先把每个 chunk 的 text 分词。
            # 如果 items 为空，就把 _bm25 设成 None。
            self._bm25 = BM25Okapi([self._tokenize(item["text"]) for item in items]) if items else None
            # 更新文件指纹，表示当前索引已经对应最新文件状态。
            self._bm25_signature = signature

    # 这个方法给 BM25 做分词。
    @classmethod
    def _tokenize(cls, value: str) -> list[str]:
        # 先标准化文本：去掉空白、统一英文大小写。
        normalized = cls._normalize_text(value)
        # 空文本直接返回空 token 列表。
        if not normalized:
            return []
        # 英文和数字按连续字符串提取；中文先按单个字提取。
        tokens = re.findall(r"[a-z0-9]+|[一-鿿]", normalized)
        # 中文只按单字会太碎，所以再额外提取相邻两个中文字符组成的双字词。
        bigrams = cls._extract_cjk_bigrams(normalized)
        # 合并单字/英文 token 和中文双字词，并去重保序。
        return list(dict.fromkeys(tokens + bigrams))

    # 这个方法计算最终综合分，直接修改 candidates 里的每个 item。
    def _score_candidates(self, query: str, candidates: list[dict[str, Any]]) -> None:
        # 取出整批 rerank 分数，用于归一化。
        rerank_scores = [item["rerank_score"] for item in candidates]
        # 取出整批向量分数，用于归一化。
        vector_scores = [item["score"] for item in candidates]

        # 这些权重决定最终分数里各部分占比。
        vector_weight = self.settings.rag_vector_weight
        rerank_weight = self.settings.rag_rerank_weight
        lexical_weight = self.settings.rag_lexical_weight
        bm25_weight = getattr(self.settings, "rag_bm25_weight", 0.0)
        # 加权平均时需要除以总权重。
        total_weight = vector_weight + rerank_weight + lexical_weight + bm25_weight
        if total_weight <= 0:
            total_weight = 1.0

        # 逐个候选计算最终分。
        for item in candidates:
            # 向量分归一化：保留相对高低，但压到统一范围。
            normalized_vector = self._normalize_score(item["score"], vector_scores)
            # rerank 分归一化：reranker 原始分可能范围不固定，所以也要统一一下。
            normalized_rerank = self._normalize_score(
                item["rerank_score"],
                rerank_scores
            )
            # 计算字面关键词匹配分。
            lexical_score = self._lexical_score(query, item["text"])
            # 保存下来，方便前端展示或调试排序原因。
            item["lexical_score"] = lexical_score
            # 最终综合分：四类分数按配置权重相加，再除以总权重。
            item["combined_score"] = (
                vector_weight * normalized_vector
                + rerank_weight * normalized_rerank
                + lexical_weight * lexical_score
                + bm25_weight * float(item.get("bm25_normalized", 0.0))
            ) / total_weight

    # 把某个分数按当前这一批分数的最大最小值做归一化。
    @staticmethod
    def _normalize_score(score: float, scores: list[float]) -> float:
        # 没有分数时返回 0，表示没有可用依据。
        if not scores:
            return 0.0
        # 找到这一批分数的最小值和最大值。
        minimum = min(scores)
        maximum = max(scores)
        # 如果最大值等于最小值，说明大家都一样，没法比较高低。
        if maximum == minimum:
            return 0.5
        # 这里归一化到 0.5 到 1.0，而不是 0 到 1。
        # 这样可以避免某个来源的最低分被压得太低，影响综合分过大。
        return 0.5 + 0.5 * (score - minimum) / (maximum - minimum)

    # 计算 query 和 text 的字面匹配程度。
    @classmethod
    def _lexical_score(cls, query: str, text: str) -> float:
        # 两边都先标准化，避免空格、大小写影响匹配。
        normalized_query = cls._normalize_text(query)
        normalized_text = cls._normalize_text(text)
        # 任意一边为空，都无法做字面匹配。
        if not normalized_query or not normalized_text:
            return 0.0
        # 如果整个问题直接出现在文本里，说明字面上非常匹配，直接给满分。
        if normalized_query in normalized_text:
            return 1.0

        # 优先使用预设心理健康关键词。
        # 例如 query 中出现“失眠”，就把它作为匹配项。
        query_terms = [term for term in cls._LEXICAL_TERMS if term in normalized_query]
        # 如果没有命中预设关键词，就用中文双字词兜底。
        # 这样即使用户问了词库外的问题，也能有一个简单的字面匹配分。
        if not query_terms:
            query_terms = cls._extract_cjk_bigrams(normalized_query)
        # 如果还是没有可用词，就返回 0。
        if not query_terms:
            return 0.0

        # direct_hits 统计 query_terms 里有多少个直接出现在候选文本里。
        direct_hits = sum(term in normalized_text for term in query_terms)
        # direct_score 是直接命中比例。
        # 比如 4 个关键词命中 2 个，分数就是 0.5。
        direct_score = direct_hits / len(query_terms)

        # 除了直接关键词匹配，还会看“主题匹配”。
        # 比如问题有“失眠”，文本里有“睡眠”“作息”，虽然不是完全同词，也算有关系。
        topic_scores = []
        # 遍历每一类主题关键词。
        for keywords in cls._TOPIC_KEYWORDS.values():
            # 只有当 query 命中这个主题时，才计算这个主题的文本命中率。
            if any(term in normalized_query for term in keywords):
                # 统计候选文本里命中这个主题多少个关键词。
                topic_hits = sum(term in normalized_text for term in keywords)
                # 主题命中比例加入列表。
                topic_scores.append(topic_hits / len(keywords))
        # 如果命中了多个主题，取最高的主题分。
        topic_score = max(topic_scores, default=0.0)
        # 最终字面分：直接关键词占 75%，主题相关占 25%，最高不超过 1。
        return min(1.0, 0.75 * direct_score + 0.25 * topic_score)

    # 文本标准化：为了让匹配更稳定。
    @staticmethod
    def _normalize_text(value: str) -> str:
        # casefold 比 lower 更彻底，适合做大小写无关比较。
        # re.sub(r"\s+", "", ...) 会去掉所有空白字符。
        return re.sub(r"\s+", "", value.casefold())

    # 提取中文双字词。
    @staticmethod
    def _extract_cjk_bigrams(value: str) -> list[str]:
        # 比如“心理健康”会切成“心理”“理健”“健康”。
        # 这样做虽然简单，但比只按单个字匹配更接近中文关键词。
        return list(
            # dict.fromkeys 用来去重并保持出现顺序。
            dict.fromkeys(
                # 从 index 开始取两个字符。
                value[index : index + 2]
                # 最后一个字符没法组成双字词，所以只遍历到 len(value) - 2。
                for index in range(len(value) - 1)
                # 只有两个字符都是中文时，才保留这个 bigram。
                if all("一" <= char <= "鿿" for char in value[index : index + 2])
            )
        )

    # 从已经打好分的候选里，挑出最终要返回给上层的结果。
    @staticmethod
    def _select_candidates(
        candidates: list[dict[str, Any]],
        result_count: int,
        *,
        max_chunks_per_document: int = 2,
        max_chunks_per_page: int = 2,
    ) -> list[dict[str, Any]]:
        # 先整体按分数排序。
        # combined_score 是最终综合分；如果没有，就退回用 rerank_score。
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.get("combined_score", item.get("rerank_score", 0.0)),
                item.get("rerank_score", 0.0),
                item.get("score", 0.0),
                item.get("chunk_id", ""),
            ),
            reverse=True,
        )

        # selected 是最终要返回的结果。
        selected: list[dict[str, Any]] = []
        # deferred 是暂时推迟的结果。
        # 比如某篇文档已经返回太多片段，它就先放这里，等结果不够时再补。
        deferred: list[dict[str, Any]] = []
        # 记录已经选过的 chunk_id，避免同一个 chunk 重复出现。
        seen_chunks: set[str] = set()
        # 记录已经选过的文本内容，避免不同 chunk 但文本一样。
        seen_texts: set[str] = set()
        # 统计每篇文档已经选了几个 chunk。
        document_counts: dict[str, int] = {}
        # 统计每个页面范围已经选了几个 chunk。
        page_counts: dict[tuple[str, int, int], int] = {}

        # 从高分到低分遍历候选。
        for item in ranked:
            # 当前候选的唯一 id。
            chunk_id = str(item.get("chunk_id", ""))
            # 文本也做一次标准化，用作内容去重。
            text_key = KnowledgeRetriever._normalize_text(str(item.get("text", "")))
            # 如果 chunk_id 已经出现过，或者文本已经出现过，就跳过。
            if chunk_id in seen_chunks or (text_key and text_key in seen_texts):
                continue

            # 优先用 document_id 识别文档；如果没有，就用 title 兜底。
            document_id = str(item.get("document_id") or item.get("title") or "")
            # page_key 用来识别“同一文档里的同一个页码范围”。
            page_key = (
                document_id,
                int(item.get("page_start", 0)),
                int(item.get("page_end", 0)),
            )
            # 如果同一文档或同一页已经选太多，就暂时推迟。
            # 这样可以让最终结果来源更分散，不会全堆在一页或一篇文档里。
            if (
                document_counts.get(document_id, 0) >= max_chunks_per_document
                or page_counts.get(page_key, 0) >= max_chunks_per_page
            ):
                deferred.append(item)
                continue

            # 通过去重和数量限制后，就正式选中这个候选。
            selected.append(item)
            # 记录 chunk_id 已经用过。
            seen_chunks.add(chunk_id)
            # 文本不为空时，也记录文本已经用过。
            if text_key:
                seen_texts.add(text_key)
            # 更新当前文档的计数。
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            # 更新当前页码范围的计数。
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
            # 如果已经选够数量，就提前结束。
            if len(selected) >= result_count:
                break

        # 如果严格控制文档/页码数量后，结果还不够，就从 deferred 里补。
        # 这一步是在“减少重复”和“保证数量”之间做平衡。
        if len(selected) < result_count:
            for item in deferred:
                # 补充时仍然要做 chunk 和文本去重。
                chunk_id = str(item.get("chunk_id", ""))
                text_key = KnowledgeRetriever._normalize_text(str(item.get("text", "")))
                if chunk_id in seen_chunks or (text_key and text_key in seen_texts):
                    continue
                # 加入最终结果。
                selected.append(item)
                # 更新去重记录。
                seen_chunks.add(chunk_id)
                if text_key:
                    seen_texts.add(text_key)
                # 补够数量就停止。
                if len(selected) >= result_count:
                    break
        return selected

    # 把 Milvus 原始 hit 转成项目统一使用的 candidate 格式。
    @staticmethod
    def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any]:
        # Milvus 的业务字段在 entity 里，外层还有 distance 等检索信息。
        entity = hit.get("entity", {})
        # 统一字段名后，后面无论是向量结果还是 BM25 结果，都能用同一套逻辑处理。
        return {
            # chunk_id 是知识片段的唯一标识。
            "chunk_id": str(entity.get("chunk_id", "")),
            # document_id 是片段所属文档；空字符串转成 None。
            "document_id": str(entity.get("document_id", "")) or None,
            # title 这里使用源文件名，后面展示来源时会用到。
            "title": str(entity.get("source_file", "未知来源")),
            # 页码信息用于回答时标注出处。
            "page_start": int(entity.get("page_start", 0)),
            "page_end": int(entity.get("page_end", 0)),
            # text 是真正给大模型参考的知识片段正文。
            "text": str(entity.get("text", "")),
            # Milvus 返回的 distance 在这里作为向量检索分数使用。
            "score": float(hit.get("distance", 0.0)),
        }


# maxsize=1 表示这个函数最多缓存一个 retriever 实例。
# 这样整个服务运行期间通常只加载一次 embedding 模型和 rerank 模型。
@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetriever:
    # 读取配置并创建检索器。
    # 后续再次调用 get_retriever() 时，会直接返回缓存里的同一个对象。
    return KnowledgeRetriever(get_settings())
