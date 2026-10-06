# -*- coding: utf-8 -*-
"""知识库服务：PDF 解析入库（分块+向量化+BM25）与混合检索+重排序。

embedder / reranker / milvus 均为注入依赖，便于单元测试与模型切换。
"""
import hashlib
# 解析：哈希模块（块级去重）
import time
# 解析：时间模块（入库时间戳）

from app.rag.bm25 import BM25Encoder
# 解析：BM25 稀疏向量编码器
from app.rag.chunking import chunk_by_headings, chunk_text, parent_child_chunk, semantic_chunk
# 解析：分块四模式函数
from app.rag.cleaning import clean_text, make_summary
# 解析：数据清洗与抽取式摘要
from app.rag.ocr import recognize_image
# 解析：图片 OCR 识别（模块级单例）
from app.rag.pdf import extract_tables, extract_text, remove_watermark
# 解析：PDF 文本提取、表格提取、去水印


# 知识库服务：入库（去水印/清洗/三模式分块/去重/向量化）与检索（混合+重排+Query改写）
class KnowledgeService:
    def __init__(
        # 解析：构造——全部依赖注入
        self,
        embedder,
        # 解析：向量化模型（BGE-m3）
        reranker,
        # 解析：重排序模型（BGE-reranker）
        milvus,
        # 解析：Milvus 向量库
        chunk_size: int = 700,
        # 解析：分块大小
        overlap: int = 80,
        # 解析：块间重叠
        remove_watermark: bool = True,
        # 解析：上传 PDF 是否自动去水印
        query_rewriter=None,
        # 解析：查询改写器（可选）
        chunking_mode: str = "sentence",
        # 解析：分块模式（sentence/heading/semantic/parent_child）
        semantic_threshold: float = 0.5,
        # 解析：语义分块相似度阈值
        parent_chunk_size: int = 2400,
        # 解析：父子块模式父块大小
    ):
        self.embedder = embedder
        # 解析：保存向量化模型
        self.reranker = reranker
        # 解析：保存重排模型
        self.milvus = milvus
        # 解析：保存 Milvus
        self.chunk_size = chunk_size
        # 解析：保存分块大小
        self.overlap = overlap
        # 解析：保存重叠量
        self.remove_watermark = remove_watermark
        # 解析：保存去水印开关
        self.query_rewriter = query_rewriter  # 可选：async rewrite(query)->list[str]
        # 解析：保存查询改写器（None 表示关闭改写）
        self.chunking_mode = chunking_mode  # sentence / heading / semantic / parent_child
        # 解析：保存分块模式
        self.semantic_threshold = semantic_threshold
        # 解析：保存语义阈值
        self.parent_chunk_size = parent_chunk_size  # 父子块模式父块大小
        # 解析：保存父块大小
        self.ocr_engine = None  # 图片 OCR 引擎（None 时用模块级单例）
        # 解析：OCR 引擎（测试注入假引擎；生产用模块级单例）
        self._bm25_cache: dict[int, BM25Encoder] = {}
        # 解析：BM25 编码器缓存（按角色键控，入库后失效）
        self._text_hashes: dict[int, set[str]] = {}
        # 解析：块文本哈希缓存（去重用，删除后失效）

    # ---------- 入库 ----------

    def ingest_pdf(self, role_id: int, pdf_bytes: bytes, source: str) -> dict:
        """解析 PDF（去水印→清洗→分块→去重）→ 向量化 → 写入 Milvus。

        去重规则：同 source 重传 = 文档级替换；文本 hash 重复 = 块级跳过。
        父子块模式：父块存 role_parents 集合，子块（带 parent_id）入检索集合。
        """
        if self.remove_watermark:
            # 解析：开启去水印
            pdf_bytes = remove_watermark(pdf_bytes)
            # 解析：去水印（文字三信号+图片多页检测，无水印原样透传）
        text = clean_text(extract_text(pdf_bytes))
        # 解析：提取文本并清洗（控制字符/垃圾行/空行）
        # 表格内容并入正文（Markdown 格式），保证表格数据可被检索
        tables = extract_tables(pdf_bytes)
        # 解析：提取 PDF 表格（Markdown 文本）
        if tables:
            # 解析：有表格
            text = text + "\n\n【表格】\n" + "\n\n".join(tables)
            # 解析：表格文本追加到正文尾部
        if self.chunking_mode == "parent_child":
            # 解析：父子块模式
            return self._ingest_parent_child(role_id, text, source)
            # 解析：走父子块专用入库流程
        chunks = self._chunk(text)
        # 解析：按配置模式分块
        if not chunks:
            # 解析：无有效块
            raise ValueError("PDF 中没有可提取的文本内容")
            # 解析：报错（上层转 400）

        self.milvus.ensure_collection(role_id)
        # 解析：确保该角色的知识库集合存在（幂等）

        # 块级去重：先与库内已有块比对（在可能的文档级删除之前算好）
        known = self._get_text_hashes(role_id)
        # 解析：取库内已有块的哈希集合
        fresh, fresh_hashes = [], set()
        # 解析：新鲜块列表与哈希集合
        for chunk in chunks:
            # 解析：逐块判重
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            # 解析：计算块文本哈希
            if digest in known or digest in fresh_hashes:
                # 解析：与库内已有或本批已收录重复
                continue
                # 解析：跳过重复块
            fresh.append(chunk)
            # 解析：新鲜块入列
            fresh_hashes.add(digest)
            # 解析：记录哈希

        existing_sources = {item["source"] for item in self.milvus.list_sources(role_id)}
        # 解析：库内已有文档名集合
        if source in existing_sources:
            # 解析：同文档重传
            if not fresh:
                # 解析：且内容完全重复
                # 同源且内容完全重复 → 无操作（不删不插）
                return {"source": source, "chunks": 0}
                # 解析：无操作返回 0 块（避免删了再插丢失数据）
            # 同源内容有更新 → 文档级替换：删旧插新
            self.milvus.delete_source(role_id, source)
            # 解析：删除旧文档全部块
            self._text_hashes.pop(role_id, None)
            # 解析：哈希缓存失效
            self._bm25_cache.pop(role_id, None)
            # 解析：BM25 缓存失效
            known = self._get_text_hashes(role_id)
            # 解析：基于删除后的库重建哈希集合
        elif not fresh:
            # 解析：新文档但全部块重复
            return {"source": source, "chunks": 0}
            # 解析：无操作返回 0 块

        chunks = fresh
        # 解析：只入库新鲜块
        dense_vectors = self.embedder.embed_documents(chunks)
        # 解析：BGE-m3 稠密向量化

        # BM25 词表 = 已有语料 + 新块，保证新旧文档统计一致
        corpus = self.milvus.all_texts(role_id) + chunks
        # 解析：完整语料（已有块+新块）
        encoder = BM25Encoder()
        # 解析：新建编码器
        encoder.fit(corpus)
        # 解析：全语料统计词表与 IDF
        sparse_rows = encoder.encode_texts(chunks)
        # 解析：新块的 BM25 稀疏向量

        now = int(time.time())
        # 解析：当前时间戳
        records = [
            # 解析：组装入库记录
            {
                "text": chunk,
                # 解析：块原文
                "summary": make_summary(chunk),
                # 解析：抽取式摘要
                "parent_id": "",
                # 解析：非父子块模式无父链接
                "dense": dense_vectors[i],
                # 解析：稠密向量
                "sparse": sparse_rows[i],
                # 解析：稀疏向量
                "source": source,
                # 解析：文档来源
                "created_at": now,
                # 解析：创建时间
                "updated_at": now,
                # 解析：更新时间
            }
            for i, chunk in enumerate(chunks)
            # 解析：逐块生成记录
        ]
        self.milvus.insert_chunks(role_id, records)
        # 解析：写入 Milvus
        self._bm25_cache.pop(role_id, None)  # 词表已变，缓存失效
        # 解析：BM25 缓存失效
        known.update(fresh_hashes)
        # 解析：哈希集合补充新块
        return {"source": source, "chunks": len(chunks)}
        # 解析：返回入库结果

    def ingest_image(self, role_id: int, image_bytes: bytes, source: str) -> dict:
        """图片入库：OCR 识别文字 → 清洗 → 分块 → 向量化（与 PDF 同链路）。"""
        ocr = self.ocr_engine if self.ocr_engine is not None else recognize_image
        # 解析：取 OCR 引擎（注入优先，否则模块级单例）
        text = clean_text(ocr(image_bytes))
        # 解析：OCR 识别并清洗
        if not text:
            # 解析：没有识别到文字
            raise ValueError("图片中没有识别到文字")
            # 解析：报错
        if self.chunking_mode == "parent_child":
            # 解析：父子块模式
            return self._ingest_parent_child(role_id, text, source)
            # 解析：走父子块入库
        chunks = self._chunk(text)
        # 解析：分块
        if not chunks:
            # 解析：无有效块
            raise ValueError("图片识别文字过短，无法分块")
            # 解析：报错

        self.milvus.ensure_collection(role_id)
        # 解析：确保集合存在
        known = self._get_text_hashes(role_id)
        # 解析：已有块哈希
        fresh, fresh_hashes = [], set()
        # 解析：新鲜块与哈希
        for chunk in chunks:
            # 解析：逐块判重
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            # 解析：块哈希
            if digest not in known and digest not in fresh_hashes:
                # 解析：不重复
                fresh.append(chunk)
                # 解析：新鲜块入列
                fresh_hashes.add(digest)
                # 解析：记录哈希
        if not fresh:
            # 解析：全部重复
            return {"source": source, "chunks": 0}
            # 解析：无操作返回

        dense_vectors = self.embedder.embed_documents(fresh)
        # 解析：稠密向量化
        corpus = self.milvus.all_texts(role_id) + fresh
        # 解析：BM25 完整语料
        encoder = BM25Encoder()
        # 解析：新建编码器
        encoder.fit(corpus)
        # 解析：统计词表
        sparse_rows = encoder.encode_texts(fresh)
        # 解析：稀疏向量
        now = int(time.time())
        # 解析：时间戳
        self.milvus.insert_chunks(role_id, [
            # 解析：组装并写入
            {
                "text": chunk,
                "summary": make_summary(chunk),
                "parent_id": "",
                "dense": dense_vectors[i],
                "sparse": sparse_rows[i],
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
            for i, chunk in enumerate(fresh)
            # 解析：逐块记录
        ])
        self._bm25_cache.pop(role_id, None)
        # 解析：BM25 缓存失效
        known.update(fresh_hashes)
        # 解析：哈希补充
        return {"source": source, "chunks": len(fresh)}
        # 解析：返回入库结果

    def _ingest_parent_child(self, role_id: int, text: str, source: str) -> dict:
        """父子块入库：父块存 role_parents 集合，子块带 parent_id 入检索集合。"""
        pairs = parent_child_chunk(text, self.parent_chunk_size, self.chunk_size, self.overlap)
        # 解析：父子块切分
        if not pairs:
            # 解析：无内容
            raise ValueError("PDF 中没有可提取的文本内容")
            # 解析：报错

        self.milvus.ensure_collection(role_id)
        # 解析：确保子块集合
        self.milvus.ensure_parent_collection(role_id)
        # 解析：确保父块集合

        # 文档级替换：同 source 先删（子块与父块都删）
        existing_sources = {item["source"] for item in self.milvus.list_sources(role_id)}
        # 解析：已有文档名
        if source in existing_sources:
            # 解析：同文档重传
            self.milvus.delete_source(role_id, source)
            # 解析：删旧子块
            self.milvus.delete_parents_by_source(role_id, source)
            # 解析：删旧父块
            self._text_hashes.pop(role_id, None)
            # 解析：哈希缓存失效
            self._bm25_cache.pop(role_id, None)
            # 解析：BM25 缓存失效

        # 子块：展平 + 去重 + 向量化
        known = self._get_text_hashes(role_id)
        # 解析：已有块哈希
        children = []  # [(parent_id, child_text)]
        # 解析：待入库子块（含父链接）
        for parent_id, _, kids in pairs:
            # 解析：逐父子对
            for kid in kids:
                # 解析：逐子块
                digest = hashlib.sha256(kid.encode("utf-8")).hexdigest()
                # 解析：子块哈希
                if digest not in known:
                    # 解析：不重复
                    children.append((parent_id, kid))
                    # 解析：收录（带父链接）
                    known.add(digest)
                    # 解析：记录哈希
        if not children:
            # 解析：全部重复
            return {"source": source, "chunks": 0}
            # 解析：无操作返回

        chunk_texts = [c for _, c in children]
        # 解析：子块文本列表
        dense_vectors = self.embedder.embed_documents(chunk_texts)
        # 解析：稠密向量化
        corpus = self.milvus.all_texts(role_id) + chunk_texts
        # 解析：BM25 语料
        encoder = BM25Encoder()
        # 解析：新建编码器
        encoder.fit(corpus)
        # 解析：统计词表
        sparse_rows = encoder.encode_texts(chunk_texts)
        # 解析：稀疏向量

        now = int(time.time())
        # 解析：时间戳
        self.milvus.insert_chunks(role_id, [
            # 解析：写入子块（带 parent_id 链接）
            {
                "text": chunk_texts[i],
                "summary": make_summary(chunk_texts[i]),
                "parent_id": children[i][0],
                "dense": dense_vectors[i],
                "sparse": sparse_rows[i],
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
            for i in range(len(children))
            # 解析：逐子块记录
        ])
        self.milvus.insert_parents(role_id, [
            # 解析：写入父块（纯文本集合）
            {
                "parent_id": parent_id,
                "text": parent_text,
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
            for parent_id, parent_text, _ in pairs
            # 解析：逐父块记录
        ])
        self._bm25_cache.pop(role_id, None)
        # 解析：BM25 缓存失效
        return {"source": source, "chunks": len(children)}
        # 解析：返回入库结果（子块数）

    def _chunk(self, text: str) -> list[str]:
        """按 chunking_mode 分块：sentence（句子聚合+重叠）/ heading（标题前置）/ semantic（相似度切分）。"""
        if self.chunking_mode == "heading":
            # 解析：标题模式
            return chunk_by_headings(text, self.chunk_size, self.overlap)
            # 解析：标题前置分块
        if self.chunking_mode == "semantic":
            # 解析：语义模式
            return semantic_chunk(text, self.embedder, self.chunk_size, self.semantic_threshold)
            # 解析：相似度切分
        return chunk_text(text, self.chunk_size, self.overlap)
        # 解析：默认句子聚合分块

    def _get_text_hashes(self, role_id: int) -> set[str]:
        """库内已有块文本的 sha256 集合（惰性构建 + 缓存）。"""
        if role_id not in self._text_hashes:
            # 解析：缓存未命中
            self._text_hashes[role_id] = {
                # 解析：构建哈希集合
                hashlib.sha256(t.encode("utf-8")).hexdigest()
                # 解析：每块文本的哈希
                for t in self.milvus.all_texts(role_id)
                # 解析：遍历库内全部块
            }
        return self._text_hashes[role_id]
        # 解析：返回缓存集合

    # ---------- 检索 ----------

    async def retrieve(
        self, role_id: int, query: str, top_k: int = 4, recall_k: int = 30
    ) -> list[str]:
        """稠密+BM25 混合检索（RRF）召回，重排序取 top_k。

        配置 query_rewriter 时：先把问题改写成多个检索变体（最多 3 个），
        逐变体检索后合并去重再统一重排；改写失败自动回退原问题。
        """
        queries = [query]
        # 解析：查询列表（默认只有原问题）
        if self.query_rewriter is not None:
            # 解析：配置了改写器
            try:
                # 解析：尝试改写
                variants = await self.query_rewriter.rewrite(query)
                # 解析：生成检索变体
                queries.extend(v for v in variants if v and v != query)
                # 解析：合并变体（去空、去原问题重复）
            except Exception:
                # 解析：改写失败
                pass  # 改写失败回退原问题
                # 解析：静默回退（不影响主流程）

        if self.chunking_mode == "parent_child":
            # 解析：父子块模式
            return self._retrieve_parents(role_id, queries, query, top_k, recall_k)
            # 解析：走父子块检索

        candidates, seen = [], set()
        # 解析：候选与去重集合
        for q in queries[:3]:
            # 解析：最多 3 个查询变体
            for text in self._hybrid_candidates(role_id, q, recall_k):
                # 解析：每个变体混合检索召回
                if text not in seen:
                    # 解析：未出现过
                    seen.add(text)
                    # 解析：标记
                    candidates.append(text)
                    # 解析：并入候选
        if not candidates:
            # 解析：无候选
            return []
            # 解析：返回空
        if self.reranker is None:
            # 解析：无重排器
            return candidates[:top_k]
            # 解析：直接截取返回
        return self.reranker.rerank(query, candidates)[:top_k]
        # 解析：重排序后取 Top4

    def _retrieve_parents(
        # 解析：父子块检索
        self, role_id: int, queries: list[str], original_query: str,
        # 解析：角色、查询变体、原问题（重排用）
        top_k: int, recall_k: int,
        # 解析：返回条数与召回条数
    ) -> list[str]:
        """父子块检索：命中子块 → 映射父块 → 去重 → 重排，返回完整父块文本。"""
        name = self.milvus.collection_name(role_id)
        # 解析：子块集合名
        parent_ids, seen = [], set()
        # 解析：父块 ID 列表与去重集合
        for q in queries[:3]:
            # 解析：逐查询变体
            encoder = self._get_bm25(role_id)
            # 解析：取 BM25 编码器
            q_sparse = encoder.encode_query(q)
            # 解析：查询稀疏向量
            q_dense = self.embedder.embed_query(q)
            # 解析：查询稠密向量
            hits = self.milvus.hybrid_search_with_parents(name, q_dense, q_sparse, recall_k)
            # 解析：混合检索（返回 子块文本+父链接 对）
            for _, pid in hits:
                # 解析：逐命中
                if pid and pid not in seen:
                    # 解析：有父链接且未收录
                    seen.add(pid)
                    # 解析：标记
                    parent_ids.append(pid)
                    # 解析：收录父块 ID

        parents = self.milvus.get_parents(role_id, parent_ids)
        # 解析：批量取回父块文本
        texts = [parents[pid] for pid in parent_ids if pid in parents]
        # 解析：按命中顺序映射成文本
        if not texts:
            # 解析：无父块
            return []
            # 解析：返回空
        if self.reranker is None:
            # 解析：无重排器
            return texts[:top_k]
            # 解析：直接截取
        return self.reranker.rerank(original_query, texts)[:top_k]
        # 解析：父块重排取 Top

    def _hybrid_candidates(self, role_id: int, query: str, recall_k: int) -> list[str]:
        """单条查询的混合检索召回。"""
        encoder = self._get_bm25(role_id)
        # 解析：取 BM25 编码器
        query_sparse = encoder.encode_query(query)
        # 解析：稀疏查询向量
        query_dense = self.embedder.embed_query(query)
        # 解析：稠密查询向量
        return self.milvus.hybrid_search(role_id, query_dense, query_sparse, recall_k)
        # 解析：Milvus 双路检索 + RRF 融合

    def _get_bm25(self, role_id: int) -> BM25Encoder:
        # 解析：取 BM25 编码器（带缓存）
        encoder = self._bm25_cache.get(role_id)
        # 解析：查缓存
        if encoder is None:
            # 解析：未命中
            encoder = BM25Encoder()
            # 解析：新建
            encoder.fit(self.milvus.all_texts(role_id))
            # 解析：在该角色全部语料上统计
            self._bm25_cache[role_id] = encoder
            # 解析：写缓存
        return encoder
        # 解析：返回编码器

    # ---------- 管理 ----------

    def list_sources(self, role_id: int) -> list[dict]:
        # 解析：文档列表
        return self.milvus.list_sources(role_id)
        # 解析：透传 Milvus 聚合结果（文档名/块数/摘要/时间）

    def delete_source(self, role_id: int, source: str) -> int:
        # 解析：删除文档
        removed = self.milvus.delete_source(role_id, source)
        # 解析：Milvus 删除并返回删除条数
        self._bm25_cache.pop(role_id, None)
        # 解析：BM25 缓存失效
        self._text_hashes.pop(role_id, None)
        # 解析：哈希缓存失效
        return removed
        # 解析：返回删除条数
