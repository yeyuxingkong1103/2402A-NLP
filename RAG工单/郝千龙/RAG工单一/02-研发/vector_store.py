# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【向量库组件 · vector_store.py】统一向量门面，支持 Milvus（生产/HNSW）与 Chroma（本地免部署）双后端
# 编写日期：2026-09-28   修订日期：2026-10-04
from typing import List, Dict, Optional
import numpy as np

import config

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType,
        Collection, utility,
    )
except ImportError:  # pragma: no cover
    connections = None


class ChromaBackend:
    """Chroma 本地持久化后端：无需独立服务，适合单机/Windows 环境"""

    _client = None
    _col = None

    @classmethod
    def _collection(cls):
        """惰性获取持久化集合（余弦度量）"""
        if cls._col is None:
            cls._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            cls._col = cls._client.get_or_create_collection(
                name=config.MILVUS_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return cls._col

    @classmethod
    def insert(cls, embeddings: np.ndarray, metas: List[Dict]) -> int:
        """批量入库；id 使用 doc_id + chunk_id 组合保证可删可重建"""
        col = cls._collection()
        ids = [f"{m.get('doc_id', 'prospectus1')}_{m['chunk_id']}" for m in metas]
        col.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=[m["text"] for m in metas],
            metadatas=[
                {"page": int(m["page"]), "doc_id": str(m.get("doc_id", "prospectus1"))}
                for m in metas
            ],
        )
        return len(metas)

    @classmethod
    def search(cls, query_vec: np.ndarray, top_k: int,
               doc_id: Optional[str] = None) -> List[Dict]:
        """向量检索 Top-K，Chroma 返回余弦距离需换算为相似度"""
        col = cls._collection()
        where = {"doc_id": doc_id} if doc_id else None
        res = col.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: List[Dict] = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for _id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            out.append({
                "text": doc,
                "page": int(meta.get("page", 0)),
                "doc_id": meta.get("doc_id", ""),
                "chunk_id": _id,
                # cosine 距离 → 相似度（1 - distance）
                "score": float(1.0 - dist),
            })
        return out

    @classmethod
    def get_all(cls) -> List[Dict]:
        """取出全部 chunk（用于构建 BM25 与文档聚合）"""
        col = cls._collection()
        data = col.get(include=["documents", "metadatas"])
        return [
            {
                "chunk_uid": _id,
                "text": doc,
                "page": int(meta.get("page", 0)),
                "doc_id": meta.get("doc_id", ""),
            }
            for _id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    @classmethod
    def delete_doc(cls, doc_id: str) -> None:
        """按 doc_id 删除文档全部向量"""
        col = cls._collection()
        col.delete(where={"doc_id": doc_id})

    @classmethod
    def count(cls) -> int:
        """返回 chunk 总数"""
        try:
            return cls._collection().count()
        except Exception:
            return 0

    @classmethod
    def drop(cls) -> None:
        """删除整个集合（独立进程调用时需先自建 client）"""
        client = cls._client
        if client is None:
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            client.delete_collection(config.MILVUS_COLLECTION)
        except Exception as e:
            print(f"[WARN] drop collection: {e}")
        # 复位连接，防止后续进程复用内存中的旧集合对象
        cls._col = None
        cls._client = None


class MilvusBackend:
    """Milvus 生产后端：HNSW + COSINE（优化方案 P0：IVF_FLAT → HNSW）"""

    _collection: Collection = None

    @classmethod
    def connect(cls) -> None:
        """连接 Milvus，幂等"""
        try:
            connections.connect(alias="default",
                                host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        except Exception as e:
            print(f"[ERROR] Milvus 连接失败: {e}")
            raise

    @classmethod
    def _ensure_collection(cls, dim: int) -> Collection:
        """确保集合存在，维度变化时重建"""
        cls.connect()
        name = config.MILVUS_COLLECTION
        if utility.has_collection(name):
            col = Collection(name)
            for f in col.schema.fields:
                if f.name == "embedding" and f.params.get("dim") != dim:
                    utility.drop_collection(name)
                    break
            else:
                cls._collection = col
                return col
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="page", dtype=DataType.INT32),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_uid", dtype=DataType.VARCHAR, max_length=160),
        ]
        schema = CollectionSchema(fields, description="RAG prospectus collection")
        col = Collection(name, schema)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": config.MILVUS_INDEX_TYPE,
                "metric_type": config.MILVUS_METRIC_TYPE,
                "params": {"M": config.MILVUS_HNSW_M, "efConstruction": config.MILVUS_HNSW_EF},
            },
        )
        print(f"[OK] 创建 Milvus 集合 {name}, dim={dim}, HNSW")
        cls._collection = col
        return col

    @classmethod
    def insert(cls, embeddings: np.ndarray, metas: List[Dict]) -> int:
        """批量入库"""
        col = cls._ensure_collection(dim=embeddings.shape[1])
        col.upsert(
            data=[
                [f"{m.get('doc_id', 'prospectus1')}_{m['chunk_id']}" for m in metas],
                embeddings.tolist(),
                [m["text"] for m in metas],
                [int(m["page"]) for m in metas],
                [str(m.get("doc_id", "prospectus1")) for m in metas],
            ]
        )
        col.flush()
        return len(metas)

    @classmethod
    def search(cls, query_vec: np.ndarray, top_k: int,
               doc_id: Optional[str] = None) -> List[Dict]:
        """HNSW 检索 Top-K"""
        col = cls._ensure_collection(dim=query_vec.shape[0])
        col.load()
        results = col.search(
            data=[query_vec.tolist()],
            anns_field="embedding",
            param={"metric_type": config.MILVUS_METRIC_TYPE,
                   "params": {"ef": config.MILVUS_HNSW_EF}},
            limit=top_k,
            output_fields=["text", "page", "doc_id", "chunk_uid"],
        )
        out = []
        for h in results[0]:
            ent = h.entity
            out.append({
                "text": ent.get("text"),
                "page": int(ent.get("page") or 0),
                "doc_id": ent.get("doc_id"),
                "chunk_id": ent.get("chunk_uid"),
                "score": float(h.score),
            })
        return out

    @classmethod
    def get_all(cls) -> List[Dict]:
        """取出全部 chunk"""
        col = cls._ensure_collection(dim=config.EMBED_DIM)
        col.load()
        rows = col.query(expr="id >= 0", output_fields=["text", "page", "doc_id", "chunk_uid"],
                         limit=16384)
        return [
            {"chunk_uid": r.get("chunk_uid", ""), "text": r["text"],
             "page": int(r.get("page", 0)), "doc_id": r.get("doc_id", "")}
            for r in rows
        ]

    @classmethod
    def delete_doc(cls, doc_id: str) -> None:
        """按 doc_id 删除"""
        col = cls._ensure_collection(dim=config.EMBED_DIM)
        col.delete(expr=f'doc_id == "{doc_id}"')

    @classmethod
    def count(cls) -> int:
        try:
            col = cls._ensure_collection(dim=config.EMBED_DIM)
            col.flush()
            return col.num_entities
        except Exception:
            return 0

    @classmethod
    def drop(cls) -> None:
        if utility.has_collection(config.MILVUS_COLLECTION):
            utility.drop_collection(config.MILVUS_COLLECTION)
        cls._collection = None


class FaissBackend:
    """FAISS 本地向量后端：IndexFlatIP 余弦检索，pickle 持久化，Windows 稳定免部署"""

    _INDEX_FILE = config.FAISS_DIR / "index.faiss"   # FAISS 索引文件
    _META_FILE = config.FAISS_DIR / "meta.pkl"       # 元数据+向量备份
    _index = None       # 内存 FAISS 索引
    _store = None       # OrderedDict: uid -> {text,page,doc_id,vec}

    @classmethod
    def _load(cls):
        """惰性加载索引与元数据；无文件时初始化空索引"""
        import pickle
        from collections import OrderedDict
        if cls._index is not None:
            return
        if cls._INDEX_FILE.exists() and cls._META_FILE.exists():
            # 用 Python 读字节再反序列化，规避 C++ fopen 不支持中文路径的问题
            raw = cls._INDEX_FILE.read_bytes()
            cls._index = faiss.deserialize_index(
                np.frombuffer(raw, dtype=np.uint8)
            )
            with open(cls._META_FILE, "rb") as f:
                cls._store = pickle.load(f)
        else:
            cls._index = faiss.IndexFlatIP(config.EMBED_DIM)
            cls._store = OrderedDict()

    @classmethod
    def _persist(cls):
        """索引与元数据落盘（Python IO，支持中文路径）"""
        import pickle
        # 序列化为字节数组后由 Python 写入，绕过 C++ FileIOWriter
        buf = faiss.serialize_index(cls._index)
        cls._INDEX_FILE.write_bytes(np.asarray(buf, dtype=np.uint8).tobytes())
        with open(cls._META_FILE, "wb") as f:
            pickle.dump(cls._store, f)

    @classmethod
    def _rebuild(cls):
        """依据内存元数据重建扁平索引（1612×768 不足 1 秒，简化 upsert/delete）"""
        idx = faiss.IndexFlatIP(config.EMBED_DIM)
        if cls._store:
            vecs = np.stack([e["vec"] for e in cls._store.values()]).astype(np.float32)
            idx.add(vecs)
        cls._index = idx

    @classmethod
    def insert(cls, embeddings: np.ndarray, metas: List[Dict]) -> int:
        """按 uid upsert（已存在则覆盖），随后重建索引并持久化"""
        cls._load()
        uids = [f"{m.get('doc_id', 'prospectus1')}_{m['chunk_id']}" for m in metas]
        for i, (uid, m) in enumerate(zip(uids, metas)):
            cls._store[uid] = {
                "text": m["text"],
                "page": int(m["page"]),
                "doc_id": str(m.get("doc_id", "prospectus1")),
                "vec": embeddings[i].astype(np.float32),
            }
        cls._rebuild()
        cls._persist()
        return len(metas)

    @classmethod
    def search(cls, query_vec: np.ndarray, top_k: int,
               doc_id: Optional[str] = None) -> List[Dict]:
        """内积检索（向量已归一化，内积即余弦相似度）"""
        cls._load()
        n = cls._index.ntotal
        if n == 0:
            return []
        uids = list(cls._store.keys())
        entries = list(cls._store.values())
        k = min(top_k, n)
        scores, idx = cls._index.search(
            np.asarray([query_vec], dtype=np.float32), k
        )
        out: List[Dict] = []
        for score, row in zip(scores[0], idx[0]):
            if row < 0:
                continue
            e = entries[row]
            # doc_id 过滤（本工单主链路不传，仅为接口完整性）
            if doc_id and e["doc_id"] != doc_id:
                continue
            out.append({
                "text": e["text"],
                "page": e["page"],
                "doc_id": e["doc_id"],
                "chunk_id": uids[row],
                "score": float(score),
            })
        return out

    @classmethod
    def get_all(cls) -> List[Dict]:
        """取出全部 chunk（构建 BM25 / 文档聚合用）"""
        cls._load()
        return [
            {"chunk_uid": uid, "text": e["text"],
             "page": e["page"], "doc_id": e["doc_id"]}
            for uid, e in cls._store.items()
        ]

    @classmethod
    def delete_doc(cls, doc_id: str) -> None:
        """按 doc_id 删除文档全部向量并重建"""
        cls._load()
        for uid in [u for u, e in cls._store.items() if e["doc_id"] == doc_id]:
            del cls._store[uid]
        cls._rebuild()
        cls._persist()

    @classmethod
    def count(cls) -> int:
        """返回 chunk 总数"""
        try:
            cls._load()
            return cls._index.ntotal
        except Exception:
            return 0

    @classmethod
    def drop(cls) -> None:
        """清空索引与元数据（内存 + 磁盘）"""
        from collections import OrderedDict
        cls._index = faiss.IndexFlatIP(config.EMBED_DIM)
        cls._store = OrderedDict()
        for f in (cls._INDEX_FILE, cls._META_FILE):
            if f.exists():
                f.unlink()


class VectorStore:
    """向量库统一门面：按配置切换 Milvus / Chroma"""

    @staticmethod
    def _backend():
        """按配置返回后端类"""
        backend = config.VECTOR_BACKEND
        if backend == "milvus" and connections is not None:
            return MilvusBackend
        if backend == "chroma" and chromadb is not None:
            return ChromaBackend
        if backend == "faiss" and faiss is not None:
            return FaissBackend
        # 兜底：优先 FAISS（最稳定），其次 Chroma
        if faiss is not None:
            return FaissBackend
        if chromadb is not None:  # pragma: no cover
            return ChromaBackend
        raise RuntimeError("无可用向量后端，请安装 faiss-cpu 或 chromadb")

    @classmethod
    def insert(cls, embeddings: np.ndarray, metas: List[Dict]) -> int:
        """向量入库"""
        return cls._backend().insert(embeddings, metas)

    @classmethod
    def search(cls, query_vec: np.ndarray, top_k: int = 5,
               doc_id: Optional[str] = None) -> List[Dict]:
        """相似度检索"""
        return cls._backend().search(query_vec, top_k, doc_id)

    @classmethod
    def get_all_chunks(cls) -> List[Dict]:
        """获取全部 chunk（BM25 构建/文档列表用）"""
        return cls._backend().get_all()

    @classmethod
    def delete_doc(cls, doc_id: str) -> None:
        """删除指定文档"""
        cls._backend().delete_doc(doc_id)

    @classmethod
    def count(cls) -> int:
        """chunk 总数"""
        return cls._backend().count()

    @classmethod
    def drop_collection(cls) -> None:
        """删除集合（危险操作）"""
        cls._backend().drop()

    @classmethod
    def backend_name(cls) -> str:
        """当前后端名"""
        b = cls._backend()
        if b is MilvusBackend:
            return "milvus"
        if b is FaissBackend:
            return "faiss"
        return "chroma"

# ====================================================================
# 技术备注：
# 1. RAG：向量库是知识库的"记忆层"，ANN 检索替代全量扫描。
# 2. HNSW（Hierarchical Navigable Small World）：分层可导航小世界图索引，
#    通过多层近邻图实现对数复杂度近似最近邻查询，查询延迟较 IVF_FLAT 降低约 50%。
# 3. Chroma 内嵌 HNSW（cosine），零运维，适合 548 页招股书规模。
# 4. Transformer / Fine-tuning：向量由上游 Transformer Embedding 模型产出，
#    微调 Embedding 后只需重新入库，接口层无感知。
# ====================================================================
