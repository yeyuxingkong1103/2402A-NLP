from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MILVUS_URI, EMB_MODEL_PATH, RERANK_MODEL_PATH
import jieba
import torch
import numpy as np

DIM = 768

class MilvusVectorStore:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = None
        self.emb_model = None
        self.rerank_tokenizer = None
        self.rerank_model = None
        self.bm25 = None
        self.all_texts = []
        self._init_client()
        self._init_models()
        self._load_bm25_from_db()

    def _init_client(self):
        print(f"✅ 连接Milvus：{MILVUS_URI}")
        self.client = MilvusClient(uri=MILVUS_URI)

    def _init_models(self):
        print(f"✅ 加载嵌入模型：{EMB_MODEL_PATH}")
        self.emb_model = SentenceTransformer(EMB_MODEL_PATH, device="cpu")
        print(f"✅ 加载重排序模型：{RERANK_MODEL_PATH}")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH)
        self.rerank_model.eval()
        print("✅ 所有模型加载完成")

    def _load_bm25_from_db(self):
        """从数据库加载已有文本构建BM25索引"""
        try:
            if self.client.has_collection(self.collection_name):
                res = self.client.query(
                    collection_name=self.collection_name,
                    filter="id >= 0",
                    output_fields=["id", "text"],
                    limit=1000
                )
                self.all_texts = [item["text"] for item in res]
                if self.all_texts:
                    tokenized = [list(jieba.cut(t)) for t in self.all_texts]
                    self.bm25 = BM25Okapi(tokenized)
                    print(f"✅ 从数据库加载{len(self.all_texts)}条文本，BM25索引构建完成")
        except Exception as e:
            print(f"⚠️ 加载BM25索引失败：{e}")

    def create_collection(self):
        if self.client.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已存在，跳过创建")
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="IP")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"集合 {self.collection_name} 创建成功")

    def insert_texts(self, text_list):
        emb_result = self.emb_model.encode(text_list)
        data = []
        for text, emb in zip(text_list, emb_result):
            data.append({"text": text, "dense_vector": emb.tolist()})
        res = self.client.insert(collection_name=self.collection_name, data=data)
        print(f"入库成功，插入条数：{res['insert_count']}")
        self.all_texts.extend(text_list)
        tokenized = [list(jieba.cut(t)) for t in self.all_texts]
        self.bm25 = BM25Okapi(tokenized)
        print("✅ BM25关键词索引构建完成")
        return res

    def list_texts(self, limit=100):
        """列出知识库中的所有文本"""
        res = self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["id", "text"],
            limit=limit
        )
        return [{"id": item["id"], "text": item["text"]} for item in res]

    def delete_by_id(self, doc_id):
        """按ID删除文档"""
        res = self.client.delete(
            collection_name=self.collection_name,
            ids=[doc_id]
        )
        # 重新加载BM25
        self._load_bm25_from_db()
        return res

    def delete_by_text(self, text):
        """按文本内容删除（模糊匹配）"""
        all_docs = self.list_texts(limit=1000)
        to_delete = [doc["id"] for doc in all_docs if text in doc["text"]]
        if to_delete:
            self.client.delete(collection_name=self.collection_name, ids=to_delete)
            self._load_bm25_from_db()
        return len(to_delete)

    def dense_search(self, query, top_k=4):
        emb_q = self.emb_model.encode(query).tolist()
        res = self.client.search(
            collection_name=self.collection_name,
            data=[emb_q],
            anns_field="dense_vector",
            limit=top_k,
            output_fields=["text"]
        )
        hit_list = []
        for hits in res:
            for hit in hits:
                hit_list.append(hit["entity"]["text"])
        return hit_list

    def bm25_search(self, query, top_k=4):
        if self.bm25 is None:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.all_texts[i] for i in top_indices]

    def hybrid_search(self, query, top_k=4):
        dense_results = self.dense_search(query, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)
        merged = []
        seen = set()
        for txt in dense_results + bm25_results:
            if txt not in seen:
                seen.add(txt)
                merged.append(txt)
        return merged[:top_k]

    def rerank(self, query, texts, top_k=2):
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors='pt', max_length=512
            )
            scores = self.rerank_model(**inputs).logits.squeeze(-1)
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        reranked = [texts[i] for i in sorted_indices]
        return reranked[:top_k]

    def search_with_rerank(self, query, recall_top_k=4, rerank_top_k=2):
        recalled = self.hybrid_search(query, top_k=recall_top_k)
        print(f"  【混合召回】{len(recalled)}条")
        reranked = self.rerank(query, recalled, top_k=rerank_top_k)
        print(f"  【重排序后】{len(reranked)}条")
        return reranked
