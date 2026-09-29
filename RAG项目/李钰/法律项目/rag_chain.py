# -*- coding: utf-8 -*-
"""
在线问答模块
功能：混合检索（Milvus 稠密 + BM25 稀疏）、RRF 融合、CrossEncoder 重排、
      多轮对话（Redis + Milvus）、角色化 LLM 生成
"""
import json
import pickle
from collections import defaultdict
from functools import lru_cache
import config as C
from logger import get_logger

log = get_logger("rag_chain")

try:
    from role_manager import RoleManager
    _ROLE_MGR = RoleManager()  # 实例化全局角色管理器
except Exception as e:
    log.warning("[RAGChain] RoleManager 未加载，使用默认角色: %s", e)
    _ROLE_MGR = None

from langchain_core.messages import SystemMessage, HumanMessage


# ======================== 中文分词 ========================
def tokenize(text):
    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return [c for c in text if c.strip()]


# ======================== 嵌入与 LLM ========================
@lru_cache(maxsize=1)
def _embedder():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=C.EMBED_MODEL, base_url=C.OLLAMA_URL)


def embed_query(text):
    return _embedder().embed_query(text)


def get_llm(temperature=None):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=C.LLM_MODEL,
        base_url=C.OLLAMA_URL,
        temperature=C.LLM_TEMPERATURE if temperature is None else temperature,
    )


# ======================== Milvus 客户端 ========================
def get_milvus():
    from pymilvus import MilvusClient
    return MilvusClient(C.MILVUS_DB)


# ======================== 混合检索 ========================
def dense_search(client, query_vec, top_k=None):
    top_k = top_k or C.TOP_K_DENSE
    if not client.has_collection(C.COLLECTION_NAME):
        return []
    client.load_collection(C.COLLECTION_NAME)
    results = client.search(
        C.COLLECTION_NAME, data=[query_vec], limit=top_k, # 指定集合、查询向量和返回数量
        output_fields=["text", "source"],
        search_params={"metric_type": C.METRIC_TYPE,
                       "params": {"nprobe": C.IVF_NPROBE}})
    return [{
        "id": h["id"],
        "score": h["distance"],
        "text": h["entity"].get("text", ""),
        "source": h["entity"].get("source", ""),
    } for h in results[0]]


def sparse_search(query, top_k=None):
    top_k = top_k or C.TOP_K_SPARSE
    if not C.BM25_INDEX_PATH.exists(): # 检查 BM25 索引文件是否存在
        return []
    with open(C.BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)  # 反序列化加载索引数据
    bm25, chunks = data["bm25"], data["chunks"]
    scores = bm25.get_scores(tokenize(query))
    # 按分数降序排序，取前 top_k 个索引
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"id": i, "score": scores[i], "text": chunks[i], "source": "BM25"}
            for i in ranked]


def rrf_fusion(dense, sparse, k=None):
    k = k or C.RRF_K
    scores, tmap = defaultdict(float), {}  # scores累计 RRF 分数，tmap保存 key 到原始条目的映射
    for rank, item in enumerate(dense):
        key = item["text"][:100]  # 用文本前 100 个字符作为去重 key
        scores[key] += 1.0 / (k + rank + 1)  # 按 RRF 公式累加稠密检索分数
        tmap[key] = item
    for rank, item in enumerate(sparse):
        key = item["text"][:100]
        scores[key] += 1.0 / (k + rank + 1)  # 累加稀疏检索 RRF 分数
        tmap[key] = item
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tmap[key] for key, _ in merged]

# ======================== 重排序 ========================
_reranker = None  # 全局缓存 CrossEncoder 重排模型，初始为空


def _get_reranker():
    """懒加载并缓存 CrossEncoder，避免每次问答都重新载入模型"""
    global _reranker
    if _reranker is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(C.RERANKER_MODEL, max_length=512)
    return _reranker

def rerank(query, candidates, top_n=None):
    top_n = top_n or C.TOP_K_FINAL
    if not candidates:
        return []
    try:
        model = _get_reranker()
        # 构造查询与候选文本对，最多取配置数量的候选
        pairs = [(query, c["text"]) for c in candidates[:C.RERANK_CANDIDATES]]
        scores = model.predict(pairs)  # 预测每对的相关性分数
        for c, s in zip(candidates[:C.RERANK_CANDIDATES], scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates[:C.RERANK_CANDIDATES],
                        key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]
    except Exception as e:
        log.warning("[Reranker] 降级为原始分数排序: %s", e)
        for c in candidates:
            c["rerank_score"] = c.get("score", 0)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


# ======================== 多轮对话记忆 ========================
class ConversationMemory:
    """Redis 存最近 N 轮，Milvus 存全量历史用于语义召回"""

    def __init__(self):
        self._mem = {}
        self.redis = self._init_redis()
        try:
            self.milvus = get_milvus()
        except Exception:
            self.milvus = None

    def _init_redis(self):
        try:
            import redis
            r = redis.Redis(host=C.REDIS_HOST, port=C.REDIS_PORT, db=C.REDIS_DB)
            r.ping()  #  测试连接
            return r
        except Exception as e:
            log.info("[Redis] 未连接(%s)，降级为内存存储", e)
            return None

    @staticmethod  # 声明静态方法
    def _key(session_id, role_id, user_id):
        return f"chat:{user_id}:{role_id}:{session_id}"

    def add(self, session_id, role, content, role_id="default", user_id="anonymous"):
        entry = {"role": role, "content": content}
        key = self._key(session_id, role_id, user_id)
        if self.redis:
            self.redis.rpush(key, json.dumps(entry, ensure_ascii=False))
            self.redis.ltrim(key, -C.HISTORY_TURNS * 2, -1)
        else:
            self._mem.setdefault(key, []).append(entry) # 降级写入内存字典

        # 同步到 Milvus（用于跨轮语义召回）
        try:
            if self.milvus and self.milvus.has_collection(C.HISTORY_COLLECTION):
                vec = embed_query(content)
                self.milvus.insert(C.HISTORY_COLLECTION, [{
                    "embedding": vec,
                    "session_id": session_id[:64],
                    "role": role[:16],
                    "content": content[:2048],
                }])
        except Exception as e:
            log.warning("[Milvus历史] 写入失败: %s", e)

    def get_history(self, session_id, n=None, role_id="default", user_id="anonymous"):
        n = n or C.HISTORY_TURNS
        key = self._key(session_id, role_id, user_id)
        if self.redis:
            raw = self.redis.lrange(key, -n * 2, -1)
            return [json.loads(r) for r in raw]
        return self._mem.get(key, [])[-n * 2:]

    def semantic_recall(self, session_id, query, top_k=None,
                        role_id="default", user_id="anonymous"):
        top_k = top_k or C.HISTORY_RECALL_K
        try:
            if not self.milvus or not self.milvus.has_collection(C.HISTORY_COLLECTION):
                return []
            self.milvus.load_collection(C.HISTORY_COLLECTION)
            vec = embed_query(query)
            res = self.milvus.search(
                C.HISTORY_COLLECTION, data=[vec], limit=top_k,
                filter=f'session_id == "{session_id[:64]}"',
                output_fields=["role", "content"],
                search_params={"metric_type": C.METRIC_TYPE})
            return [{"role": h["entity"]["role"],
                     "content": h["entity"]["content"]} for h in res[0]]
        except Exception as e:
            log.debug("[Milvus历史] 语义召回失败: %s", e)
            return []


# ======================== 提示词模板 ========================
SYSTEM_TMPL = """你正在扮演：{name}
角色类型：{type}
身份设定：{persona}
性格：{personality}
说话风格：{speaking_style}
专业领域：{expertise}
目标：{goals}
边界：{boundaries}

必须遵守：
1. 事实优先来自<参考资料>；资料没有就说"我掌握的资料里没有"。
2. 不编造法条、案号、判例。
3. 保持角色口吻，不要自称 AI 或语言模型。
4. 法律建议不替代正式律师；重大事项建议线下咨询。
"""

USER_TMPL = """<参考资料>
{context}
</参考资料>

<对话历史>
{history}
</对话历史>

用户说：{query}
请以{name}的身份回答。"""


def build_messages(role, query, docs, history):
    """把角色卡 + 检索结果 + 对话历史组装为 Chat messages"""
    system = SYSTEM_TMPL.format(
        name=role.get("name", "助手"),
        type=role.get("type", ""),
        persona=role.get("persona", ""),
        personality=role.get("personality", ""),
        speaking_style=role.get("speaking_style", ""),
        expertise=role.get("expertise", ""),
        goals="；".join(role.get("goals", []) or []),
        boundaries="；".join(role.get("boundaries", []) or []),
    )
    context = "\n---\n".join(
        f"[{i + 1}] 来源:{d.get('source', '')}\n{d.get('text', '')[:800]}"
        for i, d in enumerate(docs)
    )
    hist = ""
    for h in history:
        who = "用户" if h.get("role") == "user" else role.get("name", "助手")
        hist += f"{who}: {h.get('content', '')[:200]}\n"

    user = USER_TMPL.format(
        context=context,
        history=hist,
        query=query,
        name=role.get("name", "助手"),
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def generate(messages, temperature=None):
    llm = get_llm(temperature)
    resp = llm.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)


# ======================== RAG 链主类 ========================
_FALLBACK_ROLE = {
    "id": "default", "name": "AI助手", "type": "通用助手",
    "persona": "你是一个乐于助人的AI助手。",
    "personality": "友好、专业", "speaking_style": "简洁清晰",
    "expertise": "通用", "goals": [], "boundaries": [],
    "temperature": C.LLM_TEMPERATURE,
    "rag_top_k": C.TOP_K_FINAL, "disclaimer": "",
}


class RAGChain:
    """RAG 问答链：检索 → 融合 → 重排 → 角色化生成"""

    def __init__(self):
        self.milvus = get_milvus()
        self.memory = ConversationMemory()
        log.info("[RAGChain] 初始化完成")

    @staticmethod
    def _get_role(role_id):
        if _ROLE_MGR is not None:
            try:
                return _ROLE_MGR.get(role_id or "default")
            except Exception:
                pass
        return _FALLBACK_ROLE

    def invoke(self, query, session_id="default",
               role_id="lawyer_friend", user_id="anonymous"):
        import time
        t0 = time.time()
        role = self._get_role(role_id)
        top_k = int(role.get("rag_top_k") or C.TOP_K_FINAL)
        log.info("[invoke] user=%s role=%s sid=%s q=%s",
                 user_id, role.get("id", role_id), session_id, query[:120])

        # 1. 查询向量化 + 混合检索
        query_vec = embed_query(query)
        dense = dense_search(self.milvus, query_vec, top_k=C.TOP_K_DENSE) # 稠密检索
        sparse = sparse_search(query, top_k=C.TOP_K_SPARSE) # 稀疏 BM25 检索
        log.info("[invoke] 检索命中 dense=%d sparse=%d", len(dense), len(sparse))

        # 2. RRF 融合 + 重排
        merged = rrf_fusion(dense, sparse)
        reranked = rerank(query, merged, top_n=top_k)

        # 3. 历史（语义召回 + 近期轮次）
        recalled = self.memory.semantic_recall(
            session_id, query, role_id=role_id, user_id=user_id)
        recent = self.memory.get_history(
            session_id, role_id=role_id, user_id=user_id)
        history = recalled + recent

        # 4. 角色化生成
        messages = build_messages(role, query, reranked, history)
        answer = generate(messages, temperature=role.get("temperature"))

        # 5. 附加免责声明
        disclaimer = role.get("disclaimer") or ""
        if disclaimer and disclaimer not in answer:
            answer = answer.rstrip() + "\n\n" + disclaimer

        # 6. 记录本轮对话
        self.memory.add(session_id, "user", query,
                        role_id=role_id, user_id=user_id)
        self.memory.add(session_id, "assistant", answer,
                        role_id=role_id, user_id=user_id)

        dt = time.time() - t0
        log.info("[invoke] 完成 dt=%.2fs ans_len=%d sources=%d recall=%d",
                 dt, len(answer), len(reranked), len(recalled))

        return {
            "answer": answer,
            "role": {"id": role.get("id"), "name": role.get("name")},
            "sources": [
                {
                    "text": d.get("text", "")[:200],
                    "score": float(d.get("rerank_score", d.get("score", 0))),
                    "source": d.get("source", ""),
                }
                for d in reranked
            ],
            "history_recall": len(recalled),
        }