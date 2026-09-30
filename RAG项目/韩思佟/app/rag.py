# -*- coding: utf-8 -*-
"""
rag.py -- RAG 引擎层：检索 + Query 改写 + 生成 + 短期记忆

这是早期兼容引擎；当前正式在线答辩入口已经统一为app/single_app.py，
供 FastAPI 接口（main.py）调用。

记忆 key 设计：chat:{user_id}:{role_id}
  —— 同一用户不同角色的记忆互相隔离（对应业务规则 BR5：一轮对话只属于一个角色）
"""

import os
import re
from pathlib import Path

import jieba
from openai import OpenAI, OpenAIError
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import BASE, load_env, setting
from app.memory import ConversationMemory
from app.prompts import compose_doctor_prompt

MODEL_DIR = BASE / "models" / "bge-small-zh-v1.5"
COLLECTION = "doctor_knowledge"
TOP_K = 4
RECALL_N = 10
MAX_TURNS = 10
RERANK_N = 10
DEFAULT_KNOWLEDGE_SCOPE_TERMS = (
    "高血压", "血压", "降压", "收缩压", "舒张压", "脉压",
    "血压计", "家庭血压", "低血压", "白大衣",
)


def postprocess(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"[`*#]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokenize(text: str):
    return [w for w in jieba.cut(text) if w.strip()]


def parse_scope_terms(raw):
    """读取知识库主题词；空值、all 或 * 表示关闭门控。"""
    if raw is None:
        return DEFAULT_KNOWLEDGE_SCOPE_TERMS
    value = str(raw).strip()
    if not value or value.lower() in {"all", "*", "off", "false", "0", "none"}:
        return ()
    return tuple(term.strip().lower() for term in re.split(r"[,，;；|\n]", value) if term.strip())


REWRITE_PROMPT = """下面是一段对话历史和用户最新说的一句话。
请把最新这句话改写成一个意思完整、能独立拿去搜索的问题（补全代词和省略的信息）。
只输出改写后的问题本身，不要任何解释。

对话历史：
{history}

用户最新说的：{question}

改写后的问题："""


class RAGEngine:
    def __init__(self):
        load_env()
        api_key = os.environ.get("LLM_API_KEY", "")
        if not api_key or api_key.startswith(("sk-在这里", "请填写", "your-")):
            raise RuntimeError("未配置 LLM_API_KEY，请先在项目根目录的 .env.local 里填写")

        self.llm = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", ""),
            timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
            max_retries=0,
        )
        self.model = os.environ.get("LLM_MODEL", "")
        self.generation_options = {}
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        thinking = os.environ.get("LLM_ENABLE_THINKING", "false").lower() in {"true", "1", "yes"}
        if provider == "ollama":
            self.generation_options["extra_body"] = {"think": thinking}
        elif provider == "deepseek":
            self.generation_options["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        elif provider in {"vllm", "sglang", "xinference"} and "LLM_ENABLE_THINKING" in os.environ:
            self.generation_options["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": thinking
                }
            }
        self.max_answer_tokens = int(os.environ.get("LLM_MAX_TOKENS", "768"))
        self.embedder = SentenceTransformer(
            str(MODEL_DIR), device=os.environ.get("RAG_EMBEDDING_DEVICE", "cpu"),
            local_files_only=True,
        )
        self.client = MilvusClient(setting("RAG_MILVUS_URI", str(BASE / "db" / "milvus.db")))
        self.client.load_collection(COLLECTION)
        self.min_vector_score = float(os.environ.get("RAG_MIN_VECTOR_SCORE", "0.55"))
        self.knowledge_scope_terms = parse_scope_terms(os.environ.get("RAG_KNOWLEDGE_SCOPE_TERMS"))
        self.rerank_n = int(os.environ.get("RAG_RERANK_N", str(RERANK_N)))
        self.reranker = self._load_reranker()
        self._build_bm25_index()

        self.memory = ConversationMemory(MAX_TURNS)
    # ---------- 记忆 ----------
    def add_memory(self, user_id, role_id, role, content):
        self.memory.add(user_id, role_id, role, content)

    def get_history(self, user_id, role_id):
        return self.memory.history(user_id, role_id)

    def clear_memory(self, user_id, role_id):
        self.memory.clear(user_id, role_id)

    # ---------- 检索 ----------
    def _load_reranker(self):
        enabled = os.environ.get("RAG_RERANK_ENABLED", "auto").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None

        model_dir = Path(os.environ.get(
            "RAG_RERANK_MODEL",
            str(BASE / "models" / "bge-reranker-base"),
        ))
        try:
            if not (model_dir / "config.json").exists():
                raise FileNotFoundError(f"缺少 {model_dir / 'config.json'}")
            return CrossEncoder(
                str(model_dir),
                device=os.environ.get("RAG_EMBEDDING_DEVICE", "cpu"),
                local_files_only=True,
                max_length=512,
            )
        except Exception as exc:
            # 精排模型是增强层，文件未下载完整时继续使用混合检索。
            print(f"RAG 精排未启用：{exc}")
            return None

    def _build_bm25_index(self):
        rows = self.client.query(
            COLLECTION,
            filter="id >= 0",
            output_fields=["id", "text", "chunk_index", "source"],
            limit=2000,
        )
        uniq = {r["id"]: r for r in rows}
        self.rows = sorted(uniq.values(), key=lambda r: r["id"])
        self.by_id = {r["id"]: r for r in self.rows}
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in self.rows]) if self.rows else None

    def _vector_search(self, query, limit=RECALL_N):
        vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        hits = self.client.search(
            COLLECTION,
            data=[vec],
            limit=limit,
            output_fields=["text", "chunk_index", "source"],
        )
        return {
            h["id"]: {
                "id": h["id"],
                "vector_score": float(h["distance"]),
                "entity": h["entity"],
            }
            for h in hits[0]
        }

    def _bm25_search(self, query, limit=RECALL_N):
        if not self.bm25 or not self.rows:
            return {}
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:limit]
        return {
            self.rows[i]["id"]: {
                "id": self.rows[i]["id"],
                "bm25_score": float(score),
                "entity": self.rows[i],
            }
            for i, score in ranked
            if score > 0
        }

    @staticmethod
    def _rrf_score(result_ids, doc_id, k=60):
        if doc_id not in result_ids:
            return 0.0
        return 1.0 / (k + result_ids.index(doc_id) + 1)

    def _in_knowledge_scope(self, query):
        terms = getattr(self, "knowledge_scope_terms", DEFAULT_KNOWLEDGE_SCOPE_TERMS)
        normalized = str(query or "").lower()
        return not terms or any(term in normalized for term in terms)

    def retrieve(self, query, top_k=TOP_K):
        if not self._in_knowledge_scope(query):
            return []
        vector_hits = self._vector_search(query)
        bm25_hits = self._bm25_search(query)
        if not vector_hits and not bm25_hits:
            return []

        best_vector_score = max((h["vector_score"] for h in vector_hits.values()), default=0.0)
        if best_vector_score < self.min_vector_score and not bm25_hits:
            return []

        vector_ids = list(vector_hits.keys())
        bm25_ids = list(bm25_hits.keys())
        fused = []
        for doc_id in set(vector_ids) | set(bm25_ids):
            row = vector_hits.get(doc_id) or bm25_hits[doc_id]
            score = self._rrf_score(vector_ids, doc_id) + self._rrf_score(bm25_ids, doc_id)
            entity = row["entity"]
            fused.append({
                "id": doc_id,
                "rrf_score": score,
                "vector_score": vector_hits.get(doc_id, {}).get("vector_score"),
                "bm25_score": bm25_hits.get(doc_id, {}).get("bm25_score"),
                "entity": {
                    "text": entity["text"],
                    "chunk_index": entity["chunk_index"],
                    "source": entity["source"],
                },
            })

        fused = sorted(fused, key=lambda h: h["rrf_score"], reverse=True)
        candidates = fused[:max(top_k, self.rerank_n)]
        if self.reranker and candidates:
            pairs = [[query, item["entity"]["text"]] for item in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)
            candidates.sort(key=lambda h: h["rerank_score"], reverse=True)
        return candidates[:top_k]

    # ---------- Query 改写 ----------
    def rewrite(self, history, question):
        if "（无历史对话）" in history:
            return question
        try:
            resp = self.llm.chat.completions.create(
                model=self.model, temperature=0,
                max_tokens=128, timeout=20,
                **self.generation_options,
                messages=[{"role": "user",
                           "content": REWRITE_PROMPT.format(history=history, question=question)}],
            )
            return resp.choices[0].message.content.strip().split("\n")[0] or question
        except Exception:
            return question

    # ---------- 主流程 ----------
    def chat(self, user_id, role_id, question, persona_prompt, role_name=""):
        history = self.get_history(user_id, role_id)                 # 1. 取记忆
        search_query = self.rewrite(history, question)               # 2. 改写
        use_knowledge = role_name == "医生"
        hits = self.retrieve(search_query) if use_knowledge else []  # 3. 检索
        if use_knowledge and hits:
            context = "\n\n".join(f"[资料{i}] {h['entity']['text']}" for i, h in enumerate(hits, 1))
        elif use_knowledge:
            context = "（本次检索没有命中足够相关的资料。）"
        else:
            context = "（该角色暂无专属知识库，请按角色人设直接对话；涉及专业结论时说明需要补充对应知识库。）"

        full_prompt = compose_doctor_prompt(
            persona_prompt, history, context, question, bool(hits) if use_knowledge else False
        )

        resp = self.llm.chat.completions.create(
            model=self.model, temperature=0.3,
            max_tokens=self.max_answer_tokens,
            **self.generation_options,
            messages=[{"role": "user", "content": full_prompt}],
        )
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise OpenAIError("Model returned no answer text")
        answer = postprocess(content)
        if use_knowledge and not hits and "当前知识库未覆盖" not in answer:
            answer = "当前知识库未覆盖这个主题，以下是一般健康信息。\n\n" + answer

        self.memory.add_turn(user_id, role_id, question, answer)      # 4. 原子写回一轮记忆

        return {
            "answer": answer,
            "rewritten_query": search_query,
            "sources": [{"chunk_index": h["entity"]["chunk_index"],
                         "source": h["entity"]["source"],
                         "text": h["entity"]["text"],
                         "rrf_score": round(h["rrf_score"], 5),
                         "vector_score": round(h["vector_score"], 4) if h["vector_score"] is not None else None,
                         "bm25_score": round(h["bm25_score"], 3) if h["bm25_score"] is not None else None,
                         "rerank_score": round(h["rerank_score"], 4) if h.get("rerank_score") is not None else None}
                        for h in hits],
        }
