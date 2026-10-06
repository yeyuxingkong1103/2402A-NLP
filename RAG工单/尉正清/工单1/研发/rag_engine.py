# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""答案生成：大模型调用封装 + RAG 引擎"""
import time

import requests

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT, LLM_MAX_TOKENS, TOP_K,
)
from query_understanding import understand, detect_lang


# ---------------- 大模型调用 ----------------
class LLMError(RuntimeError):
    """调用失败时抛出，避免把错误文案当成答案传给界面。"""


def _request(prompt, system, temperature, max_tokens, timeout):
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(
            f"{LLM_API_BASE.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""
    except requests.exceptions.Timeout as exc:
        raise LLMError(f"大模型调用超时（>{timeout}s）") from exc
    except requests.exceptions.HTTPError as exc:
        raise LLMError(f"大模型返回 HTTP {resp.status_code}：{resp.text[:200]}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"大模型返回格式异常：{exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"大模型请求失败：{exc}") from exc


def chat(prompt, system="你是一个严谨的文档问答助手。", temperature=0.0,
         max_tokens=LLM_MAX_TOKENS, timeout=LLM_TIMEOUT):
    if not LLM_API_BASE or not LLM_API_KEY:
        raise LLMError("未配置 DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY 环境变量")

    content = _request(prompt, system, temperature, max_tokens, timeout).strip()
    if not content and max_tokens < 8192:
        # deepseek-flash 是推理模型，reasoning token 先于正文消耗额度。
        # 若额度被 reasoning 吃光，正文会是空串——放大额度重试一次。
        content = _request(prompt, system, temperature, max_tokens * 2, timeout).strip()
    if not content:
        raise LLMError("大模型返回空内容（reasoning 可能耗尽额度）")
    return content


# ---------------- RAG 引擎 ----------------
RAG_SYSTEM = {
    "zh": "你是一个严谨的文档问答助手，只能依据提供的上下文回答，不得编造。",
    "en": "You are a rigorous document Q&A assistant. Answer ONLY from the given context.",
}
ANSWER_PROMPT = {
    "zh": "请依据下列上下文回答问题，准确、简洁，并注明依据的页码。\n"
          "若上下文中确实找不到答案，直接回复「根据提供的上下文无法回答」，不要反复推敲。\n\n"
          "【上下文】\n{context}\n\n【问题】\n{question}\n\n【回答】\n",
    # 最后那句很重要：检索没命中时，模型会陷入长时间推理（实测单题 11 秒），
    # 明确允许它放弃可以把这个循环打断，响应时间回落到 2 秒左右。
    "en": "Answer the question using ONLY the context below. Be accurate, concise, "
          "and cite the page number(s) you used.\n"
          "If the answer is genuinely absent from the context, reply "
          "\"Cannot answer from the provided context\" without further deliberation.\n\n"
          "[Context]\n{context}\n\n[Question]\n{question}\n\n[Answer]\n",
}
PLAIN_SYSTEM = {
    "zh": "你是一个知识渊博的助手。",
    "en": "You are a knowledgeable assistant.",
}


class RAGEngine:
    def __init__(self, vector_store, top_k=TOP_K):
        self.vs = vector_store
        self.top_k = top_k

    def _retrieve_all(self, queries, top_k):
        """多路召回后轮流合并（round-robin）。

        各路查询的 RRF 分数在头名处几乎相等，直接按分数排序会退化成随机顺序，
        所以改为按名次轮流取：原问题第 1 名、重写第 1 名、原问题第 2 名……
        这样重写和子问题只能补充候选，不会把原问题的正确结果挤掉。
        """
        results = [self.vs.search(q, top_k=top_k) for q in queries if q]
        merged, seen = [], set()
        for rank in range(top_k):
            for hits in results:
                if rank >= len(hits):
                    continue
                chunk = hits[rank][0]
                key = (chunk["page"], chunk["text"][:60])
                if key not in seen:
                    seen.add(key)
                    merged.append((chunk, hits[rank][1]))
        return merged[:top_k]

    def answer(self, question, use_query_understanding=True):
        started = time.time()
        qu = understand(question) if use_query_understanding else {
            "lang": detect_lang(question), "intent": "", "rewrite": question,
            "sub_questions": [], "ambiguous": [], "clarifications": [],
            "entities": [], "numbers": [],
        }
        # 问题里带了实体（公司名等）时以原问题为主，重写只作补充召回；
        # 没带实体时（例如只问「注册资本是多少？」）改为重写优先，
        # 否则会检索到子公司的同名科目 —— 实测英文提问会答成子公司的 100 万元。
        if qu["entities"]:
            queries = [question, qu["rewrite"]]
        else:
            queries = [qu["rewrite"], question]
        hits = self._retrieve_all(queries + list(qu["sub_questions"]), self.top_k)
        context = "\n\n".join(
            f"[第{c['page']}页-{c['type']}] {c['text']}" for c, _ in hits
        )
        text = chat(
            ANSWER_PROMPT[qu["lang"]].format(context=context, question=question),
            system=RAG_SYSTEM[qu["lang"]],
        )
        return {"answer": text, "contexts": hits, "elapsed": time.time() - started, **qu}

    def answer_without_rag(self, question):
        """纯 LLM 回答，不检索 —— 用于工单要求的对比分析。"""
        started = time.time()
        text = chat(question, system=PLAIN_SYSTEM[detect_lang(question)])
        return {"answer": text, "contexts": [], "elapsed": time.time() - started}
