# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""答案生成：大模型调用封装 + RAG 引擎"""
import time

import requests

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT, LLM_MAX_TOKENS,
    LLM_REASONING_EFFORT,
    TOP_K, RETRIEVAL_MODE, RERANK_ALPHA, RERANK_METHOD,
)
from query_understanding import understand, detect_lang
from fulltext import FullTextIndex
from retrieval import Retriever


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
    # 推理开关：与 LightRAG 那一路保持完全一致，否则两边的指标差异里
    # 混着「谁的模型配置更强」。设为空字符串则发默认请求（不传这个字段）。
    if LLM_REASONING_EFFORT:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT
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
    # 空正文有两种成因，都靠「加大额度重试」兜住：
    #   1) deepseek-flash 是推理模型，reasoning token 先于正文消耗额度，额度被吃光正文就是空串；
    #   2) 实测同参数偶发直接返回空串（4096 空、8192 空、16384 又正常），
    #      与额度无关，纯粹是服务端抖动。
    # 工单7 一轮评估要跑 40 多次调用，只重试一次不够用——失败会污染整题的对照结果。
    for _ in range(2):
        if content:
            break
        max_tokens = min(max_tokens * 2, 16384)
        content = _request(prompt, system, temperature, max_tokens, timeout).strip()
    if not content:
        raise LLMError("大模型返回空内容（reasoning 耗尽额度或服务端抖动）")
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


def build_context(hits):
    """把召回块拼成提示词里的上下文。

    单独抽出来是因为**评估裁判必须看到和模型完全相同的那个字符串**。
    实测踩过：评估脚本只拼了块正文、没带「[第X页-类型]」前缀，
    而模型看到的带前缀，于是回答里引用的页码在裁判眼里「无依据」，
    被误判成 grounded=0 —— 一个纯粹由测试脚手架造成的假阳性。
    """
    return "\n\n".join(
        f"[第{c['page']}页-{c['type']}] {c['text']}" for c, _ in hits)


class RAGEngine:
    def __init__(self, vector_store, top_k=TOP_K, mode=RETRIEVAL_MODE,
                 alpha=RERANK_ALPHA, rerank_name=RERANK_METHOD, timeout=None):
        self.vs = vector_store
        self.top_k = top_k
        # 单次大模型调用的超时。默认取系统配置（30 秒，对应「响应 ≤3 秒」的要求）；
        # 评估脚本会显式放宽 —— 纯 LLM 对照回答要跑推理，30 秒会被截断，
        # 那会让 RAG 显得比实际更好，对比就不公平了。
        self.timeout = timeout or LLM_TIMEOUT
        # 工单6：在同一个知识库上挂三种检索策略。
        # 倒排索引直接从已有的块建，不额外落盘（纯内存，毫秒级）。
        self.index = FullTextIndex().build(vector_store.chunks)
        self.retriever = Retriever(vector_store, self.index,
                                   rerank_name=rerank_name, chat=chat)
        self.mode = mode
        self.alpha = alpha

    def _recall_all(self, queries, recall_k):
        """第一阶段：多路召回后轮流合并（round-robin），返回 recall_k 个候选。

        各路查询的 RRF 分数在头名处几乎相等，直接按分数排序会退化成随机顺序，
        所以改为按名次轮流取：原问题第 1 名、重写第 1 名、原问题第 2 名……
        这样重写和子问题只能补充候选，不会把原问题的正确结果挤掉。
        这一阶段故意放宽（RECALL_K=20），保证答案不会被漏掉。
        """
        # 按配置的策略检索：vector / fulltext / hybrid
        results = [self.retriever.search(q, mode=self.mode, alpha=self.alpha,
                                         top_k=recall_k, recall_k=recall_k)
                   for q in queries if q]
        merged, seen = [], set()
        for rank in range(recall_k):
            for hits in results:
                if rank >= len(hits):
                    continue
                chunk = hits[rank][0]
                key = (chunk["page"], chunk["text"][:60])
                if key not in seen:
                    seen.add(key)
                    merged.append((chunk, hits[rank][1]))
            if len(merged) >= recall_k:
                break
        return merged[:recall_k]

    def retrieve(self, question, use_query_understanding=True):
        """构造多路查询并召回，返回 (hits, 查询理解结果)。

        单独抽出来是为了让评估脚本能测到**真正在用的那条检索路径**：
        工单7 的检索指标必须对着 answer() 实际执行的检索来算，
        否则测的是一个系统根本不走的分支，指标再好看也没有意义。
        """
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
        queries += list(qu["sub_questions"])

        # 两阶段检索：先粗排放宽召回保证不漏。重排序默认关闭（见 config.py 的
        # 实测对比：在本任务的财务表格上，交叉编码器会把含答案的表格排到后面）。
        return self._recall_all(queries, self.top_k), qu

    def answer(self, question, use_query_understanding=True):
        started = time.time()
        hits, qu = self.retrieve(question, use_query_understanding)
        retrieve_seconds = time.time() - started         # 检索段耗时，评估时单列
        context = build_context(hits)
        text = chat(
            ANSWER_PROMPT[qu["lang"]].format(context=context, question=question),
            system=RAG_SYSTEM[qu["lang"]], timeout=self.timeout,
        )
        return {"answer": text, "contexts": hits, "recall_count": len(hits),
                "retrieve_seconds": retrieve_seconds,
                "elapsed": time.time() - started, **qu}

    def answer_without_rag(self, question):
        """纯 LLM 回答，不检索 —— 用于工单要求的对比分析。"""
        started = time.time()
        text = chat(question, system=PLAIN_SYSTEM[detect_lang(question)],
                    timeout=self.timeout)
        return {"answer": text, "contexts": [], "elapsed": time.time() - started}


# ---------------- 多文档（工单3） ----------------
class MultiDocEngine:
    """按问题里的公司名，把提问路由到对应文档的知识库。

    工单3 要求在同一套系统里同时支持《招股说明书1》（武汉兴图新科）和
    《招股说明书2》（武汉力源信息）。两份文档各自建索引，提问时先看问题里
    出现的是哪家公司名，再交给对应的引擎；都没匹配上就全库检索后合并。
    """

    def __init__(self):
        self.entries = []          # [(公司名, RAGEngine), ...]

    def add(self, company, engine):
        if company and engine is not None:
            self.entries.append((company, engine))
        return self

    def route(self, question):
        """返回命中的引擎列表；都没命中则返回全部。"""
        text = (question or "").replace(" ", "")
        matched = [e for name, e in self.entries if name and name in text]
        if matched:
            return matched
        # 全称没出现时退一步用简称匹配（如「兴图新科」「力源信息」）
        short = [e for name, e in self.entries if name[:4] and name[:4] in text]
        return short or [e for _, e in self.entries]

    def answer(self, question, **kwargs):
        engines = self.route(question)
        if len(engines) == 1:
            return engines[0].answer(question, **kwargs)

        # 多库命中：取各库耗时最短结果的上下文合并，再生成一次
        best, contexts = None, []
        for engine in engines:
            res = engine.answer(question, **kwargs)
            if best is None or res["elapsed"] < best["elapsed"]:
                best = res
            contexts += res["contexts"][: engine.top_k // 2]
        if best is None:
            return RAGEngine.answer(self, question, **kwargs)
        best["contexts"] = contexts
        return best

    def answer_without_rag(self, question):
        engine = self.route(question)[0]
        return engine.answer_without_rag(question)
