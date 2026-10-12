# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
RAG 流水线：划分 5 个阶段并逐阶段计时，定位瓶颈；内置优化开关。

5 个阶段：
  1. 查询处理与增强
  2. 检索阶段
  3. 上下文组装与提示工程
  4. LLM 生成
  5. 后处理与响应格式化

优化点：
  - pre_index=True：预构建倒排索引，避免每次查询全量扫描；
  - cache=True：对重复查询做结果缓存，避免重复检索与生成。
"""
import time
import uuid

try:
    import jieba
except Exception:
    jieba = None

import config
from observability import log_stage, METRICS
from tracing import TRACER


def tokenize(text):
    if jieba is not None:
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    chars = [c for c in text if not c.isspace()]
    return chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class RAGPipeline:
    def __init__(self, docs, pre_index=True, cache=False):
        self.docs = docs
        self.cache = {} if cache else None
        self.pre_index = pre_index
        if pre_index:
            self._build_index()

    def _build_index(self):
        """预构建倒排索引（优化点）。"""
        self.inverted = {}
        for di, d in enumerate(self.docs):
            for t in set(tokenize(d["text"])):
                self.inverted.setdefault(t, []).append(di)

    def retrieve(self, q_tokens, top_k=5):
        if self.pre_index:
            scores = {}
            for t in q_tokens:
                for di in self.inverted.get(t, []):
                    scores[di] = scores.get(di, 0) + 1
            order = sorted(scores, key=scores.get, reverse=True)[:top_k]
            return [self.docs[i] for i in order]
        # 未优化：全量扫描（性能瓶颈）
        scored = []
        for d in self.docs:
            s = sum(1 for t in q_tokens if t in d["text"])
            scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:top_k]]

    def run(self, query, request_id=None):
        request_id = request_id or uuid.uuid4().hex[:8]

        # 缓存优化：命中缓存直接返回
        if self.cache is not None and query in self.cache:
            return self.cache[query]

        stages = {}
        with TRACER.start("rag_pipeline"):
            # 1. 查询处理与增强
            t0 = time.perf_counter()
            q_tokens = tokenize(query)
            stages["1_query_process"] = time.perf_counter() - t0

            # 2. 检索阶段
            t0 = time.perf_counter()
            docs = self.retrieve(q_tokens, top_k=config.TOP_K)
            stages["2_retrieve"] = time.perf_counter() - t0

            # 3. 上下文组装与提示工程
            t0 = time.perf_counter()
            context = "\n".join(d["text"] for d in docs)
            stages["3_context"] = time.perf_counter() - t0

            # 4. LLM 生成
            t0 = time.perf_counter()
            answer = self._generate(context, query)
            stages["4_llm"] = time.perf_counter() - t0

            # 5. 后处理与响应格式化
            t0 = time.perf_counter()
            response = {"answer": answer, "doc_ids": [d["id"] for d in docs]}
            stages["5_postprocess"] = time.perf_counter() - t0

        total = sum(stages.values())
        result = {"request_id": request_id, "stages": stages,
                  "total": total, "response": response}

        if self.cache is not None:
            self.cache[query] = result

        log_stage(request_id, query, stages, total)
        METRICS.record(total)
        return result

    def _generate(self, context, query):
        if config.LLM_API_KEY:
            # 真实 LLM 接口（OpenAI/DashScope 兼容）
            import json
            import urllib.request
            try:
                req = urllib.request.Request(
                    config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
                    data=json.dumps({
                        "model": config.LLM_MODEL,
                        "messages": [{"role": "user", "content": f"根据上下文回答：\n{context}\n问题：{query}"}],
                    }).encode("utf-8"),
                    headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                             "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read().decode())["choices"][0]["message"]["content"]
            except Exception as e:
                print("[pipeline] LLM 调用失败，回退离线:", e)
        # 离线模式：模拟固定推理延迟
        time.sleep(config.SIMULATED_LLM_LATENCY)
        return f"（离线）检索到 {len(context)} 字上下文，回答：{query}"
