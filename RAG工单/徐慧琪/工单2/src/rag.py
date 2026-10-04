# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：RAG 全链路编排（查询理解 → 混合召回 → 精排 → 父块上下文 → 生成）

链路（每一步计时，供 ≤3 秒验收与瓶颈定位）：
    question
      └─ query_rewrite.analyze      意图/消歧/扩展/子问题         (analyze_ms)
          └─ retriever.search_multi 向量+BM25 多路召回 RRF         (retrieve_ms)
              └─ reranker.rerank    cross-encoder 精排 top-6       (rerank_ms)
                  └─ context.expand_parents 小块→大块 + 压缩       (context_ms)
                      └─ llm.chat    qwen2.5:3b 生成（带 [n] 引用）(gen_ms)

性能保障（工单 ≤3s）：
  - warmup() 预热 embedding/reranker/Ollama（keep_alive 常驻）；
  - 规则版查询改写零额外延迟；全程单次 embedding（query 只编码一次）；
  - LRU 答案缓存：重复问题直接命中；
  - LLM 失败/超时 → 抽取式降级（直接给最相关片段与页码），不返回空白。

对外接口：
  ask(question, ...) -> RagResult
  ask_stream(question, ...) -> Iterator[dict]（事件流：meta → delta → final）
  retrieve(question) -> (hits, analysis, timings)
  warmup() -> dict
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from src import (config, context as ctx_mod, llm, query_rewrite, reranker,
                 retriever)

# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class RagResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)     # 引用页码
    contexts: list[dict] = field(default_factory=list)      # 父块片段
    analysis: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    mode: str = "rag"                                       # rag | extractive | cached
    model: str = ""
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
                "citations": self.citations, "analysis": self.analysis,
                "timings": self.timings, "mode": self.mode, "model": self.model,
                "from_cache": self.from_cache,
                "n_contexts": len(self.contexts)}


# ---------------------------------------------------------------------------
# 提示词（中英双语）
# ---------------------------------------------------------------------------
_SYSTEM_ZH = """你是招股说明书问答助手，只能依据提供的【文档片段】回答。

硬性要求：
1. 片段中只要包含与问题相关的信息，就必须直接作答；只有全部片段都完全不含相关信息时，
   才回答"根据已有资料无法确定"。
2. 数字、比例、金额、名称、日期必须与片段**逐字一致**（含千分位与百分号），
   禁止改写、缩写、推算、取整或调换顺序；多条数据按片段中出现的顺序完整列出。
   若片段中出现"…分别为A、B、C和D"这样的句式，请**整句复制**其中的数字与顺序，
   不重不漏；不要从其他句子（尤其是表格的"小计/合计"行）取数。
3. 只回答被问的指标：片段里常同时出现多个指标（如营业收入总额、毛利率等），
   不要把与问题无关的其他数字列进答案。
4. 关键表述要完整引用（如"军队视频指挥领域的重要供应商"），
   不要简化成上位词（如"国防军队领域"），也不要截断专有名称。
5. 注意区分主体：本文件是"武汉兴图新科电子股份有限公司"的招股说明书；
   文中出现的证券公司、会计师事务所、律师事务所等机构（及其法定代表人、签字人）
   不属于发行人，回答关于发行人的问题时必须忽略这些机构的信息。
6. 若多个片段涉及同一事实但详略不同，以**信息最完整**的那条为准（例如含"由某某牵头、
   联合某某共同制定"的完整句子优于简称），并优先引用该片段。
7. 答案末尾用 [1][2] 标注所依据的片段编号（引用必须真实存在，不要重复标注）。
7. 回答直接给结论，**简洁但完整**：用一句话把"主体 + 指标 + 数值/名称"说清楚
   （如"报告期内，公司来自军用领域的收入分别为A、B、C和D。"），
   不要只甩一串裸数字；不复述问题、不输出分析过程、不使用 Markdown 标题。

示例（注意答案完整保留了片段中的关键表述，且没有添加片段外的数字）：
  片段1：兴图新科目前已经成为国防军队视频指挥领域的重要供应商。
  问：公司在哪个领域已经成为重要供应商？
  答：国防军队视频指挥领域的重要供应商。[1]
  片段2：报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。
  问：公司来自军用领域的收入分别是多少？
  答：6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。[2]"""

_SYSTEM_EN = """You are an assistant answering questions about a prospectus. Answer ONLY from the provided [Document Fragments].

Hard requirements:
1. If the fragments contain information relevant to the question, you MUST answer directly; answer "Cannot be determined from the available materials" ONLY when none of the fragments contain any relevant information.
2. Numbers, percentages, amounts, names and dates must be copied EXACTLY as in the fragments (keep thousands separators, percent signs and original order); never rewrite, abbreviate, round or reorder.
3. Answer only the asked metric: fragments often contain several metrics; do not list numbers irrelevant to the question.
4. Quote key expressions completely (e.g. "军队视频指挥领域的重要供应商"); do not generalize or truncate proper names.
5. Distinguish the subject: this is the prospectus of "Wuhan Xingtu Xinke Electronics Co., Ltd." Information about other parties (securities firm, accounting firm, law firm, their representatives) is NOT the issuer's information and must be ignored.
6. Cite fragment indices like [1][2] at the end; citations must exist and must not repeat.
7. Be direct and concise: no restating the question, no analysis, no Markdown headings.
8. Follow this style (the answer keeps the fragment's exact key wording and adds no outside numbers):
  Fragment 1: The company has become an important supplier in the military video command field.
  Q: In which field has the company become an important supplier?
  A: The military video command field.[1]"""


# ---------------------------------------------------------------------------
# 数字校验（防幻觉）：LLM 偶尔会写出上下文中不存在的数字（如 4,627.14→4,627.15）
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def check_numbers(answer: str, context_text: str) -> tuple[list[str], list[str]]:
    """校验答案中的数字是否都出现在上下文中，并检测列表型答案中的重复数字。

    返回 (答案中的数字列表, 可疑数字列表)。
    两类可疑：
      1. 幻觉 —— 数字在上下文中不存在（如 4,627.14 写成 4,627.15）；
      2. 重复 —— 列表型答案（≥3 个大数字）中同一数字出现多次，几乎必是复制错误
         （实测案例："6,464.51、18,780.67、14,414.16 和 6,464.51"）。
    """
    ctx_nums = {m.replace(",", "") for m in _NUM_RE.findall(context_text or "")}
    ans_nums = _NUM_RE.findall(answer or "")
    bad: list[str] = []
    for n in ans_nums:
        key = n.replace(",", "")
        if key not in ctx_nums and key not in bad:
            bad.append(n)

    big = [n for n in ans_nums if len(n.replace(",", "").replace(".", "")) >= 4]
    if len(set(big)) >= 3:
        for n in dict.fromkeys(big):
            if big.count(n) > 1 and n not in bad:
                bad.append(n)
    return ans_nums, bad


def complete_numeric_list(answer: str, context_text: str) -> tuple[str, bool]:
    """数字列表型答案的抽取式校正（防 3B 模型漏抄/重复）。

    场景（实测）：问"…分别是多少"，片段含 4 个数字，模型输出 3 个（漏第 4 个）
    或重复第 1 个。小模型的逐字复制能力有限，重试也未必纠正。
    做法：若答案包含 ≥3 个大数字（≥4 位数），且上下文中存在一个句子
    覆盖了答案的**全部**数字、并包含更多数字，则用该原句替换答案
    （原文永远比模型转述可靠）。生成失败时返回原答案。
    """
    ans_nums = _NUM_RE.findall(answer or "")
    big = [n for n in ans_nums if len(n.replace(",", "").replace(".", "")) >= 4]
    if len(set(big)) < 3:
        return answer, False
    ans_set = {n.replace(",", "") for n in big}

    best_sent, best_hit = None, 0
    for sent in re.split(r"[。；\n]", context_text or ""):
        s = sent.strip()
        if not (30 <= len(s) <= 170):
            continue
        if "|" in s:                       # 表格行（"小计 | 4,627.15 | …"）不可作答案
            continue
        if s.endswith(("，", "、", ",", "：")):
            continue
        # 去掉结尾的短残句（父块裁剪可能留下"，占"这类尾巴）
        s = re.sub(r"[，,][^，,。；]{0,6}$", "", s).strip()
        if len(s) < 30:
            continue
        s_nums = [n for n in _NUM_RE.findall(s)
                  if len(n.replace(",", "").replace(".", "")) >= 4]
        s_set = {n.replace(",", "") for n in s_nums}
        if not s_set:
            continue
        hit = len(ans_set & s_set)
        # 全覆盖 + 句子数字更全 + 命中最多者胜出
        if hit == len(ans_set) and len(s_set) > len(ans_set) and hit > best_hit:
            best_sent, best_hit = s, hit
    if best_sent:
        return best_sent + "。", True
    return answer, False


def build_messages(question: str, context_text: str, lang: str = "zh") -> list[dict]:
    system = _SYSTEM_EN if lang == "en" else _SYSTEM_ZH
    user = (f"【文档片段】\n{context_text}\n\n【问题】\n{question}"
            if lang != "en" else
            f"[Document Fragments]\n{context_text}\n\n[Question]\n{question}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 检索（含计时）
# ---------------------------------------------------------------------------
def retrieve(question: str, analysis: dict | None = None,
             top_k: int | None = None,
             use_rerank: bool = True) -> tuple[list[dict], dict, dict]:
    """查询理解 + 混合召回 + 精排。返回 (hits, analysis, timings)。"""
    timings: dict = {}

    t0 = time.time()
    analysis = analysis or query_rewrite.analyze(question)
    timings["analyze_ms"] = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    queries = query_rewrite.multi_queries(analysis)
    subs = analysis.get("sub_questions") or []
    if subs:
        # 子问题 + 主查询共同召回：统一 RRF，避免子问题权重被稀释
        hits = retriever.search_multi(queries + subs, top_k=config.VECTOR_TOP_K * 2)
    else:
        hits = retriever.search_multi(queries, top_k=config.VECTOR_TOP_K * 2)
    timings["retrieve_ms"] = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    if use_rerank and hits:
        top_in = hits[: config.RERANK_INPUT_TOP_K]
        hits = reranker.rerank(analysis["resolved"], top_in,
                               top_k=top_k or config.FINAL_TOP_K)
    else:
        hits = hits[: (top_k or config.FINAL_TOP_K)]
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)
    return hits, analysis, timings


def build_qa_context(hits: Sequence[dict]) -> tuple[str, list[dict], list[dict], dict]:
    """命中子块 → (context_text, citation_map, contexts, timings)。"""
    t0 = time.time()
    blocks = ctx_mod.expand_parents(hits, max_chars=config.CONTEXT_MAX_CHARS)
    text = ctx_mod.build_context(blocks)
    cmap = ctx_mod.citation_map(blocks)
    return text, cmap, blocks, {"context_ms": round((time.time() - t0) * 1000, 1)}


# ---------------------------------------------------------------------------
# 引用抽取与校验
# ---------------------------------------------------------------------------
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, cmap: Sequence[dict]) -> list[dict]:
    """从答案抽取 [n] 并映射为页码；无标注时返回空列表（界面回退显示全部片段页码）。"""
    idxs: list[int] = []
    for m in _CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if 1 <= n <= len(cmap) and n not in idxs:
            idxs.append(n)
    out = []
    for n in idxs:
        c = dict(cmap[n - 1])
        c["citation"] = f"[{n}]"
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# 抽取式降级
# ---------------------------------------------------------------------------
def extractive_answer(blocks: Sequence[dict]) -> str:
    """LLM 不可用时的降级答案：直接给出最相关片段原文与页码。"""
    if not blocks:
        return "根据已有资料无法确定。"
    b = blocks[0]
    text = (b.get("parent_text") or "").strip()
    if len(text) > 300:
        text = text[:300] + "…"
    page = int(b.get("page_idx", 0)) + 1
    return f"（LLM 暂不可用，以下为检索到的原文）\n{text}\n\n来源：第{page}页"


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
_cache: "OrderedDict[str, RagResult]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: str) -> RagResult | None:
    with _cache_lock:
        r = _cache.get(key)
        if r is not None:
            _cache.move_to_end(key)
        return r


def _cache_put(key: str, value: RagResult) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > config.ANSWER_CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def ask(question: str, use_cache: bool = True, use_rerank: bool = True,
        top_k: int | None = None) -> RagResult:
    """完整问答（同步）。任何环节异常都会降级而不是抛出。"""
    q = (question or "").strip()
    if not q:
        return RagResult(question="", answer="请输入问题。", mode="extractive")

    if use_cache:
        hit = _cache_get(q)
        if hit is not None:
            cached = RagResult(**{**hit.__dict__, "from_cache": True})
            return cached

    t_start = time.time()
    try:
        hits, analysis, timings = retrieve(q, top_k=top_k, use_rerank=use_rerank)
        context_text, cmap, blocks, t_extra = build_qa_context(hits)
        timings.update(t_extra)
    except Exception as exc:  # noqa: BLE001 —— 检索侧失败：明确报错而不是瞎答
        return RagResult(question=q, answer=f"检索失败：{exc}", mode="error",
                         timings={"total_ms": round((time.time() - t_start) * 1000, 1)})

    lang = analysis.get("lang", "zh")
    if not context_text.strip():
        result = RagResult(question=q, answer="根据已有资料无法确定。",
                           analysis=analysis, contexts=blocks, citations=cmap,
                           mode="rag", timings=timings)
    else:
        try:
            messages = build_messages(q, context_text, lang)
            t0 = time.time()
            resp = llm.chat(messages, temperature=0.0)
            answer = (resp.get("answer") or "").strip()

            # 数字校验：答案中出现片段里没有的数字 → 重试一次（防 LLM 数字幻觉）
            _, bad_nums = check_numbers(answer, context_text)
            if answer and bad_nums:
                retry_messages = messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content":
                        f"你上次回答中的这些数字有问题：{bad_nums}"
                        f"（在片段中不存在，或在列表中重复出现）。"
                        f"请逐字复制片段中的数字（含千分位与小数点），"
                        f"按片段原文顺序完整列出，一个都不能重复、不能遗漏，"
                        f"重新作答，只输出答案本身。"},
                ]
                resp2 = llm.chat(retry_messages, temperature=0.0)
                ans2 = (resp2.get("answer") or "").strip()
                if ans2:
                    _, bad2 = check_numbers(ans2, context_text)
                    if len(bad2) <= len(bad_nums):     # 重试不更差才采用
                        answer = ans2
                        timings["retry"] = True
            timings["gen_ms"] = round((time.time() - t0) * 1000, 1)

            if not answer:
                answer = extractive_answer(blocks)
                mode = "extractive"
            else:
                # 数字列表抽取式校正：小模型漏抄/重复数字时用原文句子替换
                answer, patched = complete_numeric_list(answer, context_text)
                if patched:
                    timings["numeric_patch"] = True
                mode = "rag"
            result = RagResult(question=q, answer=answer, analysis=analysis,
                               contexts=blocks, citations=extract_citations(answer, cmap) or cmap,
                               mode=mode, model=resp.get("model", ""), timings=timings)
        except Exception as exc:  # noqa: BLE001 —— LLM 不可用：抽取式降级
            result = RagResult(question=q, answer=extractive_answer(blocks),
                               analysis=analysis, contexts=blocks, citations=cmap,
                               mode="extractive",
                               timings={**timings, "llm_error": str(exc)})

    timings["total_ms"] = round((time.time() - t_start) * 1000, 1)
    if use_cache:
        _cache_put(q, result)
    return result


def ask_stream(question: str, use_rerank: bool = True,
               top_k: int | None = None) -> Iterator[dict]:
    """流式问答。事件：
    {"type": "meta", ...} → {"type": "delta", "text": ...} → {"type": "final", "result": RagResult}
    """
    q = (question or "").strip()
    if not q:
        yield {"type": "final", "result": RagResult(question="", answer="请输入问题。")}
        return

    t_start = time.time()
    try:
        hits, analysis, timings = retrieve(q, top_k=top_k, use_rerank=use_rerank)
        context_text, cmap, blocks, t_extra = build_qa_context(hits)
        timings.update(t_extra)
    except Exception as exc:  # noqa: BLE001
        yield {"type": "final",
               "result": RagResult(question=q, answer=f"检索失败：{exc}", mode="error")}
        return

    yield {"type": "meta", "analysis": analysis, "citations": cmap,
           "timings": dict(timings)}

    if not context_text.strip():
        result = RagResult(question=q, answer="根据已有资料无法确定。",
                           analysis=analysis, contexts=blocks, citations=cmap,
                           timings=timings)
        yield {"type": "final", "result": result}
        return

    messages = build_messages(q, context_text, analysis.get("lang", "zh"))
    parts: list[str] = []
    t0 = time.time()
    ttft_ms = None
    try:
        for delta in llm.chat_stream(messages):
            if ttft_ms is None:
                ttft_ms = round((time.time() - t_start) * 1000, 1)
            parts.append(delta)
            yield {"type": "delta", "text": delta}
    except Exception as exc:  # noqa: BLE001
        result = RagResult(question=q, answer=extractive_answer(blocks),
                           analysis=analysis, contexts=blocks, citations=cmap,
                           mode="extractive",
                           timings={**timings, "llm_error": str(exc)})
        yield {"type": "final", "result": result}
        return

    timings["gen_ms"] = round((time.time() - t0) * 1000, 1)
    timings["ttft_ms"] = ttft_ms
    timings["total_ms"] = round((time.time() - t_start) * 1000, 1)
    answer = "".join(parts).strip()
    result = RagResult(question=q, answer=answer or extractive_answer(blocks),
                       analysis=analysis, contexts=blocks,
                       citations=extract_citations(answer, cmap) or cmap,
                       mode="rag", timings=timings)
    _cache_put(q, result)
    yield {"type": "final", "result": result}


# ---------------------------------------------------------------------------
# 预热与状态
# ---------------------------------------------------------------------------
def _ensure_gpu_headroom(need_mb: int = 2800) -> None:
    """加载本地模型前检查显存；不足时请求 Ollama 释放（warmup 末尾会重新预热）。

    本机实测：Ollama 常驻 qwen2.5:3b（~3GB）时，新进程加载 bge-reranker-v2-m3
    会因显存竞争触发**进程级段错误**（连 try/except 都来不及执行）。
    因此在加载 embedding/reranker 之前主动让 Ollama 让路。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, _total = torch.cuda.mem_get_info()
        if free / (1024 * 1024) < need_mb:
            print(f"[warmup] 显存剩余 {free / 1024 / 1024:.0f}MB < {need_mb}MB，"
                  f"请求 Ollama 暂时释放…", flush=True)
            from src import llm
            llm.unload()
            torch.cuda.empty_cache()
            time.sleep(2.5)
    except Exception:  # noqa: BLE001
        pass


def warmup() -> dict:
    """服务启动预热：embedding / reranker / BM25 / Ollama 全部就绪，避免首题冷启动。

    顺序很重要：**先加载本地模型（embedding/reranker），最后预热 Ollama**，
    并在加载前用 _ensure_gpu_headroom 确保显存干净（见该函数说明）。
    """
    out: dict = {}
    t0 = time.time()
    _ensure_gpu_headroom()
    try:
        from src import embedder
        embedder.get_model()
        out["embedding"] = True
    except Exception as exc:  # noqa: BLE001
        out["embedding"] = f"失败: {exc}"
    # 再查一次：embedding 已占 ~1.5G，reranker 加载前需要足够的连续显存，
    # 否则实测会触发进程级段错误（Windows/CUDA 边界情况，Python 无法捕获）
    _ensure_gpu_headroom(need_mb=2000)
    try:
        reranker.get_model()
        out["reranker"] = True
    except Exception as exc:  # noqa: BLE001
        out["reranker"] = f"失败: {exc}"
    try:
        retriever.bm25_stats()
        out["bm25"] = True
    except Exception as exc:  # noqa: BLE001
        out["bm25"] = f"失败: {exc}"
    try:
        llm.warmup()
        out["llm"] = True
    except Exception as exc:  # noqa: BLE001
        out["llm"] = f"失败: {exc}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def status() -> dict:
    """系统状态（界面侧栏展示）。"""
    from src import embedder, vector_store
    try:
        from src import reranker as rk
        rk_info = rk.info()
    except Exception:  # noqa: BLE001
        rk_info = {}
    return {
        "workorder": config.WORKORDER_ID,
        "vector_store": vector_store.info(),
        "embedding": embedder.info(),
        "reranker": rk_info,
        "llm": llm.info(),
    }
