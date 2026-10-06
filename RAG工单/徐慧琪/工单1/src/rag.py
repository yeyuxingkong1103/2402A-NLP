# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：问答引擎（Query 理解 → 混合检索 → 精排 → 生成 → 引用）

工单功能点对应：
  （1）Query 理解
        · 意图识别 : fact_lookup / enumerate / compare / calculate / definition / other
        · 消歧     : 把指代（"该公司""本公司"）补全为文档中的规范主体名，
                     并把问题里的口语化说法对齐到招股书用词
        · 分解     : 多跳问题拆成子问题（如"……分别是多少？"拆成逐项）
  （2）检索与生成
        · 文档解析 -> 已在 pdf_parser / chunker 完成
        · 向量检索 -> retriever（向量 + BM25 混合、RRF 融合）+ reranker 精排
        · 答案生成 -> LangChain 编排 + 本机 Ollama qwen2.5:3b，带 [n] 引用标注
  （3）用户体验
        · 答案 + 出处（页码、章节路径）；流式输出；纯 LLM 对照

框架选型：LangChain（工单指定 LangChain / LlamaIndex 二选一）
  - ChatPromptTemplate + LCEL（prompt | llm | parser）组织链；
  - 底层 still 走本机 Ollama（langchain-ollama 的 ChatOllama），
    只访问 /api/chat，绝不 /api/pull，因此不会触发任何模型下载。

性能设计（工单要求 ≤3 秒）：
  - 查询理解用一次短 JSON 调用（num_predict 限制在 160），并缓存同题结果；
  - 检索/精排都在 GPU 上，百毫秒级；
  - 生成阶段用**流式**，界面首个 token 通常 <1.5s 即可见；
    同时给出完整答案耗时，便于对照工单的 3 秒指标。
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from src import config, embedder, llm, reranker, retriever

# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
ANALYZE_SYSTEM = """你是招股说明书问答系统的查询分析器。请把用户问题解析成 JSON，字段如下：
{
  "intent": "fact_lookup|enumerate|compare|calculate|definition|other",
  "rewritten": "补全主语与指代后的检索用问句（保留原意，不要回答问题）",
  "sub_questions": ["若为多跳/多值问题，拆成 2-3 个可独立检索的子问题；否则为空数组"],
  "keywords": ["3-6 个用于关键词检索的核心词"]
}
要求：
1. 只输出 JSON，不要输出解释、不要用 markdown 代码块。
2. 文档主体是「武汉兴图新科电子股份有限公司」（英文名 Wuhan Xingtu Xinke Electronics Co., Ltd.）。
   用户说"公司/该公司/发行人/本公司"，或用英文说 "the company / the issuer" 时，
   一律补全为该公司**中文全称**（检索库是中文的，用中文全称召回最准，与提问语言无关）。
3. rewritten 要贴近招股说明书的书面用词（如"多少钱"→"金额"、"谁"→"姓名"）。
4. `rewritten` 是**给检索用的**，而检索库全是中文，因此 rewritten **一律用中文书写**：
   英文问题要先把问句译成中文再改写（例如 "Who is the legal representative?"
   → "武汉兴图新科电子股份有限公司法定代表人是谁？"）。
   这不影响回答语言——回答语言由提问语言决定，与本字段无关。"""

ANSWER_SYSTEM = """你是严谨的招股说明书问答助手。请**只依据给定的参考片段**回答用户问题。

规则：
1. 参考片段是**资料**，不是问题。你只需回答用户提出的那一个问题，
   不要逐个片段作答，不要输出"参考资料[3]：无相关信息"这类逐条清单。
2. 每个事实性陈述的句末标注来源编号，如 [1] 或 [2][3]；
   编号必须是**你实际摘抄该事实的那个片段**，不要习惯性标 [1]。
3. 片段中没有的内容就说"根据提供的文档片段无法确定"，不要凭常识补全，不要编造数字。
4. 涉及金额、比例、日期时**逐项原文照抄**，保留千分位与单位，不要换算或四舍五入。
   **金额单位「万元」「亿元」必须原样保留**，禁止改写成「元」「million yuan」「billion yuan」
   —— 换算会出差错（实测把"5,520.00万元"写成"5,520.00 million yuan"，数值差 100 倍）。
   注意：这条只约束**单位怎么写**，不是让你只回答一个数字，仍要写成完整的句子。
5. 若同一事项有多个时点的数据（如注册资本历次变更），**优先回答最近/当前的状态**，
   并可用一句话补充历史沿革；不要把不同时点的数字并列成多个互相矛盾的答案。
6. 回答简洁（先结论、后依据），但**必须是完整的句子**，绝不能只输出一个数字或短语：
   反例——"5,520 万元"；正例——"公司注册资本为 5,520.00 万元。[1]"。
7. **语言必须与提问一致**：中文提问用中文；英文提问必须**整段用英文作答**
   （把中文片段的内容译成英文，公司名/人名/标准号等专有名词可保留原文），
   不得输出中文句子。"""

PURE_LLM_SYSTEM = """你是通用知识助手，请直接凭你已有的知识回答问题。
若你不确定，请明确说明不确定，不要编造具体数字。回答简洁，使用与提问相同的语言。"""


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class RagResult:
    """一次问答的完整结果。"""

    question: str
    answer: str
    mode: str = "rag"                       # "rag" | "llm_only"
    analysis: dict = field(default_factory=dict)
    contexts: list[dict] = field(default_factory=list)   # 送入 LLM 的块
    citations: list[dict] = field(default_factory=list)  # 答案中实际引用的块
    timings: dict = field(default_factory=dict)
    model: str = ""
    repairs: list[dict] = field(default_factory=list)   # 引用修正记录
    truncated: bool = False                             # 是否因"编号失控"被提前中断

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer, "mode": self.mode,
            "analysis": self.analysis, "repairs": self.repairs,
            "contexts": [{"chunk_id": c.get("chunk_id"), "page": c.get("page_idx", 0) + 1,
                          "heading_path": c.get("heading_path"),
                          "rerank_score": c.get("rerank_score"),
                          "rrf_score": c.get("rrf_score"),
                          "text": c.get("text", "")} for c in self.contexts],
            "citations": self.citations,
            "timings": self.timings,
            "model": self.model,
            "answer_length": len(self.answer),
        }


# ---------------------------------------------------------------------------
# 1) Query 理解
# ---------------------------------------------------------------------------
_COMPANY = "武汉兴图新科电子股份有限公司"
# 英文环境下用户可能用注册英文名提问，消歧校验时一并认作"主体已明确"
_COMPANY_EN = "Wuhan Xingtu Xinke Electronics"
_analysis_cache: dict[str, dict] = {}
_ANALYSIS_CACHE_MAX = 500

# 中文指代：长的排前面（正则是左起优先，这样"该公司"不会被拆成"该"+"公司"）
_ZH_REF_RE = re.compile(r"该公司|本公司|发行人|公司")
# 英文指代：the company / the issuer / the registrant ...
_EN_REF_RE = re.compile(
    r"\bthe\s+(?:[Cc]ompany|[Ii]ssuer|[Rr]egistrant|[Ff]irm|[Cc]orporation)\b")
_ANY_CJK_RE = re.compile(r"[一-鿿]")
_ANY_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """粗判提问语言，返回 'zh' 或 'en'。

    只用于两处：把语言硬约束贴到问题旁边、以及消歧模式选择，
    因此按字符占比粗判即可，不需要引入语言识别依赖。
    """
    t = text or ""
    cjk = len(_ANY_CJK_RE.findall(t))
    latin = len(_ANY_LATIN_RE.findall(t))
    if cjk == 0 and latin > 0:
        return "en"
    if latin > cjk * 2 and latin >= 4:
        return "en"
    return "zh"


def disambiguate(text: str) -> str:
    """把指代补全为公司全称（中文"该公司/发行人"与英文"the company"同等对待）。

    英文问句原先不做消歧 —— 实测 "Who is the legal representative of the company?"
    因此召回到**别的公司**（武汉兴图互联、另一家科技公司），答案张冠李戴。
    补成中文全称即可修复：bge-m3 是中英双语模型，中文实体名放进英文问句
    照样能对上下文的向量，不影响跨语言检索。
    """
    if not text or _COMPANY in text:
        return text
    for pattern in (_ZH_REF_RE, _EN_REF_RE):
        m = pattern.search(text)
        if m:
            return text[:m.start()] + _COMPANY + text[m.end():]
    return text


def _llm(temperature: float | None = None, num_predict: int | None = None,
         json_mode: bool = False):
    """构造 ChatOllama（LangChain）。只连本机服务，不会下载模型。"""
    from langchain_ollama import ChatOllama

    kwargs = dict(
        model=llm.resolve_model(),
        base_url=config.OLLAMA_HOST,
        temperature=config.OLLAMA_TEMPERATURE if temperature is None else temperature,
        num_ctx=config.OLLAMA_NUM_CTX,
        num_predict=num_predict or config.OLLAMA_NUM_PREDICT,
        keep_alive="30m",
        # 【必须有超时】实测过一次单题卡住约 16 分钟（显存被桌面应用挤占，
        # Ollama 退化到 CPU 后近乎停滞）。ollama 客户端默认不设读超时，
        # 流式请求一挂就是无限等；这里显式兜底，超时按"LLM 不可用"处理。
        client_kwargs={"timeout": config.LLM_TIMEOUT_S},
    )
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def _to_lc_messages(messages: list[dict]) -> list:
    """把 [{role, content}] 转成 LangChain 消息对象。

    直接用 SystemMessage/HumanMessage 而不是 ChatPromptTemplate：
    系统提示里含 JSON 示例（{"intent": ...}），若走模板会被当成模板变量，
    报 "Input to ChatPromptTemplate is missing variables" —— 这里从源头绕开。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    mapping = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [mapping.get(m.get("role", "user"), HumanMessage)(content=m.get("content", ""))
            for m in messages]


def _content_to_text(content) -> str:
    """兼容 LangChain 不同版本：content 可能是 str 或内容块列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", "") or "")
        return "".join(parts)
    return str(content or "")


def _fallback_analysis(question: str) -> dict:
    """规则版 Query 理解：意图识别 + 消歧 + 分解 + 关键词。

    它有两个身份：
      1. **快路径主力** —— 问题已含公司全称、无消歧需求时直接用它，
         省掉一次 LLM 调用（实测约 1.4 秒）；
      2. **LLM 失败时的兜底** —— 保证链路不中断。

    因此规则要写得足够好：工单要求的三项能力（意图识别/消歧/分解）
    它都必须真的产出，而不是占位。
    """
    q = question.strip()
    if detect_language(q) == "en":
        if re.search(r"\b(how many|how much|list|respectively)\b", q, re.I):
            intent = "enumerate"
        elif re.search(r"\b(what|who|which|when|where|whose|how)\b", q, re.I):
            intent = "fact_lookup"
        else:
            intent = "other"
    else:
        intent = "other"
        if re.search(r"分别是多少|分别|各是|有哪些|哪些", q):
            intent = "enumerate"
        elif re.search(r"多少|几|比重|比例|金额", q):
            intent = "fact_lookup"
        elif re.search(r"什么是|含义|定义", q):
            intent = "definition"

    rewritten = disambiguate(q)
    keywords = [w for w in re.findall(r"[一-鿿]{2,6}|[A-Za-z]{2,}", q)
                if w not in ("多少", "哪些", "什么", "根据", "关于")][:6]
    return {"intent": intent, "rewritten": rewritten,
            "sub_questions": _rule_sub_questions(q, intent), "keywords": keywords,
            "source": "rule"}


def _rule_sub_questions(q: str, intent: str) -> list[str]:
    """规则分解：产出**与原文不同**的检索变体。

    【为什么必须产出变体】`retrieve()` 会把 [原问题, rewritten, *子问题] 去重后
    逐条检索再 RRF 融合。快路径下 rewritten == 原问题，若子问题再为空，
    **就只剩一条检索式**，候选集比 LLM 路径窄一截 —— 实测 Q207 因此多召回了
    一个无关片段（p479 的"募集资金合计40,584.83万元"），模型被带偏，
    把补充流动资金的金额答成了注册资本 5,520 万元（4 次里错 3 次）。

    所以这里不是"锦上添花"，而是快路径能否成立的关键：
      1. 「分别是多少」→ 换成「各年度分别是多少」，换一种检索说法；
      2. 剥掉冗长的公司全称 → 让检索看到不含主体噪声的语义核心。
         原问题（含全称）已作为第 1 条查询锚定主体，两条互补。

    【长度门槛为什么是 12】剥掉主体后如果只剩一句泛问（如"参与制定了哪个技术标准？"），
    这条查询会丢掉全部锚点、把无关片段拉进候选：实测 Q95 因此从稳定的 0.5 掉到 0.0
    （召回了第 156 页某高管的任职经历）。核心足够长（≥12 字）时才说明它自带
    足够的限定词（时间范围、指标名、行业名），剥主体才是净收益。
    """
    subs: list[str] = []
    if re.search(r"分别是多少|分别为多少", q):
        subs.append(re.sub(r"分别是多少[？?]?", "各年度分别是多少？", q))
    if _COMPANY in q:
        core = q.replace(_COMPANY, "").strip(" ，,、")
        # 量长度时要甩掉句末标点，否则"参与制定了哪个技术标准？"会因末尾的
        # "？"凑够 12 字而蒙混过关（实测这正是它把 Q95 拉垮的原因）
        if len(core.rstrip("？?！!。. ")) >= 12:
            subs.append(core)
    return subs[:3]


def _needs_llm_analysis(question: str) -> bool:
    """判断是否值得为这个问题付一次 LLM 分析调用（实测约 1.4 秒）。

    问题里已经出现公司全称时，Query 理解最主要的收益——"该公司指的是谁"的
    消歧——已经不存在，规则分析足以给出意图/分解/关键词，因此走快路径。
    含指代的问句仍交给 LLM 处理。

    工单 10 道必测题全部含公司全称，因此全部命中快路径。
    """
    q = (question or "").strip()
    if not q:
        return False
    if detect_language(q) == "en":
        # 英文问句必须交给 LLM：检索库是中文的，要让模型把问句先译成中文再改写，
        # 否则英文 query 与中文语料的向量距离偏远。实测同义的中英问句
        # 召回结果不同——"法定代表人是谁"能召回第 52 页（程家明），
        # 而英文版却召回到第 255 页另一家公司的人。
        return True
    if _COMPANY in q:
        return False          # 主体已明确，无需消歧
    if len(q) <= 6:
        return False          # 极短问句规则足够
    return True


def norm_analyze_mode(analyze) -> str:
    """把历史布尔参数与新的三态字符串统一成 'auto' | 'always' | 'off'。

    兼容旧调用：True（旧默认）→ auto，False → off。
    """
    if analyze is True:
        return "auto"
    if analyze is False:
        return "off"
    if analyze in ("auto", "always", "off"):
        return analyze
    return "auto"


def analyze_query(question: str, use_cache: bool = True,
                  mode: str = "auto") -> dict:
    """意图识别 + 消歧 + 分解。返回 {intent, rewritten, sub_questions, keywords, source}。

    mode 三态（见 norm_analyze_mode）：
      · "auto"   —— 主体已明确的问题走规则（省约 1.4 秒），含指代的才调 LLM；
      · "always" —— 一律调 LLM（慢，但改写质量最高，便于对照）；
      · "off"    —— 一律走规则。
    """
    question = (question or "").strip()
    if not question:
        return _fallback_analysis(question)
    if use_cache and question in _analysis_cache:
        return _analysis_cache[question]

    t0 = time.time()
    if mode == "off" or (mode == "auto" and not _needs_llm_analysis(question)):
        # 快路径：规则已经能给出意图/消歧/分解，不必付一次 LLM 调用
        result = _fallback_analysis(question)
    else:
        try:
            obj = llm_json(ANALYZE_SYSTEM, question, num_predict=200)
            result = {
                "intent": str(obj.get("intent") or "other"),
                "rewritten": str(obj.get("rewritten") or question),
                "sub_questions": [str(s) for s in (obj.get("sub_questions") or [])][:3],
                "keywords": [str(k) for k in (obj.get("keywords") or [])][:6],
                "source": "llm",
            }
        except Exception as exc:
            print(f"[rag] 查询理解失败，使用规则兜底：{exc}", flush=True)
            result = _fallback_analysis(question)

    # 消歧校验：原问题含公司全称、而改写句反而丢了主体时，说明"消歧"帮了倒忙，
    # 直接把主体补回改写句，避免召回漂移。
    # 英文问句用注册英文名指代主体时同样视为"主体已明确"。
    if (_COMPANY in question or _COMPANY_EN in question) \
            and _COMPANY not in result["rewritten"]:
        result["rewritten"] = question
    elif _COMPANY not in result["rewritten"]:
        # 兜底：问题里含指代（中文"该公司"/英文 "the company"）却没被消歧，
        # 用规则补一次。实测英文问句若不补，会召回到**别的公司**上去。
        fixed = disambiguate(question)
        if fixed != question:
            result["rewritten"] = fixed

    result["elapsed"] = round(time.time() - t0, 3)
    # 有界缓存：服务长时间运行时避免无限增长（超出后丢弃最早的一半）
    if len(_analysis_cache) >= _ANALYSIS_CACHE_MAX:
        for key in list(_analysis_cache)[: _ANALYSIS_CACHE_MAX // 2]:
            _analysis_cache.pop(key, None)
    _analysis_cache[question] = result
    return result


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def llm_json(system: str, user: str, num_predict: int = 200) -> dict:
    """让本机 LLM 输出 JSON 并解析成 dict（供 Query 理解、评估裁判等复用）。

    这是本模块对外暴露的"结构化调用"入口，避免外部模块直接依赖下划线私有函数。
    """
    chain = _llm(temperature=0.0, num_predict=num_predict, json_mode=True)
    raw = _content_to_text(chain.invoke(_to_lc_messages(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}])).content)
    return json.loads(_strip_code_fence(raw))


# ---------------------------------------------------------------------------
# 2) 检索 + 精排
# ---------------------------------------------------------------------------
def retrieve(question: str, analysis: dict | None = None,
             top_k: int | None = None, hybrid: bool | None = None,
             use_rerank: bool | None = None) -> tuple[list[dict], dict]:
    """混合召回 → RRF 融合 → Cross-Encoder 精排。返回 (候选块, 各阶段耗时)。

    hybrid / use_rerank 允许调用方（界面开关）覆盖 config 里的全局默认值。
    """
    analysis = analysis or analyze_query(question)
    timings: dict[str, float] = {}

    # 原始问题永远参与召回：改写出的问句一旦丢了主体（实测出现过
    # "…改写为『该公司注册资本是多少』"），召回会漂到别的公司/主体上去。
    # 原始问句是最可靠的锚点，改写句只作为补充召回。
    queries = [question, analysis.get("rewritten") or question]
    queries += [s for s in analysis.get("sub_questions", []) if s]
    queries = list(dict.fromkeys(q for q in queries if q))[:4]

    t0 = time.time()
    merged: dict[str, dict] = {}
    for q in queries:
        hits = retriever.search(q, top_k=config.VECTOR_TOP_K, hybrid=hybrid)
        for rank, h in enumerate(hits, start=1):
            key = h.get("chunk_id") or f"{h.get('page_idx')}-{h.get('text','')[:32]}"
            # 同一块被多条查询命中时，按"最好名次"计分并累计小量加权，避免重复块挤占名额
            score = h.get("rrf_score", 0.0) / max(rank, 1)
            if key in merged:
                prev = merged[key]
                prev["rrf_score"] = prev.get("rrf_score", 0.0) + score * 0.5
                prev.setdefault("matched_queries", []).append(q)
            else:
                h["matched_queries"] = [q]
                h["rrf_score"] = score
                merged[key] = h
    timings["retrieve_s"] = round(time.time() - t0, 3)

    candidates = sorted(merged.values(), key=lambda c: c.get("rrf_score", 0.0),
                        reverse=True)[:config.RERANKER_TOP_K_IN]

    t0 = time.time()
    k = top_k or config.FINAL_TOP_K
    if use_rerank is False:
        # 界面关掉精排时走这里：直接按融合名次截断，不再调用 Cross-Encoder
        final = candidates[:k]
    else:
        final = reranker.rerank(question, candidates, top_k=k)
    timings["rerank_s"] = round(time.time() - t0, 3)
    timings["candidates"] = len(candidates)
    timings["reranked"] = bool(use_rerank is not False)
    return final, timings


# ---------------------------------------------------------------------------
# 3) 上下文与引用
# ---------------------------------------------------------------------------
def build_context(chunks: list[dict]) -> str:
    """把候选块编号后拼成送入 LLM 的上下文。"""
    parts = []
    for i, c in enumerate(chunks, start=1):
        head = " > ".join(c.get("heading_path") or [])
        page = int(c.get("page_idx", 0)) + 1
        body = (c.get("raw_text") or c.get("text", "")).strip()
        label = "表格" if c.get("type") == "table" else "正文"
        parts.append(f"[{i}] （{label}｜第{page}页｜{head}）\n{body}")
    return "\n\n".join(parts)


_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_CITE_ANY_RE = re.compile(r"\[(\d{1,3})\]")
_CITE_RUN_RE = re.compile(r"(?:\[\d{1,3}\][ \t]*){3,}")
# 模型把提示词里的占位符当成示例抄进答案：实测出现过 "15,000 万元 [1] [编号]"
_PLACEHOLDER_CITE_RE = re.compile(r"\[(?:编号|数字|引用|来源|序号|n|citation)\]", re.I)
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CJK_RE = re.compile(r"[一-鿿]{2,}")


def clean_answer(text: str, n_chunks: int) -> str:
    """清理跑飞的引用编号。

    【为什么需要】实测第 531 题（"法定代表人是谁"）出现过模型"编号失控"：
    答案正文只有一句"法定代表人：程家明"，后面却跟着 [1] [2] [5] [6] [7] …
    **一路编号到 [65]**（上下文里只有 6 条片段），一直生成到 token 上限才停，
    该题耗时从 1.9 秒涨到 7.0 秒，答案也被垃圾编号淹没。

    清理三件事（都在模型输出之后、展示之前）：
      1. 删掉**越界编号**（引用不存在的片段，无任何信息量）；
      2. 把连续 3 个以上的编号串折叠为该串里前 3 个**不同**编号；
      3. 压缩由此产生的多余空格。
    """
    if not text:
        return text
    text = _PLACEHOLDER_CITE_RE.sub("", text)   # 清掉被抄进来的占位符 [编号]
    text = _CITE_ANY_RE.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_chunks else "", text)

    def _collapse(match: re.Match) -> str:
        nums = list(dict.fromkeys(re.findall(r"\d{1,3}", match.group(0))))[:3]
        return "".join(f"[{n}]" for n in nums)

    text = _CITE_RUN_RE.sub(_collapse, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([。，；！？])", r"\1", text)
    return text.strip()


def _facts(text: str) -> tuple[set[str], set[str]]:
    """抽取一句话里的"可核对要素"：(数字集合, 中文二元词组集合)。

    中文必须切成二元词组（bi-gram）。早期版本用 [一-鿿]{2,} 直接匹配，
    结果整句中文变成**一个**元素，重合度非 0 即 1，核验形同虚设。
    """
    text = _CITE_RE.sub("", text or "")
    nums = set(_NUM_RE.findall(text))
    grams: set[str] = set()
    for run in _CJK_RE.findall(text):
        if len(run) < 2:
            continue
        grams.update(run[i:i + 2] for i in range(len(run) - 1))
    return nums, grams


def _score(sent_nums: set[str], sent_grams: set[str],
           chunk_nums: set[str], chunk_grams: set[str]) -> float:
    """句子与片段的契合度。数字权重更高——招股书问答里数字就是事实本身。"""
    gram_ov = len(sent_grams & chunk_grams) / max(len(sent_grams), 1)
    if not sent_nums:
        return gram_ov
    num_ov = len(sent_nums & chunk_nums) / len(sent_nums)
    return 0.7 * num_ov + 0.3 * gram_ov


def verify_citations(answer: str, chunks: list[dict],
                     abs_threshold: float = 0.55,
                     rel_threshold: float = 0.75) -> dict[int, dict]:
    """核验 [n] 引用是否真的指向"支撑该句"的片段。

    【为什么需要】实测发现 3B 模型会把数字抄对、却标到排序第一的块上
    （例如问"军用领域收入分别是多少"，它引用的是讲"占比"的第 1 块，
    而金额其实出自第 2 块）。这类"张冠李戴"在招股书问答里是硬伤，
    因此做一次独立核验：把回答按句拆开，逐句与各片段求数字/二元词组契合度，
    取契合度最高者为该句的真实出处。

    判定为可信需满足其一：
      · 绝对分 >= abs_threshold（片段确实覆盖了该句的主要数字与用词）；
      · 相对分 >= rel_threshold × 最佳片段得分（改写表述导致绝对分偏低，
        但只要它和最佳片段基本同档，就算可接受）。

    返回 {引用编号: {"verified", "overlap", "best_index", ...}}。
    只做标记、不篡改模型输出，核验结论以 verified 字段呈现给用户。
    """
    # 模型常把引用写在句末标点**之后**（"…补充流动资金。 [1]"）。若直接按标点切句，
    # 引用标号会被切进下一句，而下一句没有事实内容 —— 核验就会把它误判为"无依据"。
    # 这里先做一次归一化：把紧跟标点的引用标号挪回标点之前。
    normalized = re.sub(r"([。；！？\n])\s*((?:\[\d{1,2}\]\s*)+)", r"\2\1", answer or "")

    sentences = [s for s in re.split(r"(?<=[。；！？\n])", normalized) if s.strip()]
    chunk_facts = [_facts(c.get("raw_text") or c.get("text", "")) for c in chunks]

    report: dict[int, dict] = {}
    for sent in sentences:
        cited = [int(n) for n in _CITE_RE.findall(sent)]
        if not cited:
            continue
        sent_nums, sent_grams = _facts(sent)
        if not sent_nums and not sent_grams:
            continue
        scores = [_score(sent_nums, sent_grams, cn, cg) for cn, cg in chunk_facts]
        best = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
        best_score = scores[best] if best >= 0 else 0.0

        for n in cited:
            if not (1 <= n <= len(chunks)):
                continue
            score = scores[n - 1]
            entry = report.setdefault(n, {"verified": False, "overlap": 0.0,
                                          "best_index": best + 1, "best_overlap": 0.0})
            entry["overlap"] = round(max(entry["overlap"], score), 3)
            entry["best_overlap"] = round(max(entry["best_overlap"], best_score), 3)
            if score >= abs_threshold or (best_score > 0
                                          and score >= rel_threshold * best_score):
                entry["verified"] = True
    return report


def repair_citations(answer: str, chunks: list[dict],
                     max_overlap: float = 0.30,
                     min_best_overlap: float = 0.60) -> tuple[str, list[dict]]:
    """把**明显标错**的引用编号改成核验出的正确编号，返回 (修正后答案, 修正记录)。

    【为什么敢改】仅修正证据非常明确的case：
      · 模型标的片段与该句几乎没有交集（overlap < max_overlap）；
      · 而另一个片段与该句高度重合（best_overlap >= min_best_overlap）。
    典型例子：问"军用领域收入分别是多少"，模型抄对了金额、却标到只讲"占比"的
    片段上（overlap 0.25），而同时含金额与占比的片段重合度 0.8+ —— 这种情况下
    把 [1] 改成 [2] 是有据可查的纠正，不是猜测。

    修正会被记录并**在界面上明确告知用户**（"已按内容核验修正 N 处"），
    原始编号保留在 original_index 中可追溯。
    """
    report = verify_citations(answer, chunks)
    repairs: list[dict] = []
    used_from: set[int] = set()
    used_to: set[int] = set()

    for n, check in report.items():
        if check.get("verified"):
            continue
        best = check.get("best_index")
        overlap = check.get("overlap", 1.0)
        best_overlap = check.get("best_overlap", 0.0)
        if not best or best == n or not (1 <= best <= len(chunks)):
            continue
        if overlap >= max_overlap or best_overlap < min_best_overlap:
            continue
        if n in used_from or best in used_to:
            continue  # 同一编号只改一次，避免连锁替换
        used_from.add(n)
        used_to.add(best)
        repairs.append({"from": n, "to": best,
                        "from_page": int(chunks[n - 1].get("page_idx", 0)) + 1,
                        "to_page": int(chunks[best - 1].get("page_idx", 0)) + 1,
                        "overlap": overlap, "best_overlap": best_overlap})

    if not repairs:
        return answer, []

    # 【必须一次替换完成】早期版本用 fixed.replace() 逐条替换，是**全局**字符串替换：
    # 修完 [1]→[2] 之后，下一条 [2]→[3] 会把刚写进去的 [2] 也一并改成 [3]
    # （链式传导），结果是"修正记录说改成了 [2]，答案里却是 [3]"。
    # 改成一次正则替换：每个原始编号最多替换一次，替换结果不会被再次扫描。
    mapping = {r["from"]: r["to"] for r in repairs}
    fixed = _CITE_ANY_RE.sub(
        lambda m: (f"[{mapping[int(m.group(1))]}]"
                   if int(m.group(1)) in mapping else m.group(0)),
        answer)
    return fixed, repairs


def extract_citations(answer: str, chunks: list[dict],
                      report: dict | None = None) -> list[dict]:
    """从答案里解析 [n] 引用，映射回具体块（页码、章节），并附带核验结论。

    report 可传入 repair_citations 已经算好的核验结果，避免重复计算。
    """
    report = report if report is not None else verify_citations(answer, chunks)
    used = []
    seen = set()
    for m in _CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if n in seen or not (1 <= n <= len(chunks)):
            continue
        seen.add(n)
        c = chunks[n - 1]
        check = report.get(n, {})
        used.append({
            "index": n,
            "chunk_id": c.get("chunk_id"),
            "page": int(c.get("page_idx", 0)) + 1,
            "heading_path": c.get("heading_path") or [],
            "type": c.get("type", "text"),
            "rerank_score": c.get("rerank_score"),
            "snippet": (c.get("raw_text") or c.get("text", "")).strip()[:180],
            # 核验字段：verified=False 表示该编号可疑，best_index 是核验出的更可能出处
            "verified": bool(check.get("verified", False)),
            "overlap": check.get("overlap", 0.0),
            "suggested_index": (check.get("best_index")
                                if not check.get("verified") else None),
        })
    return used


# 英文提问时的语言硬约束。放在**用户问题正下方**而不是只写在 system 提示里 ——
# 实测 3B 模型对 system 里"用提问相同的语言作答"这句基本无视，英文问句照答中文；
# 贴到问题旁边（离生成位置最近）才压得住。
_LANG_DIRECTIVE_EN = (
    "\nLANGUAGE REQUIREMENT: The user asked in English. Write the ENTIRE answer "
    "in English — translate the Chinese source passages into English. Keep proper "
    "nouns (company names, person names, standard codes) as-is, and do NOT answer "
    "in Chinese.")


def build_messages(question: str, contexts: list[dict],
                   history: list[dict] | None = None) -> list[dict]:
    """构造生成用消息列表（system + 历史 + 带上下文的当前问题）。"""
    ctx = build_context(contexts)
    # 【别写 "[编号]" 这种占位符】3B 模型会把它当成示例原样抄进答案里，
    # 实测答案出现过 "15,000 万元 [1] [编号]" 这种带占位符的畸形输出。
    # 给一个具体样例（[1]）它才照做。
    tail = "请依据上述片段，用完整的句子作答，并在每个事实句末标注来源片段的编号（如 [1]）。"
    if detect_language(question) == "en":
        tail += _LANG_DIRECTIVE_EN
    user = (f"参考片段：\n{ctx}\n\n"
            f"用户问题：{question}\n\n"
            f"{tail}")
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for turn in (history or [])[-4:]:
        msgs.append({"role": turn.get("role", "user"),
                     "content": turn.get("content", "")})
    msgs.append({"role": "user", "content": user})
    return msgs


# ---------------------------------------------------------------------------
# 4) 对外问答接口
# ---------------------------------------------------------------------------
# 连续 4 个及以上引用编号收尾 —— 判定为"编号失控"（正常回答不会这样堆引用）
_RUNAWAY_RE = re.compile(r"(?:\[\d{1,3}\][ \t]*){4,}$")


def _looks_runaway(text: str) -> bool:
    """检测模型是否陷入"引用编号复读"。"""
    return bool(_RUNAWAY_RE.search(text[-160:]))


# 剔除引用标号后，只剩空白与标点即视为"没有正文"
_TRIVIAL_STRIP_RE = re.compile(
    r"[\s　。，、；：！？.,;:!?~…—\-\[\]（）()【】《》\"'“”‘’]+")


def _has_substance(text: str) -> bool:
    """答案里是否有实际内容，而不只是一串引用编号。

    实测英文提问时 3B 模型偶发只输出 "[2][3][4]" 就停下（见工单核查记录），
    界面上呈现为一个空答案，用户完全无法使用。
    """
    t = _CITE_ANY_RE.sub("", text or "")
    t = _TRIVIAL_STRIP_RE.sub("", t)
    return len(t) >= 4


def _regenerate(question: str, contexts: list[dict],
                history: list[dict] | None) -> str:
    """重试一次生成（非流式）。仅在模型没吐出正文时调用，属罕见路径。"""
    try:
        chain = _llm()
        msgs = build_messages(question, contexts, history)
        msgs[-1]["content"] += ("\n\n注意：上一次回答只输出了引用编号、没有正文。"
                               "这次必须写出完整的句子。")
        ai = chain.invoke(_to_lc_messages(msgs))
        return _content_to_text(ai.content).strip()
    except Exception as exc:
        print(f"[rag] 重试生成失败：{exc}", flush=True)
        return ""


def ask(question: str, top_k: int | None = None,
        history: list[dict] | None = None,
        analyze: bool | str = "auto", hybrid: bool | None = None,
        use_rerank: bool | None = None) -> RagResult:
    """RAG 问答（非流式）。

    内部复用 ask_stream 而不是单独调 invoke：这样非流式路径同样享有
    "编号失控提前中断"的保护，并能一并拿到首字耗时。对外仍是同步返回。

    analyze 支持三态（"auto"/"always"/"off"）或旧布尔值，见 norm_analyze_mode。
    """
    result = RagResult(question=question, answer="", mode="rag")
    for event in ask_stream(question, top_k=top_k, history=history, analyze=analyze,
                            hybrid=hybrid, use_rerank=use_rerank):
        if event["type"] == "meta":
            result.analysis = event["analysis"]
            result.contexts = event["contexts"]
            result.timings = dict(event["timings"])
        elif event["type"] == "done":
            result.answer = event["answer"]
            result.citations = event["citations"]
            result.repairs = event.get("repairs") or []
            result.timings = event["timings"]
            result.truncated = bool(event.get("truncated"))
            result.model = event.get("model") or result.model
    if not result.model:
        try:
            result.model = llm.resolve_model()
        except Exception:
            result.model = config.OLLAMA_MODEL
    return result


def _model_name(ai_message) -> str:
    """从 LangChain 返回里取实际使用的模型名。"""
    try:
        meta = getattr(ai_message, "response_metadata", None) or {}
        return meta.get("model_name") or meta.get("model") or llm.resolve_model()
    except Exception:
        return llm.resolve_model()


def ask_stream(question: str, top_k: int | None = None,
               history: list[dict] | None = None,
               analyze: bool | str = "auto", hybrid: bool | None = None,
               use_rerank: bool | None = None) -> Iterator[dict]:
    """RAG 问答（流式）。逐段 yield：
        {"type": "meta",   ...分析/耗时/候选块}
        {"type": "delta",  "text": "增量"}
        {"type": "done",   "answer": 全文, "citations": [...], "timings": {...}}
    """
    t_all = time.time()
    t0 = time.time()
    analysis = analyze_query(question, mode=norm_analyze_mode(analyze))
    timings = {"analyze_s": round(time.time() - t0, 3),
               "analyze_path": "规则" if analysis.get("source") == "rule" else "LLM"}

    # 检索也要兜底：知识库没建好 / collection 被删 / Qdrant 文件锁被占，
    # 都会在这一步抛错。早期版本没有 try，界面直接甩出原始 traceback。
    try:
        contexts, t_ret = retrieve(question, analysis, top_k=top_k,
                                   hybrid=hybrid, use_rerank=use_rerank)
    except Exception as exc:
        msg = (f"知识库暂不可用：{exc}\n\n"
               f"请确认已构建知识库（命令行执行 `python build_index.py`），"
               f"且没有其他程序正在占用本地向量库。")
        timings["total_s"] = round(time.time() - t_all, 3)
        yield {"type": "meta", "analysis": analysis, "contexts": [], "timings": timings}
        yield {"type": "delta", "text": msg}
        yield {"type": "done", "answer": msg, "citations": [], "repairs": [],
               "truncated": False, "timings": timings}
        return
    timings.update(t_ret)
    yield {"type": "meta", "analysis": analysis, "contexts": contexts,
           "timings": dict(timings)}

    if not contexts:
        answer = "未在知识库中检索到与该问题相关的内容。"
        timings["total_s"] = round(time.time() - t_all, 3)
        yield {"type": "delta", "text": answer}
        yield {"type": "done", "answer": answer, "citations": [], "timings": timings}
        return

    t0 = time.time()
    ttft = None
    buf: list[str] = []
    last_msg = None
    truncated = False
    # 用同步的 stream（不是 astream）：astream 返回 async_generator，
    # 在 Streamlit 这类同步上下文里会报 "'async_generator' object is not iterable"。
    # _llm() 会调 llm.resolve_model()，Ollama 没起或模型缺失时抛 LLMUnavailable，
    # 因此它必须也在 try 之内（早期版本放在 try 外，此时整页会崩）。
    try:
        chain = _llm()
        for piece_msg in chain.stream(
                _to_lc_messages(build_messages(question, contexts, history))):
            last_msg = piece_msg
            piece = _content_to_text(getattr(piece_msg, "content", piece_msg))
            if not piece:
                continue
            if ttft is None:
                ttft = time.time() - t0
                timings["ttft_s"] = round(ttft, 3)
            buf.append(piece)
            joined = "".join(buf)

            # 检出编号失控就立即断开，不必等模型把 token 上限耗完
            # （实测"法定代表人是谁"会从 [1][2] 一路编号到 [65]，7 秒才停）
            if _looks_runaway(joined):
                truncated = True
                break
            # 兜底时限：正常生成 1~2 秒，超过 LLM_TIMEOUT_S 说明服务端已不正常
            # （实测出现过单题挂约 16 分钟），宁可返回已生成的部分，也不要一直吊着用户
            if time.time() - t0 > config.LLM_TIMEOUT_S:
                truncated = True
                print(f"[rag] 生成超过 {config.LLM_TIMEOUT_S}s，已中断", flush=True)
                break
            yield {"type": "delta", "text": piece}
    except Exception as exc:
        # 读超时 / 连接中断 / LLM 不可用：保留已生成的部分，别让整个问答失败
        # （读超时本身由 _llm() 的 client_kwargs 控制，这里只负责兜住异常）
        print(f"[rag] 生成中断（{type(exc).__name__}: {exc}），返回已生成内容", flush=True)
        if not buf:
            hint = (f"生成失败：{exc}\n\nOllama 未启动或本机缺少该模型时，"
                    f"请先执行 `ollama serve` 并确认 `ollama list` 里有 "
                    f"{config.OLLAMA_MODEL}。")
            timings["total_s"] = round(time.time() - t_all, 3)
            yield {"type": "delta", "text": hint}
            yield {"type": "done", "answer": hint, "citations": [], "repairs": [],
                   "truncated": False, "timings": timings}
            return

    answer = clean_answer("".join(buf).strip(), len(contexts))
    # 空答案守卫：模型只吐了引用编号、没有正文时补一次。
    # 实测英文提问偶发（例如只输出 "[2][3][4]" 就停），界面上就是个空答案。
    retried = False
    if not _has_substance(answer):
        retried = True
        print("[rag] 回答只有引用编号没有正文，重试一次", flush=True)
        retry_text = _regenerate(question, contexts, history)
        if _has_substance(retry_text):
            answer = clean_answer(retry_text, len(contexts))
        else:
            answer = ("模型本次未能生成有效回答（只输出了引用编号）。"
                      "建议换个问法，或改用中文提问重试。")
    timings["generate_s"] = round(time.time() - t0, 3)
    timings["total_s"] = round(time.time() - t_all, 3)
    fixed, repairs = repair_citations(answer, contexts)
    yield {"type": "done", "answer": fixed, "repairs": repairs, "truncated": truncated,
           "retried": retried,
           "model": _model_name(last_msg) if last_msg is not None else "",
           "citations": extract_citations(fixed, contexts),
           "timings": timings}


def ask_pure_llm(question: str, history: list[dict] | None = None) -> RagResult:
    """纯 LLM 基线（不检索），用于工单要求的"RAG vs 只用 LLM"对比。"""
    t_all = time.time()
    msgs = [{"role": "system", "content": PURE_LLM_SYSTEM}]
    for turn in (history or [])[-4:]:
        msgs.append({"role": turn.get("role", "user"),
                     "content": turn.get("content", "")})
    msgs.append({"role": "user", "content": question})

    try:
        ai = _llm().invoke(_to_lc_messages(msgs))
        answer, model = _content_to_text(ai.content).strip(), _model_name(ai)
    except Exception as exc:
        # 对照页也要能容错：Ollama 没起时给出提示，而不是抛 traceback
        answer, model = f"（纯 LLM 不可用：{exc}）", ""
    return RagResult(question=question, answer=answer, mode="llm_only", model=model,
                     timings={"generate_s": round(time.time() - t_all, 3),
                              "total_s": round(time.time() - t_all, 3)})


def warmup() -> dict:
    """服务启动预热：Ollama 常驻 + BM25 索引 + 精排权重 + embedding 权重。

    embedding 必须一并预热：它是最懒加载的一环，若留到第一次提问时加载，
    冷启动的 3~8 秒会全部记进那一题的响应时间，污染工单的"≤3 秒"指标。
    """
    info: dict = {}
    t0 = time.time()
    try:
        llm.warmup()
        info["llm"] = llm.resolve_model()
    except Exception as exc:
        info["llm_error"] = str(exc)
    try:
        embedder.get_model()
        info["embedding"] = embedder.info()["device"]
    except Exception as exc:
        info["embedding_error"] = str(exc)
    try:
        retriever.stats()
        info["bm25_docs"] = retriever.stats()["bm25_docs"]
    except Exception as exc:
        info["retriever_error"] = str(exc)
    try:
        if reranker.is_available():
            reranker.get_model()
            info["reranker"] = "ready"
    except Exception as exc:
        info["reranker_error"] = str(exc)
    info["elapsed_s"] = round(time.time() - t0, 2)
    return info


def _cli() -> None:
    import sys

    q = " ".join(sys.argv[1:]) or "武汉兴图新科电子股份有限公司注册资本是多少？"
    res = ask(q)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2)[:2000])
    print("\n--- 纯 LLM 对照 ---")
    pure = ask_pure_llm(q)
    print(pure.answer[:400])
    print("\n耗时:", pure.timings)


if __name__ == "__main__":  # 冒烟：python -m src.rag "注册资本是多少？"
    bootstrap.run_with_large_stack(_cli)
