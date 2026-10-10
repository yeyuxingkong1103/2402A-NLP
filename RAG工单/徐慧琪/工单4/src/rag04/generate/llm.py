# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""LLM 生成：DeepSeek API 主通道 + 本地 Ollama 兜底。

三级降级：API → Ollama → 返回检索原文。
任何一级失败都不空手而归（硬性要求 7）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from rag04.config import Settings
from rag04.schema import Answer, Hit

logger = logging.getLogger("rag04.llm")

# C4（终审）：降级链总时限。原实现 API 单次超时 llm_timeout_s（默认 30s）、
# Ollama llm_timeout_s*4（默认 120s），挂死的后端会让一次请求阻塞 ~150s
# ——3s 预算的 50 倍，压测时把线程池 worker 一路钉死。现在两级共享同一个
# deadline：min(llm_timeout_s, 20s)。
# 取舍（必须写明）：慢但能用的后端只要在时限内返回就照常作答（不因收紧而
# 误杀正常慢响应）；超时则落到检索原文兜底，仍满足硬性要求 7「不空手而归」。
# 主通道实测 0.7~0.9s（关 thinking），正常路径完全不受该上限影响。
_CHAIN_DEADLINE_S = 20.0
_MIN_REQUEST_TIMEOUT_S = 0.5

# Task 13 教训：热路径上不得重复创建后端客户端（连同连接池/TLS 会话一起复用）。
# 键 = (base_url, api_key)，进程内复用；即使调用方每次新建 LLMClient 也不退化。
_CLIENT_CACHE: dict[tuple[str, str], object] = {}

_CJK = re.compile(r"[一-鿿]")

_DOMAIN_HINTS_ZH = re.compile(
    r"招股|说明书|意向书|发行|股本|募集|资金|关联方|注册资本|法定代表人"
    r"|销售部|销售处|组织结构|IC市场|应用结构|增长率|军用|主营业务"
    r"|力源|兴图|新科|电子信息|技术标准|科技进步|供应商|上游|下游"
)

# C1（终审）：英文题表按 16 道英文题（questions.py 的 question_en）的真实用词
# 整理。原实现只有一个中英混排的白名单，英文侧仅 shares/prospectus/company/
# revenue/subsidiary/department 六个词条，实测 16 道英文题里 8 道
# （id 2/4/95/957/795/543/531/207）被判「与招股书无关」，在进入 LLM 之前就
# 短路成固定拒答——英文评估因此完全没跑到模型。本表覆盖英文题集里的领域词，
# 并保持「明显离题仍拒答」（天气/笑话等无任何领域词）。
_DOMAIN_HINTS_EN = re.compile(
    r"prospectus|issuance|shares?|shareholding|capital|funds?|revenue|income"
    r"|subsidiar|company|co\.,?\s*ltd|department|sales|offices?"
    r"|market|industr|enterprise|business|organizational chart"
    r"|related[-\s]?part|military|supplier|upstream|downstream"
    r"|electronic|information technology|technical standard|technology"
    r"|legal representative|reporting period|raised|invest|project"
    r"|award|prize|growth|Wuhan|Xingtu|Xinke|P&S",
    re.IGNORECASE,
)


def is_refusal_question(question: str, lang: str | None = None) -> bool:
    """明显离题或空问题 → 拒答，避免幻觉（硬性要求 7 用户输入错误）。

    ``lang`` 缺省按问题本身的语言推断（有 CJK 即中文），生成主链路显式传入
    已判定的语言。**必须按语言各查各的提示词表**：把中文白名单套到英文题上
    会把真实英文题判为离题（C1：8/16 道英文题被短路成固定拒答）。这里是
    「明显离题」的粗筛，不是精确的领域判别——域内但用词生僻的问题会放行进
    LLM，由 prompt 与 RC6/RC7 的答案级拒答判定兜底。
    """
    q = (question or "").strip()
    if len(q) < 2:
        return True
    if lang is None:
        lang = "zh" if _CJK.search(q) else "en"
    hints = _DOMAIN_HINTS_ZH if lang == "zh" else _DOMAIN_HINTS_EN
    return not hints.search(q)


# RC6：域内「检索没召回到 → 模型答无法确定」的拒答识别。只看答案开头
# （前 60 字）：先给出答案、后文补充「某细节无法确定」不算拒答；开头就是
# 「根据文档内容无法确定」的是。判据来源：id531 原型答案
# 「根据文档内容无法确定。\n\n检索片段中出现了多个不同的法定代表人姓名
# （陈爱民、程家明、胡家望）…」——逐字命中 answer_key「程家明」被判 correct。
#
# 复核修正（Fix pass 1）：裸否定不得触发——「没有控制关系的关联方包括…」
# （id4 负向问句的正确答法）、「无法回避的是…」（行文套语）都曾被误判为拒答。
# Fix pass 2：套式里的裸否定同样会命中（「根据文档内容，未分配利润…」的「未」
# 曾误伤），故甲、乙两支一律要求「否定词紧跟答案动词」；裸「未+答案动词」
# （「未找到…片段」）也补进乙支，堵住 RC6 的另一个入口。
# Fix pass 3：否定词与动词之间允许一个枚举修饰语（无法准确判断/不能完全确定…），
# 否则插一个词就能把 RC6 假阳性又放回来。
# 三条择一：
#   甲 套式：根据/依据/基于 … 内容/片段/原文 … + (否定词[+修饰语]+答案动词 或 抱歉类)；
#   乙 无套话：否定词[+修饰语]+答案动词（可带 ≤16 字前缀），如「未能立即找到…」；
#   丙 英文拒答句式。
# 已知边界：本启发式只覆盖这两种拒答形态；结构性方案（让 LLM 直接产出
# 「证据是否充分」的结构化字段，正则仅作非 JSON 后端的兜底）已排入 RC-2。
_ANSWER_VERBS = (
    r"(?:确定|给出|找到|提供|提及|作答|回答|判断|说明|披露|查明|查证|确认|得知)"
)
_REFUSAL_NEGATORS = r"(?:无法|不能|没有|并未|未(?:能)?|拒绝|不(?:能|会|予))"
# Fix pass 3：否定词与答案动词之间允许一个**有界**修饰语（「无法准确判断」
# 「不能完全确定」「没有直接披露」「未能立即找到」）。必须枚举而非 \w*——
# 通配会重新放行「未分配利润」「未经审计」这类财报术语。
_MODIFIER_GAP = r"\s*(?:准确|明确|完全|直接|确切|具体|清楚|立即|进一步)?\s*"
_NEG_VERB = _REFUSAL_NEGATORS + _MODIFIER_GAP + _ANSWER_VERBS
_ANSWER_REFUSAL_LEAD = re.compile(
    r"^\s*(?:根据|依据|基于)[^。！？\n]{0,24}?"
    r"(?:内容|片段|信息|上下文|原文|材料|资料|文档|招股(?:说明|意向)书)"
    r"\s*(?:中|里|内)?\s*[，,：:]?\s*"
    r"(?:" + _NEG_VERB + r"|抱歉|很抱歉|对不起)"
    r"|"
    r"^\s*[^。！？\n]{0,16}?" + _NEG_VERB +
    r"|"
    r"^\s*[^.!?\n]{0,40}?\b(?:i (?:cannot|can't|am unable to|could not|have no)"
    r"|sorry|unable to|cannot (?:determine|answer|find)"
    r"|not (?:mentioned|specified|disclosed|available|provided))\b",
    re.IGNORECASE,
)


def detect_answer_refusal(answer: str, head_chars: int = 60) -> bool:
    """答案文本开头是否是拒答表述（id531 式「根据文档内容无法确定」）。

    只认两种形态：套式拒答（根据…内容… + 否定词+答案动词，如「无法确定」）
    或否定词紧跟答案动词（无法确定/未能提供/没有提及/未找到…，可带短前缀）。
    裸的「没有+X」名词短语、「未分配利润/未经审计」这类术语、"无法回避"
    这类行文套语都不算拒答（Fix pass 1/2 复核指出的误判）。
    """
    return bool(_ANSWER_REFUSAL_LEAD.search((answer or "").strip()[:head_chars]))


# RC7：结构化字段的两种形态。首选第一行的 JSON（prompt 要求），同时兼容
# 纯文本标签行（不同后端/模型对 JSON 指令的遵从度不一）。均只看答案前几行。
_ANSWERABLE_LINE = re.compile(
    r'^\s*(?:\*\*|`+)?\s*(?:answerable|证据充分性|能否作答|可答性|可否作答)'
    r'\s*(?:\*\*|`+)?\s*[:：=]\s*(?:\*\*|`+)?\s*'
    r'(true|false|yes|no|是|否|可答|不可答|充分|不足)',
    re.IGNORECASE,
)
_JSON_BLOCK = re.compile(r"\{[^{}]*\}")


def parse_answerability(text: str, max_lines: int = 6) -> tuple[bool | None, str]:
    """解析 LLM 结构化输出的证据充分性字段（RC7 主判据）。

    返回 ``(answerable, 正文)``：识别到标记时从正文中**剥离**标记行（含
    ```json 围栏行），避免标记本身污染答案与正则兜底；未识别到返回
    ``(None, 原文)``，调用方回退正则（非 JSON 后端 / 模型未遵从格式）。
    兼容：JSON 行、```json 围栏、``answerable: true``、``证据充分性：不足``。
    """
    text = (text or "").strip()
    if not text:
        return None, ""
    lines = text.split("\n")

    def _rest(idx: int) -> str:
        kept = [ln for j, ln in enumerate(lines)
                if j != idx and not ln.strip().startswith("```")]
        return "\n".join(kept).strip()

    for i, line in enumerate(lines[:max_lines]):
        raw_line = line.strip()
        if not raw_line:
            continue
        if raw_line.startswith("```"):
            inner = raw_line.strip("`").strip()      # ```json → "json"；``` → ""
            if not inner or inner.lower() == "json":
                continue                             # 纯围栏行：标记在下一行
            cand = inner                             # 单行内联 ```json {...}```
        else:
            cand = raw_line
        m = _JSON_BLOCK.search(cand)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
            if isinstance(obj, dict) and "answerable" in obj:
                v = obj["answerable"]
                if isinstance(v, bool):
                    return v, _rest(i)
                if isinstance(v, str) and v.strip().lower() in (
                        "true", "false", "yes", "no", "是", "否"):
                    return v.strip().lower() in ("true", "yes", "是"), _rest(i)
        lm = _ANSWERABLE_LINE.match(cand)
        if lm:
            raw = lm.group(1).strip().lower()
            return raw in ("true", "yes", "是", "可答", "充分"), _rest(i)
        break                          # 首行不是标记 → 不再向后扫描（正文可能含引号）
    return None, text


def _is_unsupported_param_error(e: Exception) -> bool:
    """判定异常是否属于「服务端不认识 thinking 参数」这一类。

    与 Task 6 视觉链路同一判定思路（见 rag04.ingest.vlparser）：不能只测错误
    文本里有无 thinking —— OpenAI 兼容服务端拒绝未知参数的报文常是
    `Unrecognized request argument`/`extra fields not permitted`。故以错误
    类别为主触发：带了 thinking 时任何 400/422 均视为「本后端不支持该参数」。
    """
    if "thinking" in str(e).lower():
        return True
    try:
        from openai import BadRequestError
        if isinstance(e, BadRequestError):
            return True
    except Exception:
        pass
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    return isinstance(status, int) and status in (400, 422)


class LLMClient:
    """多后端 LLM 客户端。"""

    def __init__(self, settings: Settings, client=None) -> None:
        self.s = settings
        self._client = client
        self.backend = "deepseek" if client is not None else "uninitialized"

    def _default_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        key = os.environ.get("API") or os.environ.get("QWEN_API") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("未找到 LLM API key")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
        cached = _CLIENT_CACHE.get((base, key))
        if cached is None:
            # 显式超时：OpenAI SDK 默认 600s，会让主通道长时间挂死而不降级。
            # max_retries=0：SDK 内部重试（默认 2 次）与三级降级链重复，且
            # 每次重试都可能再吃满一个超时，硬性 3 秒目标下必须交给自建降级链。
            cached = OpenAI(api_key=key, base_url=base,
                            timeout=self.s.llm_timeout_s, max_retries=0)
            _CLIENT_CACHE[(base, key)] = cached
        self._client = cached
        self.backend = "deepseek"
        return self._client

    @staticmethod
    def _request_timeout(deadline: float | None) -> float | None:
        """本链剩余预算 → 单次请求超时。

        ``deadline`` 为 absolute ``time.perf_counter()`` 时刻（None = 不设限，
        用后端默认超时）。每次真正发起请求时才计算，故「API 剥离 thinking 重试」
        这类二次请求不会各自吃满一份超时。
        """
        if deadline is None:
            return None
        return max(_MIN_REQUEST_TIMEOUT_S, deadline - time.perf_counter())

    def _generate_api(self, question: str, hits: list[Hit],
                      deadline: float | None = None) -> str:
        from rag04.generate.prompt import build_messages

        cli = self._default_client()
        kwargs: dict = dict(
            model=self.s.llm_model,
            messages=build_messages(question, hits),
            max_tokens=self.s.llm_max_tokens,
            temperature=0.0,
        )
        # 实测（deepseek-v4-flash 文本模型）：默认先输出思维链（观测
        # reasoning_tokens 86~870、reasoning_content 非空），首答 2.8~4.8s
        # 且有一次答案不完整；显式关闭 thinking 后降至 0.7~0.9s、答案完整。
        # 硬性要求「3 秒内作答」，故默认关闭；若服务端不识别该参数
        # （换 endpoint/模型），剥掉后立即重试，不影响降级链。
        tmo = self._request_timeout(deadline)
        if tmo is not None:
            kwargs["timeout"] = tmo          # 单次调用不得超过本链剩余预算
        try:
            resp = cli.chat.completions.create(
                **kwargs, extra_body={"thinking": {"type": "disabled"}})
        except Exception as e:
            if not _is_unsupported_param_error(e):
                raise
            logger.warning("服务端可能不识别 thinking 参数（%s），剥离后重试",
                           type(e).__name__)
            tmo = self._request_timeout(deadline)
            if tmo is not None:
                kwargs["timeout"] = tmo
            resp = cli.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _generate_ollama(self, question: str, hits: list[Hit],
                         deadline: float | None = None) -> str:
        import requests
        from rag04.generate.prompt import build_messages

        msgs = build_messages(question, hits)
        prompt = "\n\n".join(m["content"] for m in msgs)
        # 原为 llm_timeout_s * 4（默认 120s）：与 API 的 30s 叠加成 ~150s 的
        # 挂死窗口。此处同样收敛到本链剩余预算（见 _CHAIN_DEADLINE_S）。
        tmo = self._request_timeout(deadline)
        r = requests.post(
            f"{self.s.ollama_url.rstrip('/')}/api/generate",
            json={"model": self.s.llm_fallback_model, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.0}},
            timeout=self.s.llm_timeout_s * 4 if tmo is None else tmo,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        """生成答案。含拒答判断与三级降级。"""
        from rag04.generate.prompt import build_citations, detect_lang

        t0 = time.perf_counter()
        lang = detect_lang(question)

        if is_refusal_question(question, lang=lang):
            return Answer(
                question=question,
                answer=("抱歉，该问题与招股说明书内容无关，我无法作答。请提出与文档相关的问题。"
                        if lang == "zh" else
                        "Sorry, this question is unrelated to the prospectus documents. "
                        "Please ask something about the documents."),
                lang=lang, citations=[], hits=hits,
                latency_ms=(time.perf_counter() - t0) * 1000,
                llm_backend="refusal", refused=True,
                answerable=False, refusal_source="offtopic",
            )

        # 三级降级共享一个 deadline（C4）：挂死的后端最多吃掉本链预算，
        # 之后直接落到检索原文，而不是把请求线程钉在这条链上。
        deadline = t0 + min(self.s.llm_timeout_s, _CHAIN_DEADLINE_S)

        text, backend = "", ""
        try:
            text = self._generate_api(question, hits, deadline=deadline)
            backend = "deepseek"
        except Exception as e:
            logger.warning("API 生成失败，转本地 Ollama：%s: %s", type(e).__name__, e)

        if not text:
            left = deadline - time.perf_counter()
            if left > _MIN_REQUEST_TIMEOUT_S:
                try:
                    text = self._generate_ollama(question, hits, deadline=deadline)
                    backend = "ollama"
                except Exception as e:
                    logger.error("Ollama 亦失败，返回检索原文：%s: %s", type(e).__name__, e)
            else:
                logger.error("API 已用尽降级链预算（剩余 %.1fs），跳过 Ollama，"
                             "直接返回检索原文", left)

        if not text:
            text = self._retrieval_only_answer(question, hits, lang)
            backend = "retrieval_only"

        # RC7：结构化 answerable 字段为主判据；正则降为兜底。
        answerable, prose = parse_answerability(text)
        if answerable is not None:
            text = prose or text
        # 正则**单向兜底**：只增不减拒答 —— 字段称可答但正文是拒答时仍判拒答，
        # 否则「先列举再拒答」式答案（id531）会被结构化字段洗回假阳性。
        regex_refused = detect_answer_refusal(text)
        if answerable is False:
            refused, source = True, "structured"
        elif answerable is True:
            refused = regex_refused
            source = "structured+regex" if regex_refused else "structured"
        else:
            refused = regex_refused
            source = "regex"
        if backend == "retrieval_only":
            # C5（终审）：检索原文兜底**不是答案**。原始片段未经模型归纳，
            # 单要点问题（如 id531「法定代表人是谁」）只要命中的片段里出现该
            # 要点就会被 coverage 字面判对，虚高 answer_accuracy。故在源头记
            # refused（不计正确），refusal_source 保留 retrieval_only 以便
            # 报告区分「模型拒答」与「后端不可用」。
            source = "retrieval_only"
            refused = True

        return Answer(
            question=question, answer=text, lang=lang,
            citations=build_citations(hits), hits=hits,
            latency_ms=(time.perf_counter() - t0) * 1000,
            llm_backend=backend,
            # RC6/RC7：域内拒答同样要标记，否则评分会把「无法确定」当成答对
            refused=refused,
            answerable=answerable,
            refusal_source=source,
        )

    @staticmethod
    def _retrieval_only_answer(question: str, hits: list[Hit], lang: str) -> str:
        """最终兜底：把召回原文按页列出，保证不空手而归。"""
        if not hits:
            return "未检索到相关内容。" if lang == "zh" else "No relevant content retrieved."
        head = ("（模型服务不可用，以下为检索到的原始内容）\n\n" if lang == "zh"
                else "(LLM unavailable; showing raw retrieved content)\n\n")
        body = "\n\n".join(
            f"[{h.doc_id} p{h.page} {h.block_type}] {h.text[:600]}" for h in hits[:3]
        )
        return head + body
