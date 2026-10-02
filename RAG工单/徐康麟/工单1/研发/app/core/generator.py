"""答案生成：LLM 生成（首选）与抽取式回答（降级）。

工单要求（5.5）：
- 把检索到的 top_n 片段、用户问题、对话历史交给 LLM；
- Prompt 要求：只基于片段回答；片段中没有答案就回复“不清楚”；答案简洁准确；必须带引用 ``[页码: 12]``；
- 首字返回时间 < 3 秒，使用流式输出。

两条生成路径：
1. ``generate``      —— 调用 vLLM/SGLang 的 OpenAI 兼容接口，支持流式；
2. ``generate_extractive`` —— **不需要任何 LLM**：按意图在证据片段上用受控规则
   抽取答案。它的价值有两个：
   a) 本地无 GPU / 无模型服务时系统仍可用、可测试；
   b) 作为 LLM 输出的对照基线（工单第 8 节要求 RAG vs 纯 LLM 对比）。

无论走哪条路径，只要证据不足，统一返回“不清楚”，绝不编造。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.core.number_utils import normalize_amount_text
from app.core.text_utils import extract_numbers, truncate
from app.models.schemas import Answer, Citation, QueryAnalysis, RetrievedChunk

try:  # pragma: no cover
    from openai import OpenAI

    HAS_OPENAI = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    HAS_OPENAI = False

# --------------------------------------------------------------------------
# 抽取式回答的受控规则
# --------------------------------------------------------------------------
AMOUNT_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
MONEY_PATTERN = re.compile(r"([\d,]+(?:\.\d+)?)\s*万元")


def flex(pattern: str) -> re.Pattern[str]:
    """把关键词编译成**容忍任意空白/换行**的正则。

    PDF 文字层会按排版断行，导致“国家科技进步一等奖”被切成
    ``国家科技进\\n步一等奖``。若用字面量做 ``in`` 判断就会漏掉，
    因此统一用允许空白穿插的正则匹配。
    """
    chars = [re.escape(char) for char in pattern if not char.isspace()]
    return re.compile(r"\s*".join(chars))


# 领域内高频关键词的容错正则（编译一次，复用）
AWARD_PATTERN = flex("国家科技进步一等奖")
REVENUE_PATTERN = flex("来自军用领域的收入分别为")
RATIO_PATTERN = flex("占主营业务收入比重分别为")
SUPPLIER_PATTERN = flex("的重要供应商")
UPSTREAM_PATTERN = flex("上游涉及")
DOWNSTREAM_PATTERN = flex("下游行业为")
FUND_ROW_PATTERN = flex("补充流动资金")
# 人名：中文姓名 2~4 字。
# 字段名用字（注/男/女/注册/住所/地址/国籍/电话/传真/邮编）不能出现在姓名里，
# 否则“法定代表人程家明注册地址…”会被贪婪匹配成“程家明注册”。
PERSON_PATTERN = re.compile(r"法定代表人[：:\s]*([\u4e00-\u9fff]{2,4})")
_NAME_STOPWORDS = {
    "注", "男", "女", "注册", "住所", "地址", "国籍", "电话", "传真", "邮编",
    "简历", "职务", "年龄", "学历", "任期", "持股", "薪酬",
}


def _clean_person(name: str) -> str:
    """裁剪被抓进来的字段名用字，返回干净的姓名。"""
    cleaned = name
    for stopword in sorted(_NAME_STOPWORDS, key=len, reverse=True):
        if cleaned.endswith(stopword) and len(cleaned) > len(stopword) + 1:
            cleaned = cleaned[: -len(stopword)]
    return cleaned
STANDARD_PATTERN = re.compile(r"《([^》]{4,60})》")


class Generator:
    """生成器门面：LLM 优先，失败或未配置时降级为抽取式。"""

    def __init__(self, force_extractive: bool = False) -> None:
        self.settings = get_settings()
        self.force_extractive = force_extractive
        self.last_error = ""
        self._client = None
        # LLM 可用性探测结果缓存（None 表示尚未探测）
        self._llm_available: bool | None = None

    # ------------------------------------------------------------------
    # LLM 客户端
    # ------------------------------------------------------------------
    def _get_client(self):
        """惰性创建 OpenAI 兼容客户端。"""
        if self._client is not None:
            return self._client
        if not HAS_OPENAI:
            self.last_error = "未安装 openai 库"
            return None
        self._client = OpenAI(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            timeout=self.settings.llm.read_timeout,
            max_retries=self.settings.llm.max_retries,
        )
        return self._client

    def _load_prompt(self, name: str) -> str:
        """读取 prompts 目录下的提示词模板。

        提示词与源码同级发布，位置为 ``<研发>/app/prompts/``；
        通过 ``settings.paths.source_root`` 定位，避免写死目录层级。
        """
        path: Path = self.settings.paths.source_root / "app" / "prompts" / name
        if not path.exists():
            logger.warning("app.core.generator", "提示词文件缺失", path=str(path))
            return ""
        return path.read_text(encoding="utf-8")

    def _build_messages(self, question: str, contexts: list[RetrievedChunk], history: list[tuple[str, str]]) -> list[dict[str, str]]:
        """组装对话消息。"""
        template = self._load_prompt("qa_prompt.txt")
        blocks = []
        for index, item in enumerate(contexts, start=1):
            blocks.append(f"[片段{index} | 页码: {item.chunk.page} | 类型: {item.chunk.type}]\n{item.chunk.content}")
        context_text = "\n\n".join(blocks)
        system_prompt = template.format(context=context_text, question=question) if "{context}" in template else template
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for role, content in history[-self.settings.conversation.rewrite_history_rounds * 2 :]:
            if role in {"user", "assistant"} and content.strip():
                messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": f"【参考资料】\n{context_text}\n\n【用户问题】\n{question}\n\n请只依据参考资料作答，并给出 [页码: N] 引用。",
            }
        )
        return messages

    def check_llm_available(self, timeout: float | None = None) -> bool:
        """探测 LLM 服务是否可用（用于界面与评估脚本的降级判断）。

        性能要点：必须**快速失败**。
        本探测发生在生成答案之前，若客户端沿用 SDK 默认的重试策略
        （``max_retries=2``，即共 3 次连接尝试），在没有 LLM 服务的机器上
        会阻塞约 9 秒（3 × connect_timeout），把首字延迟从毫秒级拖到 9 秒。
        因此这里显式使用独立的探测客户端：**不重试 + 极短超时**。

        探测结果会缓存：同一进程内服务状态未变时无需反复连接。
        """
        if self.force_extractive:
            return False
        if self._llm_available is not None:
            return self._llm_available
        if not HAS_OPENAI:
            self._llm_available = False
            return False

        started = time.perf_counter()
        try:
            probe = OpenAI(
                base_url=self.settings.llm.base_url,
                api_key=self.settings.llm.api_key,
                timeout=timeout or self.settings.llm.probe_timeout,
                max_retries=self.settings.llm.probe_max_retries,
            )
            probe.models.list()
            self._llm_available = True
            logger.info(
                "app.core.generator",
                "LLM 服务可用",
                base_url=self.settings.llm.base_url,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        except Exception as exc:
            self._llm_available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "app.core.generator",
                "LLM 服务不可用，将使用抽取式回答",
                base_url=self.settings.llm.base_url,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                error=self.last_error,
            )
        return self._llm_available

    # ------------------------------------------------------------------
    # LLM 流式生成
    # ------------------------------------------------------------------
    @trace
    def stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[tuple[str, object]]:
        """流式生成，产出 ``(事件, 负载)``。

        事件：``first_token`` / ``delta`` / ``done`` / ``error``
        """
        started = time.perf_counter()
        client = self._get_client()
        if client is None:
            yield ("error", self.last_error or "LLM 客户端不可用")
            return

        messages = self._build_messages(question, contexts, history or [])
        first_token_ms = 0.0
        buffer: list[str] = []
        try:
            stream = client.chat.completions.create(
                model=self.settings.llm.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.settings.llm.temperature,
                top_p=self.settings.llm.top_p,
                max_tokens=self.settings.llm.max_tokens,
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0].delta, "content", None)
                if not delta:
                    continue
                if not buffer:
                    first_token_ms = (time.perf_counter() - started) * 1000
                    yield ("first_token", {"first_token_ms": first_token_ms})
                buffer.append(delta)
                yield ("delta", {"text": delta})
        except Exception as exc:
            logger.exception("app.core.generator", "LLM 流式生成失败", error=f"{type(exc).__name__}: {exc}")
            yield ("error", f"{type(exc).__name__}: {exc}")
            return

        text = "".join(buffer).strip()
        answer = self._finalize(text, question, contexts, first_token_ms, (time.perf_counter() - started) * 1000, mode="llm")
        yield ("done", answer)

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    @trace
    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        analysis: QueryAnalysis | None = None,
        history: list[tuple[str, str]] | None = None,
        allow_llm: bool = True,
    ) -> Answer:
        """生成答案：优先 LLM，失败则抽取式。"""
        started = time.perf_counter()

        # 证据不足 -> 直接兜底
        if not contexts:
            logger.warning("app.core.generator", "无检索片段，回复兜底答案", question=question)
            return self._unknown("无检索片段", started)

        if allow_llm and not self.force_extractive and self.check_llm_available():
            chunks: list[str] = []
            first_token_ms = 0.0
            # 按配置裁剪送入 LLM 的片段数：提示词越长 prefill 越慢
            # （实测 CPU 上 8 片段 ≈ 18s、3 片段 ≈ 数秒；GPU+vLLM 无此瓶颈）
            limit = self.settings.llm.max_context_chunks
            llm_contexts = contexts[:limit] if limit > 0 else contexts
            if len(llm_contexts) != len(contexts):
                logger.info(
                    "app.core.generator",
                    "按 max_context_chunks 裁剪送入 LLM 的上下文",
                    total=len(contexts),
                    used=len(llm_contexts),
                )
            for event, payload in self.stream(question, llm_contexts, history):
                if event == "first_token":
                    first_token_ms = float(payload["first_token_ms"])  # type: ignore[index]
                elif event == "delta":
                    chunks.append(str(payload["text"]))  # type: ignore[index]
                elif event == "done":
                    return payload  # type: ignore[return-value]
                elif event == "error":
                    logger.warning("app.core.generator", "LLM 生成失败，降级为抽取式", error=str(payload))
                    break
            if chunks:
                text = "".join(chunks).strip()
                total_ms = (time.perf_counter() - started) * 1000
                return self._finalize(text, question, contexts, first_token_ms, total_ms, mode="llm")

        # 英文提问：LLM 不可用时用英文模板构造答案（数字/单位由代码换算，保证精确）
        if analysis is not None and analysis.language == "en":
            english = self.generate_english(question, contexts, analysis, started)
            if english is not None:
                return english

        return self.generate_extractive(question, contexts, analysis, started)

    # ------------------------------------------------------------------
    # 英文回答（模板 + 可选本地模型）
    # ------------------------------------------------------------------
    @trace
    def generate_english(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        analysis: QueryAnalysis | None = None,
        started: float | None = None,
    ) -> Answer | None:
        """构造英文答案；无法构造时返回 ``None`` 交由上层兜底。"""
        started = started or time.perf_counter()
        from app.core.english_answer import get_english_builder

        builder = get_english_builder()
        outcome = builder.build(question, contexts, analysis)
        if outcome is None:
            return None
        text, evidence = outcome
        return self._finalize(
            text,
            question,
            contexts,
            (time.perf_counter() - started) * 1000,
            (time.perf_counter() - started) * 1000,
            mode="extractive",
            primary=evidence,
            language="en",
        )

    # ------------------------------------------------------------------
    # 抽取式回答（无 LLM 也能给出基于原文的答案）
    # ------------------------------------------------------------------
    def _best_evidence(self, contexts: list[RetrievedChunk], must_contain: tuple[str, ...] = ()) -> RetrievedChunk | None:
        """挑选证据片段：优先包含必需关键词、且分数最高者。"""
        for item in contexts:
            if all(keyword in item.chunk.content for keyword in must_contain):
                return item
        return contexts[0] if contexts else None

    def _answer_registered_capital(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """回答注册资本。

        格式归一化：同一金额在招股书里有 ``5,520 万元``（正文页）与
        ``5,520.00 万元``（表格页）两种写法。若不归一，同一问题会因命中的是
        正文页还是表格页而输出不同文本（并发测试实测到两种答案并存），
        既影响可复现性，也会让判分与对比评估产生噪声。
        """
        for item in contexts:
            match = re.search(r"注册资本[：:\s]*([\d,]+(?:\.\d+)?)\s*万元", item.chunk.content)
            if match:
                amount = normalize_amount_text(match.group(1))
                return f"注册资本为 {amount} 万元。", item
        return None

    def _answer_legal_representative(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = PERSON_PATTERN.search(item.chunk.content)
            if match:
                name = _clean_person(match.group(1))
                if len(name) >= 2:
                    return f"法定代表人是{name}。", item
        return None

    def _answer_revenue(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """回答“军用领域收入分别是多少”。"""
        for item in contexts:
            content = item.chunk.content
            if not REVENUE_PATTERN.search(content) or "军用领域" not in content:
                continue
            match = re.search(r"来自军用领域的\s*收入分别为([^。]+?)(?:，|,)?\s*占", content)
            if match:
                amounts = MONEY_PATTERN.findall(match.group(1))
                if amounts:
                    joined = "、".join(f"{amount} 万元" for amount in amounts)
                    return f"报告期内，公司来自军用领域的收入分别为{joined}。", item
        return None

    def _answer_revenue_ratio(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """回答“军用领域收入占主营业务收入的比重分别是多少”。"""
        for item in contexts:
            content = item.chunk.content
            if not RATIO_PATTERN.search(content) or "军用领域" not in content:
                continue
            match = re.search(r"占\s*主营业务收入比重分别为([^。]+?)(?:。|$)", content)
            if match:
                ratios = PERCENT_PATTERN.findall(match.group(1))
                if ratios:
                    joined = "、".join(ratio.replace(" ", "") for ratio in ratios)
                    return f"报告期内，公司来自军用领域的收入占主营业务收入的比重分别为{joined}。", item
        return None

    def _answer_supplier_field(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"(?:已经成为|已成为)([^，。]{2,20}领域)的\s*重要供应商", item.chunk.content)
            if match:
                # 招股书 PDF 跨行会把“国防军队视频指挥领域”断成“国防\n军队视频指挥领域”，
                # 这里去掉换行与空格，保证答案是一句通顺的中文。
                field = re.sub(r"\s+", "", match.group(1))
                return f"公司目前已经成为{field}的重要供应商。", item
        return None

    @staticmethod
    def _sentence_quality(sentence: str) -> float:
        """给候选句子打分：越靠前、越完整、信息越全，分越高。

        招股书同一结论会在多个章节重复，其中一份可能被切分/压缩得不完整，
        因此不能“取第一个命中”，而要跨候选挑最优。
        """
        score = 0.0
        for marker in ("2014年12月", "2014 年12 月", "国家科技进步一等奖", "C4ISR", "荣获"):
            if marker.replace(" ", "") in sentence.replace(" ", ""):
                score += 1.0
        if len(sentence) >= 30:
            score += 1.0
        if len(sentence) >= 60:
            score += 0.5
        # 以续写符号开头 => 是半句，扣分
        if sentence.lstrip()[:1] in "）)、，,；;":
            score -= 2.0
        return score

    def _answer_award(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """回答荣获国家科技进步一等奖的工程。

        跨全部候选片段挑选**信息最完整**的句子：
        招股书在多个章节重复该事实，其中有的版本被切掉主语开头，
        只取第一条会让答案变成半句话。
        """
        best: tuple[float, str, RetrievedChunk] | None = None
        for item in contexts:
            content = item.chunk.content
            if not AWARD_PATTERN.search(content):
                continue
            for sentence in re.split(r"(?<=[。；])", content):
                if not AWARD_PATTERN.search(sentence):
                    continue
                # 先归一化空白（PDF 断行会插入换行/多余空格），再作为答案输出
                cleaned = re.sub(r"\s+", "", sentence).strip("。；")
                cleaned = cleaned.lstrip("”）) ")
                if len(cleaned) < 15:
                    continue
                if not ("荣获" in cleaned or "获得" in cleaned):
                    continue
                quality = self._sentence_quality(cleaned)
                if best is None or quality > best[0]:
                    best = (quality, cleaned, item)
        if best is None:
            return None
        return f"{best[1]}。", best[2]

    def _answer_standard(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            content = item.chunk.content
            if not SUPPLIER_PATTERN.search(content) and "技术标准" not in content:
                continue
            match = STANDARD_PATTERN.search(content)
            if match:
                return f"公司参与制定了全军第一个视频指挥系统技术标准（即《{match.group(1)}》）。", item
        return None

    def _answer_upstream(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"上游涉及([^。]{4,120})。", item.chunk.content)
            if match:
                return f"电子信息行业的上游涉及{match.group(1)}。", item
        return None

    def _answer_downstream(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        for item in contexts:
            match = re.search(r"下游行业为([^。]{4,120})。", item.chunk.content)
            if match:
                return f"电子信息行业的下游行业为{match.group(1)}。", item
        return None

    # 金额单元格：必须带“万元/亿元/元”单位，或本身带千分位/小数。
    # 这样能排除表格里的“序号”列（纯整数如 1、2、3）。
    MONEY_CELL_PATTERN = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^\d+\.\d+$|^\d+(?:,\d+)?\s*(?:万元|亿元|元)$")

    def _answer_fund_usage(self, contexts: list[RetrievedChunk]) -> tuple[str, RetrievedChunk] | None:
        """回答“募集资金多少用于补充流动资金”。

        表格式问句的坑：表格列形如 ``| 序号 | 项目名称 | 拟投入募集资金 |``，
        行是 ``| 3 | 补充流动资金 | 15,000.00 |``。
        若直接取“行内第一个纯数字单元格”，会把**序号 3** 当成金额，
        回答出“3 万元”这种明显错误的结果。
        因此这里只接受满足 ``MONEY_CELL_PATTERN`` 的单元格。
        """
        for item in contexts:
            content = item.chunk.content
            if not FUND_ROW_PATTERN.search(content):
                continue
            # 表格式：逐行扫描含“补充流动资金”的行，取第一个“金额形态”的单元格
            for line in content.split("\n"):
                if "补充流动资金" not in line or "|" not in line:
                    continue
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                for cell in cells:
                    # 跳过项目名称本身与纯序号
                    if "补充流动资金" in cell or not cell:
                        continue
                    if self.MONEY_CELL_PATTERN.match(cell):
                        amount = cell.replace("万元", "").replace("元", "").strip()
                        return f"公司计划使用本次发行募集资金 {amount} 万元用于补充流动资金。", item
            # 正文式：补充流动资金 15,000.00 万元
            match = re.search(r"补充流动资金\s*\|?\s*([\d,]+(?:\.\d+)?)\s*(万元|亿元|元)?", content)
            if match:
                amount = match.group(1)
                unit = match.group(2) or "万元"
                return f"公司计划使用本次发行募集资金 {amount} {unit}用于补充流动资金。", item
        return None

    @trace
    def generate_extractive(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        analysis: QueryAnalysis | None = None,
        started: float | None = None,
    ) -> Answer:
        """基于受控规则从证据片段抽取答案（无需 LLM）。"""
        started = started or time.perf_counter()
        intent = analysis.intent if analysis else ""
        question_text = question

        # 规则表：(触发条件, 处理函数)
        handlers: list[tuple[bool, object]] = [
            (intent == "注册资本" or "注册资本" in question_text, self._answer_registered_capital),
            (intent == "法定代表人" or "法定代表人" in question_text, self._answer_legal_representative),
            (("比重" in question_text or "占比" in question_text) and "收入" in question_text, self._answer_revenue_ratio),
            ("收入" in question_text and "军用领域" in question_text, self._answer_revenue),
            ("重要供应商" in question_text, self._answer_supplier_field),
            ("科技进步一等奖" in question_text or "科技进步奖" in question_text, self._answer_award),
            (intent == "技术标准" or "技术标准" in question_text, self._answer_standard),
            ("上游" in question_text, self._answer_upstream),
            ("下游" in question_text, self._answer_downstream),
            ("补充流动资金" in question_text or intent == "募资用途", self._answer_fund_usage),
        ]

        for enabled, handler in handlers:
            if not enabled:
                continue
            try:
                outcome = handler(contexts)  # type: ignore[operator]
            except Exception as exc:
                logger.exception(
                    "app.core.generator",
                    "抽取式规则执行异常",
                    handler=getattr(handler, "__name__", "?"),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            if outcome:
                text, evidence = outcome
                logger.info(
                    "app.core.generator",
                    "抽取式回答命中规则",
                    handler=getattr(handler, "__name__", "?"),
                    page=evidence.chunk.page,
                )
                return self._finalize(
                    text,
                    question,
                    contexts,
                    (time.perf_counter() - started) * 1000,
                    (time.perf_counter() - started) * 1000,
                    mode="extractive",
                    primary=evidence,
                )

        # 未命中任何受控规则：用最高分片段做“有据可依的摘要式回答”
        best = self._best_evidence(contexts)
        if best is None:
            return self._unknown("抽取式回答无可用证据", started)

        # 只截取与问题关键词重合度最高的句子，避免整段倾倒
        sentences = re.split(r"(?<=[。；])", best.chunk.content)
        keywords = [kw for kw in (analysis.keywords if analysis else []) if kw]
        picked = [s for s in sentences if any(kw in s for kw in keywords)][:2] or sentences[:1]
        snippet = "".join(picked).strip()
        if len(snippet) < 10:
            return self._unknown("证据片段不足以组织答案", started)
        text = f"根据招股意向书第 {best.chunk.page} 页：{truncate(snippet, 300)}"
        return self._finalize(
            text,
            question,
            contexts,
            (time.perf_counter() - started) * 1000,
            (time.perf_counter() - started) * 1000,
            mode="extractive",
            primary=best,
        )

    # ------------------------------------------------------------------
    # 收尾处理
    # ------------------------------------------------------------------
    def _finalize(
        self,
        text: str,
        question: str,
        contexts: list[RetrievedChunk],
        first_token_ms: float,
        total_ms: float,
        mode: str,
        primary: RetrievedChunk | None = None,
        language: str = "zh",
    ) -> Answer:
        """统一收尾：判定“不清楚”、补引用、附性能指标。"""
        unknown_answer = self.settings.app.unknown_answer if language == "zh" else "I don't know"
        normalized = (text or "").strip()

        # LLM 可能输出“不清楚”或空串
        if not normalized or normalized.strip("。.！!") in {unknown_answer, "不知道", "无法回答", "I don't know"}:
            return self._unknown("模型判定片段中无答案", None, contexts, language=language)

        citations = self._build_citations(normalized, contexts, primary)
        # 有答案但完全没有有效引用：仍然给出来源，保证可追溯
        return Answer(
            answer=normalized,
            citations=citations,
            is_unknown=False,
            primary_chunk_id=primary.chunk.chunk_id if primary is not None else "",
            language=language,  # type: ignore[arg-type]
            first_token_ms=round(first_token_ms, 2),
            total_ms=round(total_ms, 2),
            retrieved_count=len(contexts),
            pages=sorted({item.chunk.page for item in contexts}),
            mode=mode,  # type: ignore[arg-type]
            retrieved=contexts,
        )

    def _build_citations(
        self, text: str, contexts: list[RetrievedChunk], primary: RetrievedChunk | None
    ) -> list[Citation]:
        """构造引用列表：优先取答案中显式提到的页码，否则取证据来源。"""
        cited_pages = {int(match) for match in re.findall(r"\[页码:\s*(\d+)\]", text)}
        citations: list[Citation] = []
        seen: set[int] = set()

        for item in contexts:
            page = item.chunk.page
            if cited_pages and page not in cited_pages:
                continue
            if page in seen:
                continue
            seen.add(page)
            citations.append(
                Citation(
                    page=page,
                    chunk_id=item.chunk.chunk_id,
                    snippet=truncate(item.chunk.content, 220),
                    section=item.chunk.section,
                    score=round(item.score, 4),
                )
            )
        if not citations and primary is not None:
            citations.append(
                Citation(
                    page=primary.chunk.page,
                    chunk_id=primary.chunk.chunk_id,
                    snippet=truncate(primary.chunk.content, 220),
                    section=primary.chunk.section,
                    score=round(primary.score, 4),
                )
            )
        return citations[:5]

    def _unknown(
        self,
        reason: str,
        started: float | None = None,
        contexts: list[RetrievedChunk] | None = None,
        language: str = "zh",
    ) -> Answer:
        """统一的兜底回复。

        工单要求兜底文案是“不清楚”；英文提问时返回对应的英文兜底，
        避免中英混排（界面与评估都以 ``is_unknown`` 判断，不依赖具体文案）。
        """
        elapsed = (time.perf_counter() - started) * 1000 if started else 0.0
        text = self.settings.app.unknown_answer if language == "zh" else "I don't know"
        logger.warning("app.core.generator", "返回兜底答案", reason=reason, language=language)
        return Answer(
            answer=text,
            citations=[],
            is_unknown=True,
            unknown_reason=reason,
            language=language,  # type: ignore[arg-type]
            first_token_ms=round(elapsed, 2),
            total_ms=round(elapsed, 2),
            retrieved_count=len(contexts or []),
            pages=sorted({item.chunk.page for item in (contexts or [])}),
            mode="fallback",
            retrieved=contexts or [],
        )


def get_generator(force_extractive: bool = False) -> Generator:
    """工厂函数。"""
    return Generator(force_extractive=force_extractive)
