"""Query 理解：意图识别、页码过滤、多轮改写、查询扩展与变体生成。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 检索优化（对应 设计/接口设计.md §2.10、优化方案设计.md §2.3 M3.7/M3.8）

两级改写（M3.7）：
1. 有 LLM 时用 ``query_rewrite_prompt.txt`` + 最近 3 轮历史生成独立问题（超时 0.5 s，失败自动降级）；
2. 无 LLM 时用**规则改写**（指代消解 + 主题槽位替换 + 语言桥接）。

查询扩展（M3.8）：领域同义词表（``language.ZH_SYNONYMS``）生成 2~4 个变体
（原句 / 去主体名 / 关键词串 / 同义替换），多路检索后按变体相对置信度平方衰减合并。
变体列表写入日志与 retrieval_traces，便于审计。
"""

from __future__ import annotations

import re
import threading

from app.core.config import get_settings
from app.core.language import (
    ZH_SYNONYMS,
    bridge_query_to_chinese,
    detect_english_intent,
    detect_language,
    english_keywords,
    issuer_anchored_query,
    resolve_answer_language,
)
from app.core.logging_conf import logger, trace
from app.core.text_utils import STOPWORDS, dedupe_keep_order, tokenize
from app.models.schemas import QueryAnalysis

#: 意图 -> 触发词（工单点名的字段类别）
INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "收入": ("收入", "营业收入", "主营业务收入", "营收"),
    "占比": ("占比", "比重", "比例", "构成", "百分之", "％", "%"),
    "标准": ("技术标准", "标准", "行业标准", "国家标准"),
    "上下游": ("上游", "下游", "上下游", "产业链"),
    "注册资本": ("注册资本", "股本", "实收资本"),
    "法定代表人": ("法定代表人", "法人代表", "董事长", "总经理"),
    "募资用途": ("募集资金", "募资", "募集资金运用", "募投项目", "补充流动资金"),
    "供应商": ("供应商", "供货商", "采购"),
    "客户": ("客户", "顾客", "销售对象"),
    "荣誉": ("科技进步奖", "科学技术进步奖", "获奖", "荣誉"),
}

#: 追问信号词（出现即认为可能依赖上文）
FOLLOWUP_MARKERS = ("它", "他", "她", "其", "该", "上述", "这个", "那个", "这", "那", "此", "同样", "另外")
#: 页码过滤表述
PAGE_PATTERNS = (
    re.compile(r"第\s*(\d{1,4})\s*页"),
    re.compile(r"[pP]\.?\s*(\d{1,4})\b"),
    re.compile(r"[pP]age\s*(\d{1,4})\b", re.IGNORECASE),
)
#: 多子问题切分标志
SPLIT_MARKERS = ("？", "?", "；", ";")
COMPANY_NAME = "武汉兴图新科电子股份有限公司"
#: 主体名（去掉公司全称后用于变体）
SUBJECT_PATTERN = re.compile(r"(武汉[\u4e00-\u9fff]{2,10})?(股份有限公司|有限公司|公司)")


class QueryUnderstanding:
    """规则为主、LLM 可选的 Query 理解器。"""

    def __init__(self, use_llm: bool = False) -> None:
        self._settings = get_settings()
        self._use_llm = use_llm

    # ------------------------------------------------------------------
    def detect_intent(self, question: str) -> tuple[str, list[str]]:
        """识别意图与命中关键词（命中多个时取触发词最长的类别）。"""
        try:
            text = question or ""
            best = ("其他", [])
            best_len = 0
            for intent, triggers in INTENT_PATTERNS.items():
                hits = [word for word in triggers if word in text]
                if not hits:
                    continue
                score = sum(len(word) for word in hits)
                if score > best_len:
                    best = (intent, hits)
                    best_len = score
            return best
        except Exception:
            logger.exception("app.core.query_understanding", "意图识别失败")
            return "其他", []

    def detect_ambiguities(self, question: str) -> tuple[list[str], list[str]]:
        """识别模糊表述并给出提示（工单"模糊问题识别"能力）。"""
        try:
            text = question or ""
            ambiguities: list[str] = []
            notes: list[str] = []
            if re.search(r"第\s*\d+\s*页", text) is None and any(word in text for word in ("那一页", "哪一页")):
                ambiguities.append("未指定页码")
                notes.append("可在问题中写明「第 N 页」以启用页码过滤")
            if len(text) <= 6 and not any(word in text for word in INTENT_PATTERNS):
                ambiguities.append("问题过短")
                notes.append("问题过短可能缺少主体，已尝试结合多轮历史改写")
            if any(word in text for word in ("多少", "几个")) and not detect_language(text) == "en":
                if not any(word in text for word in ("收入", "占比", "注册资本", "募集资金", "客户", "供应商")):
                    ambiguities.append("对象不明确")
                    notes.append("缺少明确对象，已按 Top-5 片段作答")
            return ambiguities, notes
        except Exception:
            logger.exception("app.core.query_understanding", "模糊度识别失败")
            return [], []

    def decompose(self, question: str) -> list[str]:
        """把并列式问题拆成子问题（"分别是多少""以及"等）。"""
        try:
            text = question or ""
            parts = [part.strip() for part in re.split(r"[？?；;]", text) if part.strip()]
            if len(parts) <= 1:
                return []
            return [part + "？" if not part.endswith("？") else part for part in parts[1:]][:3]
        except Exception:
            logger.exception("app.core.query_understanding", "子问题分解失败")
            return []

    def extract_page_filter(self, question: str) -> list[int]:
        """提取"第 479 页 / p479 / page 479"形式的页码过滤条件。"""
        try:
            text = question or ""
            pages: list[int] = []
            for pattern in PAGE_PATTERNS:
                for match in pattern.finditer(text):
                    value = int(match.group(1))
                    if 1 <= value <= 2000 and value not in pages:
                        pages.append(value)
            return pages
        except Exception:
            logger.exception("app.core.query_understanding", "页码过滤提取失败")
            return []

    # ------------------------------------------------------------------
    def _topic_terms(self, history: list[tuple[str, str]]) -> list[str]:
        """从历史中抽取主题词（用于指代消解）。"""
        terms: list[str] = []
        for user_text, _ in history[-3:]:
            if COMPANY_NAME in user_text:
                terms.append(COMPANY_NAME)
            matches = SUBJECT_PATTERN.findall(user_text or "")
            for head, _suffix in matches:
                if head:
                    terms.append(f"{head}股份有限公司")
            for token in tokenize(user_text or ""):
                if token in STOPWORDS or len(token) < 2 or token.isdigit():
                    continue
                if token in FOLLOWUP_MARKERS:
                    continue
                terms.append(token)
        return dedupe_keep_order(terms)[:8]

    @trace
    def rewrite_with_history(self, question: str, history: list[tuple[str, str]]) -> tuple[str, bool]:
        """把追问改写为**独立问题**（确定性规则，不依赖 LLM）。

        触发条件（二者其一，且必须存在历史）：
        1. 含指代词（它/其/该/上述/这个/那个…）；
        2. 省略主语的省略式追问（以「那/那么/还有/另外」开头或以「呢」结尾）。

        改写动作：指代消解 → 主体补全（招股书是**单一发行人**文档，缺主体时默认补发行人全称）
        → 去掉句首话语标记 → 把「…呢？」规范成「…是多少？」。产出必须是可直接检索的完整问句。
        """
        try:
            text = (question or "").strip()
            if not text or not history:
                return text, False
            has_pronoun = any(marker in text for marker in FOLLOWUP_MARKERS)
            ellipsis = text.startswith(("那", "那么", "还有", "另外", "然后")) or text.rstrip("？?。").endswith("呢")
            if not (has_pronoun or ellipsis):
                return text, False

            rewritten = text
            # 1) 指代消解：指代词替换为发行人全称（单一发行人文档的默认主体）
            for marker in ("该公司", "这个公司", "上述公司", "这家公司", "它", "他", "她", "其", "该公司"):
                rewritten = rewritten.replace(marker, COMPANY_NAME)
            # 2) 主体补全：仍无主体（全称/简称都不在）时补上发行人全称
            if not self._has_subject(rewritten):
                rewritten = f"{COMPANY_NAME}{rewritten}"
            # 3) 去句首话语标记（"那/那么/还有/然后"等只起衔接作用，进检索会稀释语义）
            rewritten = re.sub(r"^(?:那|那么|还有|另外|然后|以及)+", "", rewritten).strip()
            # 4) 规范省略式问尾："…的注册资本呢？" → "…注册资本是多少？"
            rewritten = re.sub(r"的?\s*呢\s*[？?。]?$", "是多少？", rewritten)
            rewritten = re.sub(r"\s+", "", rewritten)
            changed = rewritten.strip() != text
            if changed:
                logger.info(
                    "app.core.query_understanding",
                    "多轮追问已改写为独立问题",
                    original=text[:80],
                    rewritten=rewritten[:120],
                )
            return rewritten.strip(), changed
        except Exception:
            logger.exception("app.core.query_understanding", "多轮改写失败，回退原问题")
            return question, False

    @staticmethod
    def _has_subject(text: str) -> bool:
        """问句里是否已含可识别主体（发行人全称 / 简称 / 公司自称）。"""
        raw = text or ""
        if COMPANY_NAME in raw or "兴图新科" in raw:
            return True
        return any(word in raw for word in ("本公司", "发行人", "公司"))

    # ------------------------------------------------------------------
    def expand_queries(self, question: str, rewritten: str = "") -> list[tuple[str, str]]:
        """生成查询变体：``[(查询串, 变体类型)]``（去重、2~4 条）。"""
        try:
            base = (rewritten or question or "").strip()
            if not base:
                return []
            variants: list[tuple[str, str]] = [(base, "original")]
            # 1) 去主体名（实测去掉公司全称后语义相似度更高）
            stripped = base.replace(COMPANY_NAME, " ").replace("武汉兴图新科电子股份有限公司", " ")
            stripped = re.sub(r"\s+", " ", stripped).strip(" ，,、的")
            if stripped and stripped != base:
                variants.append((stripped, "no_subject"))
            # 2) 关键词串（实义词拼接，突出字段名）
            keywords = [token for token in tokenize(base) if token not in STOPWORDS and len(token) > 1]
            if keywords:
                keyword_query = " ".join(dedupe_keep_order(keywords)[:12])
                if keyword_query and keyword_query != base:
                    variants.append((keyword_query, "keywords"))
            # 3) 同义替换
            synonyms = [word for key, values in ZH_SYNONYMS.items() if key in base for word in values]
            if synonyms:
                synonyms = dedupe_keep_order(synonyms)[:4]
                variants.append((f"{base} {' '.join(synonyms)}", "synonym"))
            return dedupe_keep_order_by_text(variants)[:4]
        except Exception:
            logger.exception("app.core.query_understanding", "查询扩展失败")
            return [(question, "original")] if question else []

    # ------------------------------------------------------------------
    def search_queries(self, analysis: QueryAnalysis) -> list[tuple[str, str]]:
        """由分析结果给出实际检索用的变体列表。

        **追问场景的关键点**：``analysis.search_query`` 在中文追问时等于"改写后的独立问句"
        （见 ``analyze``），因此这里用 ``search_query`` 作为基串，而不是拿带指代词的原始问句去检索。

        **发行人锚定**：招股说明书是**单一发行人**文档，发行人自身即默认主体；
        因此只要基串不含发行人可识别名称，就**总是**补一个 ``issuer_anchor`` 变体
        （不再以"问题串里恰好含全称"为条件——那正是 t12 缺陷的根因）。
        """
        try:
            base = analysis.search_query or analysis.rewritten or analysis.original
            variants = self.expand_queries(analysis.original, base)
            if analysis.language == "en" and analysis.search_query:
                variants.insert(0, (analysis.search_query, "language_bridge"))
            anchored = issuer_anchored_query(base)
            if anchored and anchored != base:
                insert_at = 1 if (analysis.language == "en" and analysis.search_query) else 0
                variants.insert(insert_at, (anchored, "issuer_anchor"))
            return dedupe_keep_order_by_text(variants)[:4]
        except Exception:
            logger.exception("app.core.query_understanding", "检索变体生成失败")
            return [(analysis.original, "original")]

    # ------------------------------------------------------------------
    @trace
    def analyze(self, question: str, history: list[tuple[str, str]] | None = None) -> QueryAnalysis:
        """完整 Query 理解：语言 → 改写 → 意图 → 扩展 → 页码过滤。"""
        try:
            text = (question or "").strip()
            language = detect_language(text)
            answer_language = resolve_answer_language(text, self._settings.language.default_answer_language)
            intent, keywords = self.detect_intent(text)
            # 英文意图判定（中文规则对英文无效，交给 language 的英文规则补齐）
            english_intent = detect_english_intent(text) if language == "en" else ""
            search_query, bridge = (text, "none")
            if language == "en":
                # 意图 → **纯中文问句**桥接（t11 修复：不得产出中英混排串）
                search_query, bridge = bridge_query_to_chinese(text, intent=english_intent or intent)
                if not intent or intent == "其他":
                    intent = english_intent or intent
                    keywords = dedupe_keep_order([*keywords, *english_keywords(text)])
            rewritten, changed = self.rewrite_with_history(text, history or [])
            # 中文追问：**检索串必须用改写后的独立问句**，否则会拿带指代词的句子去检索
            # （t12 根因之一：改写结果只写进 rewritten 字段、却未进入检索变体）
            if language == "zh" and changed and rewritten:
                search_query, bridge = rewritten, "history_rewrite"
            ambiguities, _notes = self.detect_ambiguities(text)
            expansions = [word for key, values in ZH_SYNONYMS.items() if key in text for word in values]
            return QueryAnalysis(
                original=text,
                rewritten=rewritten if changed else "",
                intent=intent,
                keywords=dedupe_keep_order(keywords),
                expansions=dedupe_keep_order(expansions)[:6],
                sub_questions=self.decompose(text),
                ambiguities=ambiguities,
                page_filter=self.extract_page_filter(text),
                is_followup=bool(history) and changed,
                language=answer_language,
                search_query=search_query,
                language_bridge=bridge,
            )
        except Exception:
            logger.exception("app.core.query_understanding", "Query 理解失败，退回原问题")
            return QueryAnalysis(original=question, language="zh", search_query=question, language_bridge="none")


def dedupe_keep_order_by_text(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """按查询串去重并保持顺序。"""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for text, kind in items:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((key, kind))
    return out


_understanding: QueryUnderstanding | None = None
_lock = threading.Lock()


def get_query_understanding(use_llm: bool = False) -> QueryUnderstanding:
    """获取进程级 Query 理解器单例。"""
    global _understanding
    with _lock:
        if _understanding is None or use_llm:
            _understanding = QueryUnderstanding(use_llm=use_llm)
    return _understanding
