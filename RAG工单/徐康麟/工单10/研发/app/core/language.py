# -*- coding: utf-8 -*-
"""工单4 语言识别与中英文双路径支持。

工单编号：人工智能NLP-RAG 项目-金融问答系统部署

职责：
    * ``detect_language`` —— 按 CJK / 拉丁字符占比判定 ``zh`` / ``en`` / ``mixed``；
    * ``resolve_language`` —— 合并「用户显式指定 --lang」与「自动识别」；
    * ``english_variants`` —— 为英文提问生成检索用中文候选（**由 LLM 生成，失败即降级**，
      绝不在本地硬编码中英词典，避免把语料事实写死进代码）。

设计依据（设计/接口设计.md §2.2）：``--lang zh|en|auto``；两条路径共用同一检索与合成逻辑，
英文路径额外做「问题→中文检索式」的翻译，并用英文作答；**引用页码口径不变**。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def _lazy_logger(logger: Any, module: str = "language") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class LanguageInfo:
    """语言判定结果。"""

    requested: str
    resolved: str
    cjk_chars: int
    latin_chars: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_language(text: str) -> str:
    """按字符占比判定语言：``zh``（含中文）/ ``en``（纯拉丁）/ ``mixed``（无法判定）。"""
    cjk = len(_CJK.findall(text or ""))
    latin = len(_LATIN.findall(text or ""))
    if cjk > 0 and latin > cjk:
        return "mixed"
    if cjk > 0:
        return "zh"
    if latin > 0:
        return "en"
    return "mixed"


def resolve_language(question: str, requested: str = "auto", *, logger: Any = None) -> LanguageInfo:
    """合并用户指定与自动识别。

    * ``requested == "auto"``：按 ``detect_language`` 判定（中文优先，mixed 按中文处理）；
    * 显式 ``zh``/``en``：以用户为准，但在用户选择与文本明显不符时写 WARNING 留痕。
    """
    log = _lazy_logger(logger)
    with log.enter("resolve_language", {"requested": requested,
                                        "question_digest": (question or "")[:60]}) as span:
        cjk = len(_CJK.findall(question or ""))
        latin = len(_LATIN.findall(question or ""))
        detected = detect_language(question)
        wanted = str(requested or "auto").strip().lower()
        if wanted in {"zh", "en"}:
            resolved = wanted
            reason = f"用户显式指定 {wanted}"
            if wanted == "en" and cjk > latin:
                log.log_event("language.mismatch", level="WARNING", requested=wanted, detected=detected,
                              cjk=cjk, latin=latin, action="按用户指定执行，但如实留痕")
        else:
            resolved = "zh" if detected in {"zh", "mixed"} else "en"
            reason = f"自动识别 detected={detected}"
        info = LanguageInfo(requested=wanted, resolved=resolved, cjk_chars=cjk,
                            latin_chars=latin, reason=reason)
        log.log_event("language.detected", **info.to_dict())
        span.set_output(info.to_dict())
        return info


def english_variants(question: str, *, llm_client: Any = None, cfg: Any = None,
                     logger: Any = None) -> dict[str, Any]:
    """英文提问 → 中文检索式（LLM 翻译 + **结构保全校验 + 一次重试 + 缓存**）。

    T7 修复的根因（T6 §8.1 缺口2/缺口3）：旧实现只取「LLM 返回的第一行」，
    译文一旦漏掉半句（如「…how many sales offices make up the Key Account Sales Department」
    被翻没），后续合成器只能拿到半句证据 → 英文答案不完整 / 图像题被拒答。
    现在：
        ① **数字与子句保全校验**：英文原句里的数字/问句子句必须出现在译文里（数字是硬指标）；
        ② 未通过 → 带失败原因**重试一次**（仍 temperature=0）；
        ③ 仍不通过 → 标记 ``translation_degraded=True`` 并返回最佳候选（由上层显式提示，不静默）；
        ④ 结果进「改写缓存」（第三级缓存），同问句第二次不再调 LLM。

    返回 ``{"zh_query", "backend", "elapsed_ms", "error", "attempts", "translation_degraded",
    "missing_digits", "cached"}``。
    """
    log = _lazy_logger(logger)
    with log.enter("english_variants", {"question_digest": (question or "")[:60],
                                        "has_llm": llm_client is not None}) as span:
        import time

        started = time.perf_counter()
        out: dict[str, Any] = {"zh_query": None, "backend": "none", "elapsed_ms": 0.0, "error": None,
                               "attempts": 0, "translation_degraded": False, "missing_digits": [],
                               "cached": False}
        if not str(question or "").strip():
            out["error"] = "空问题"
            span.set_output(out)
            return out
        # ④ 改写缓存：同问句直接复用译文（含质量标记）
        cache = None
        try:
            from .cache import get_cache

            cache = get_cache(cfg, logger=log)
            if cache.enabled:
                hit, value = cache.rewrite.get(cache.rewrite_key(question, kind="en2zh"))
                if hit and isinstance(value, dict) and value.get("zh_query"):
                    return {**value, "cached": True}
        except Exception as exc:  # noqa: BLE001 —— 缓存不可用不得阻断翻译
            log.log_event("language.rewrite_cache_failed", level="WARNING",
                          error=f"{type(exc).__name__}: {exc}")

        if llm_client is None:
            out["error"] = "未提供 LLM 客户端（英文路径退化为多语向量检索）"
            log.log_event("language.en_no_llm", level="WARNING", **out)
            span.set_output(out)
            return out
        from .prompts_loader import load_prompt  # 延迟导入，避免循环

        base_template = load_prompt("en_translate_prompt", logger=log)
        best: str = ""
        for attempt in range(1, 3):
            template = base_template if attempt == 1 else _RETRY_TEMPLATE
            prompt = template.replace("{question}", question or "")
            if attempt == 2:
                prompt = prompt.replace("{previous}", best or "（空）").replace(
                    "{missing}", "、".join(out["missing_digits"]) or "（结构不完整）")
            try:
                result = llm_client.generate_full(prompt, max_tokens=120, temperature=0.0)
                text = _clean_query(result.text or "")
                out["attempts"] = attempt
                out["backend"] = result.backend
                if len(text) > len(best):
                    best = text
                ok, missing = translation_preserved(question or "", best)
                out["missing_digits"] = missing
                out["translation_degraded"] = not ok
                log.log_event("language.en_translation_check", attempt=attempt, ok=ok,
                              missing=missing[:6], chars=len(best))
                if ok:
                    break
            except Exception as exc:  # noqa: BLE001 —— 翻译失败必须降级且留痕，绝不阻断问答
                out["error"] = f"{type(exc).__name__}: {exc}"
                log.log_event("language.en_translate_failed", level="WARNING", error=out["error"])
                break
        out["zh_query"] = best or None
        out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if not best:
            out["error"] = out["error"] or "LLM 返回空翻译"
        if cache is not None and best:
            try:
                cache.rewrite.put(cache.rewrite_key(question, kind="en2zh"),
                                  {k: v for k, v in out.items() if k != "cached"},
                                  saved_ms=out["elapsed_ms"])
            except Exception as exc:  # noqa: BLE001
                log.log_event("language.rewrite_cache_put_failed", level="WARNING",
                              error=f"{type(exc).__name__}: {exc}")
        span.set_output({k: v for k, v in out.items() if k != "zh_query"})
        return out


#: 翻译重试提示词（内嵌常量）：只在第一次译文「丢内容」时使用。
#: 说明：主提示词仍读 ``研发/app/prompts/en_translate_prompt.txt``；
#: 重试提示词仅失败路径需要，内嵌在本模块内以免把 T7 的改动扩散到提示词目录。
_RETRY_TEMPLATE = """你是中英翻译器。上一次翻译**漏掉了原问题的部分内容**，请重译一遍。

要求（逐条遵守）：
1. 只输出这一句中文，**不要任何标签/前缀**（如「公司名：」「翻译：」）、不要解释、不要引号、不要换行；
2. **原问题的每一个分句都要翻到**，包括「and …」后面的第二个问点；漏一个分句就算失败；
3. **原问题里出现的数字必须原样保留**（年份、比例、编号都要在）；
4. 保留原问法结构：「…有几个…构成」「…有哪些」「…分别是多少」「…最快 / …负增长」；
5. 公司名：能确定为中方发行人的中文全称就写中文全称；不确定就保留英文原文，不要音译编造；
6. 字段名/术语译成中文（注册资本 / 法定代表人 / 组织结构图 / 销售处 / 募集资金 / 军用领域 等）。

原问题（英文）：{question}
上一次的译文（**漏了内容，不要照抄**）：{previous}
上次漏掉的数字/结构提示：{missing}
重译后的中文检索式：
"""


#: 译文里必须保留的「结构信号」（英文问法的中文对应措辞；命中任一即算结构保留）
_STRUCTURE_HINTS: tuple[tuple[str, ...], ...] = (
    ("几个", "多少", "几家", "几项", "几"),          # 计数
    ("哪些", "什么", "哪一", "哪个", "是何", "是谁"),  # 列举/指认
    ("分别", "依次"),                                # 多点数值
    ("组织", "结构", "构成", "下设"),                 # 层级
    ("上游", "下游"),                                # 上下游
    ("最快", "负增长", "最高", "最低", "增长"),        # 结论
    ("标准", "规范"),                               # 标准名
    ("工程", "奖项", "一等奖"),                       # 工程/奖项
    ("注册资本", "法定代表人", "募集资金", "发行股数", "股权", "比重", "比例", "金额", "收入"),
)


def translation_preserved(english: str, chinese: str) -> tuple[bool, list[str]]:
    """译文结构保全校验：**数字必须全在**；英文问句的每个子句要能对应到某类结构信号。

    返回 ``(是否通过, 缺失数字列表)``。判据刻意保守：只惩罚「明显丢内容」的翻译，
    不做语义相似度判断（那会引入新的不确定性）。
    """
    import re as _re

    en = str(english or "")
    zh = str(chinese or "").replace(" ", "")
    if not zh:
        return False, []
    missing: list[str] = []
    for number in _re.findall(r"\d+(?:[.,]\d+)?", en):
        key = number.replace(",", "")
        if key not in zh.replace(",", ""):
            missing.append(number)
    if missing:
        return False, missing
    clauses = [c for c in _re.split(r"(?:[,;]|\band\b|\bor\b|\?|？)", en) if len(c.strip()) > 8]
    if len(clauses) <= 1:
        return True, []
    covered = sum(1 for group in _STRUCTURE_HINTS if any(word in zh for word in group))
    # 多子句英文问句：至少要有「子句数-1」类结构信号（例如计数 + 列举 = 2 类）
    return covered >= min(2, len(clauses) - 1), missing


#: 翻译结果里需要剥掉的前缀标签（T5 实测：模型会输出「公司名：…」这类标签，污染检索式）
_LABEL_PREFIX = re.compile(r"^[^：:]{1,12}[：:]\s*")


def _clean_query(text: str) -> str:
    """清洗 LLM 译文：取首个非空行、剥掉「标签：」前缀与引号。"""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    query = lines[0]
    query = _LABEL_PREFIX.sub("", query) if _LABEL_PREFIX.match(query) else query
    return query.strip().strip("\"'「」“”").strip()
