# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单8 落地式生成：**确定性草稿 → LLM 润色 → 数值原子保全护栏**。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

为什么必须加这一层（实测教训，详见 ``优化/图谱抽取优化说明.md`` §3）：
    只用「证据块 + 自由生成」提示词时，3B 模型会**改写数字**。实测 Q07：
    证据块里明明是「营业收入 137,958（百万元）· 同比 18.2%」（即 1,379.58 亿元、拨备覆盖率 183.12%），
    模型却写成「营业收入 1,323.46 亿元、拨备覆盖率 188.75%」——关键词覆盖率直接归零，
    端到端严格准确率从 23.53% 掉到 11.76%。
处置（与 工单7 生成层同思想，但按 Graph RAG 的证据形态自行实现，避免跨工单私有接口耦合）：
    1. **确定性草稿**：从证据块里按「问句实词命中 + 是否含数值 + 证据排名」打分，抽出候选句子与
       **数值原子**（金额/比例/日期，含千分位与单位）；
    2. **LLM 润色**：把草稿与证据块一起给模型，要求「只能使用草稿与证据里的数字，不得改写」；
    3. **原子保全护栏**：逐个检查草稿里的数值原子是否出现在候选答案中；
       缺失 ⇒ 判定为「改写数字」，用草稿兜底（`llm_rejected`），并如实记录缺失原子；
    4. 所有路径都返回**可追溯到证据块**的文本（不编造）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 数值原子：金额（带千分位/小数/单位）、比例（%／个百分点／倍）、日期（年/月/日）、大额整数
ATOM_PATTERNS: tuple[str, ...] = (
    r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|百万元|元|亿美元|亿港元)",
    r"\d+(?:\.\d+)?\s*(?:%|个百分点|倍)",
    r"(?:19|20)\d{2}\s*年(?:\d{1,2}\s*月)?(?:\d{1,2}\s*日)?",
    r"\d[\d,]{2,}(?:\.\d+)?",
)
_ATOM_RE = re.compile("|".join(f"(?:{p})" for p in ATOM_PATTERNS))
#: 句子切分：只按**语义标点**切（。；）与**空行**切。
#: ⚠️ 不能按单个换行切（实测）：PDF 抽取的每个 text block 自带换行，
#:   正文里「对公、资金同业也进\n一步理顺了思路」会把一句话腰斩成两段，
#:   正确答案句因此被拆散、长度/权重门槛全部落空（Q01 三次迭代都卡在这里）。
#: 表格块额外按 ``|`` 切（工单7 的表格块是 Markdown 行，整行远超长度上限）。
_SENT_SPLIT = re.compile(r"[。；;]|\n\s*\n|\s\|\s")
#: PDF 换行/空格噪声：合并前先去掉（保证一句话是连续的）
_SOFT_BREAK = re.compile(r"[\s\u3000]+")


def _analysis_terms(analysis: Any) -> list[str]:
    """从问句理解结果里取**高置信实词**（用于草稿句子加权）。

    依据（实测）：``query_understanding`` 抽出的 ``field_terms`` / ``keywords`` 是
    「问的到底是什么字段」的直接信号（如 Q07 的 营业收入/净利润/不良贷款率/拨备覆盖率），
    用它给句子加权，能让「含真正答案字段」的句子稳定胜出，
    而不是靠 IDF 猜（IDF 只反映词的稀有度，不反映「它是不是问题的答案」）。
    """
    if analysis is None:
        return []
    out: list[str] = []
    for key in ("field_terms", "keywords", "subject_terms"):
        values = getattr(analysis, key, None)
        if isinstance(values, (list, tuple)):
            for value in values:
                text = str(value).strip()
                if 2 <= len(text) <= 12 and text not in out:
                    out.append(text)
    return out

GROUNDED_PROMPT = """你是严谨的金融年报问答助手。请**依据给定的证据块与草稿**，输出一段通顺的答案。

铁律（违反即视为失败）：
1. **所有数字必须原样照抄**草稿或证据块（含千分位与单位），不得换算、不得四舍五入、不得改动；
   草稿与证据里没有的数字，**一个都不要写**。
2. 只使用草稿与证据块里的信息，不得引入任何外部知识。
3. 保留草稿里每一项的要点（枚举题不要漏项），可以用更自然的语言组织。
4. 中文提问用中文回答，英文提问用英文回答；答案 ≤ 240 字，不要输出「根据证据」这类套话。

【证据块】
{evidence}

【确定性草稿（事实基准）】
{draft}

【用户问题】
{question}

【答案】
"""


@dataclass(slots=True)
class GroundedResult:
    """落地式生成结果（含护栏结论与实测耗时，供报告引用）。"""

    text: str
    method: str                       # llm_polished | draft_fallback | extractive_draft
    backend: str = "ollama"
    model: str = "qwen2.5:3b"
    atoms: list[str] = field(default_factory=list)
    missing_atoms: list[str] = field(default_factory=list)
    llm_candidate: str | None = None
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "backend": self.backend, "model": self.model,
                "atoms": self.atoms[:20], "atoms_total": len(self.atoms),
                "missing_atoms": self.missing_atoms[:20], "chars": len(self.text or ""),
                "elapsed_ms": round(self.elapsed_ms, 2), "error": self.error}


def numeric_atoms(text: str) -> list[str]:
    """抽取文本中的数值原子（去重保序；用于「原子保全护栏」）。"""
    out: list[str] = []
    for match in _ATOM_RE.finditer(str(text or "")):
        atom = re.sub(r"\s+", "", match.group(0))
        if atom and atom not in out:
            out.append(atom)
    return out


def atom_preserved(candidate: str, atoms: Sequence[str], *, keep_ratio: float = 1.0) -> tuple[bool, list[str]]:
    """候选答案是否保住了全部（或 ``keep_ratio`` 比例的）数值原子。

    ⚠️ 归一化口径（实测踩坑）：模型常把 ``1,379.58`` 写成 ``1379.58``、把 ``269,005.23万元``
    写成 ``269005.23万元``。若按字面比对，**正确改写**也会被判成「丢失原子」⇒ 全部退回草稿，
    LLM 形同没用上。因此比对前统一去掉千分位逗号与空白（**数值本身仍必须完全一致**）。
    """
    haystack = re.sub(r"\s+", "", str(candidate or "")).replace(",", "")
    missing = [atom for atom in atoms
               if re.sub(r"\s+", "", atom).replace(",", "") not in haystack]
    total = len(atoms)
    if total == 0:
        return True, []
    kept = total - len(missing)
    return (kept / total) >= float(keep_ratio), missing


def build_draft(question: str, chunks: Sequence[Mapping[str, Any]], *, analysis: Any = None,
                max_sentences: int = 8, max_chars: int = 900,
                min_word_hits: int = 1, min_sentence_mass: float = 0.5,
                max_sentence_chars: int = 220, field_weight: float = 2.0) -> str:
    """确定性草稿：按「问句实词命中 + **问句字段词**命中 + 含数值 + 证据排名」给句子打分。

    设计要点（可解释、可复现；都是实测调出来的）：
        * 只从**证据块原文**里抽句子，绝不生成新内容；
        * **空格归一**后再匹配：PDF 抽取会把词组切断（实测「对公、资金同业也进 一步理顺…」
          中间有空格，导致「资金同业」匹配失败、正确句子被漏掉）；
        * **问句理解的字段词**（营业收入/净利润/不良贷款率…）给 ``field_weight`` 倍权重 ——
          它直接回答「问的是哪个指标」，比 IDF 猜词更可靠（见 ``_analysis_terms``）；
        * 必须有实词命中才入选：否则表格块里的任意数字行都会进草稿
          （实测 Q01 的草稿被「资产重组承诺」表格里的 90.75%/269,005.23万元 占满）；
        * 同页最多 3 句，保证草稿覆盖多页证据；表格块给轻微加成（数值题常用）。
    """
    terms = _question_terms(question)
    weights = _term_weights(terms)
    field_terms = _analysis_terms(analysis)
    scored: list[tuple[float, int, str]] = []
    # **块级排序 + 只取最相关的若干块**（实测必需）：
    #   `expand_with_page_context` 会按页补进一堆表格块；若让它们都参与草稿，
    #   一个只因为含「平安」二字而擦边相关的「资产重组承诺」表（p81）会带来
    #   90.75% / 269,005.23万元 等**与问题无关**的数值原子，草稿与护栏全被它带偏。
    #   因此：① 块级按「**块内最好的句子**的 IDF 加权得分」排序（用 工单7 的 BM25 IDF 表），
    #      只取前 4 块 —— 注意不是按「整块总覆盖」排：实测 Q01 的董事长致辞块（p5-x0）虽然
    #      整页词多，但真正的答案句在 **p5-x1**（「零售进一步巩固收入贡献的支柱地位」），
    #      按总覆盖排会把 p5-x1 挤出候选；
    #   ② **每块最多贡献 3 句**（长块不能靠句子多寡霸占草稿）；
    #   ③ **句子级门槛**：命中权重实词数 ≥ ``min_word_hits``、权重和 ≥ ``min_sentence_mass``、
    #      且长度 ≤ ``max_sentence_chars`` —— 实测「资产重组承诺」表格的超长单元格
    #      只靠「平安 / 结构 / 风险」等高频词凑分，必须被这道门槛挡住。
    candidates: list[tuple[float, int, float, list[tuple[float, int, str]]]] = []
    for rank, chunk in enumerate(chunks[:8]):
        modality = str(chunk.get("type") or "text")
        raw_text = str(chunk.get("content") or "")
        # PDF 的软换行/多余空白先消掉（见 _SENT_SPLIT 注释）；
        # 但**表格块不能消换行** —— 它是 Markdown 行结构，消掉后整张表变成一行 984 字的长文本，
        # 所有句子门槛全部落空（实测 Q07 草稿直接为空），故表格块保留换行并按行切。
        if modality == "table":
            normalized = raw_text
            pieces = [seg for seg in normalized.split("\n") if seg.strip()]
        else:
            normalized = _SOFT_BREAK.sub("", raw_text)
            pieces = _SENT_SPLIT.split(normalized)
        local: list[tuple[float, int, str]] = []
        for raw_sentence in pieces:
            sentence = raw_sentence.strip(" |")
            if len(sentence) < 6 or len(sentence) > int(max_sentence_chars):
                continue
            if _is_noise_line(sentence):        # 来源行/表标题/页码等元信息不是答案内容
                continue
            matched = [weight for term, weight in zip(terms, weights) if term in sentence]
            field_hits = sum(1 for term in field_terms if term in sentence)
            mass = sum(matched) + float(field_weight) * field_hits
            if (len(matched) + field_hits) < int(min_word_hits) or mass < float(min_sentence_mass):
                continue
            has_number = bool(_ATOM_RE.search(sentence))
            local.append((mass * 2.0 + (1.0 if has_number else 0.0)
                          + (0.6 if modality == "table" else 0.0) - rank * 0.25, rank, sentence[:220]))
        if not local:
            continue
        local.sort(key=lambda item: -item[0])
        best = local[0][0]
        candidates.append((best - rank * 0.35, rank, best, local[:3]))
    candidates.sort(key=lambda item: -item[0])
    for _score, _rank, _best, local in candidates[:4]:
        scored.extend(local)
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked: list[str] = []
    used_pages: dict[Any, int] = {}
    for _score, rank, sentence in scored:
        if sentence in picked:
            continue
        page = int(chunks[rank].get("page") or 0) if rank < len(chunks) else 0
        if used_pages.get(page, 0) >= 3:            # 同页最多 3 句，保证草稿覆盖多页证据
            continue
        used_pages[page] = used_pages.get(page, 0) + 1
        picked.append(sentence)
        if len(picked) >= int(max_sentences) or sum(len(s) for s in picked) >= int(max_chars):
            break
    return "；".join(picked)


def generate_grounded(question: str, chunks: Sequence[Mapping[str, Any]], *, llm: Any,
                      language: str = "zh", logger: Any = None, analysis: Any = None,
                      max_tokens: int = 380) -> GroundedResult:
    """落地式生成主入口：草稿 → LLM 润色 → 原子护栏 → 兜底。"""
    import time

    from .llm import render_evidence

    started = time.perf_counter()
    draft = build_draft(question, chunks, analysis=analysis)
    atoms = numeric_atoms(draft) or numeric_atoms("".join(str(c.get("content") or "") for c in chunks[:2]))
    if not draft:
        draft = _fallback_excerpt(chunks)
        return GroundedResult(text=draft, method="extractive_draft", atoms=[], elapsed_ms=0.0,
                              error="证据不足以形成草稿（直接返回证据原文）")
    if llm is None or not getattr(llm, "available", lambda: False)():
        return GroundedResult(text=draft, method="extractive_draft", atoms=atoms,
                              backend="rule", model="rule-based",
                              elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                              error="Ollama 不可用 ⇒ 直接返回确定性草稿")
    prompt = (GROUNDED_PROMPT
              .replace("{evidence}", render_evidence(chunks, max_chars=420))
              .replace("{draft}", draft)
              .replace("{question}", str(question or "")))
    if language == "en":
        prompt = prompt.replace("中文提问用中文回答，英文提问用英文回答", "请用英文回答（English）")
    result = llm._chat(prompt, temperature=0.1, max_tokens=int(max_tokens))
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    if not result.ok or not str(result.text or "").strip():
        return GroundedResult(text=draft, method="extractive_draft", atoms=atoms,
                              elapsed_ms=elapsed, error=result.error or "模型返回空")
    candidate = str(result.text).strip()
    ok, missing = atom_preserved(candidate, atoms, keep_ratio=1.0)
    if ok:
        return GroundedResult(text=candidate, method="llm_polished", backend="ollama",
                              model=result.model, atoms=atoms, llm_candidate=candidate,
                              elapsed_ms=elapsed)
    if logger is not None:
        logger.log_event("graphrag.atom_guard_rejected", level="WARNING", missing=missing[:8],
                         atoms=len(atoms), action="改用确定性草稿（模型改写了数值）")
    return GroundedResult(text=draft, method="draft_fallback", backend="ollama", model=result.model,
                          atoms=atoms, missing_atoms=missing, llm_candidate=candidate,
                          elapsed_ms=elapsed, error=f"原子保全未通过，缺失 {len(missing)} 个数值")


def _is_noise_line(sentence: str) -> bool:
    """是否是「元信息行」而非答案内容（来源行 / 表标题 / 页码 / Markdown 分隔行）。

    依据（实测）：表格块每行都带 ``[来源] …第 81 页``，这些行含公司名与页码，
    在实词打分里能拿到不低的权重，会把草稿塞满无信息量的元信息（实测 Q07 草稿前 3 句全是来源行）。
    """
    text = str(sentence or "").strip()
    if not text:
        return True
    if text.startswith(("[来源]", "[表标题]", "[Markdown]", "|---", "| ---")):
        return True
    if re.fullmatch(r"[\|\-\s:]+", text):
        return True
    if re.search(r"第\s*\d+\s*页\s*$", text) and len(text) > 24:
        return True
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9（）()《》\-—\s]{0,12}年度报告\s*\d*", text):
        return True
    return False


def _term_weights(terms: Sequence[str], *, fallback: float = 1.0) -> list[float]:
    """问句实词的 **IDF 权重**（复用 工单7 的 BM25 IDF 表；取不到就退回等权）。

    为什么要加权（实测）：qwen/年报语料里「平安」「银行」「业务」这类高频词几乎每块都有，
    若等权计数，**只因为含「平安」二字**的无关表格块会与真正的答案句同分
    （实测 Q01：p81「资产重组承诺」表靠命中「平安」挤进草稿，把 90.75%/269,005.23万元
    这些与问题无关的数值原子带进护栏，正确答案反而被挤掉）。
    用 IDF 加权后，稀有实词（关键因素 / 逾越者联盟 / 拨备覆盖率）权重高，
    高频词权重低 ⇒ 打分能反映「这块到底讲了什么」。
    """
    idf: dict[str, float] = {}
    try:
        from app.core.retriever import load_bundle

        bundle = load_bundle(logger=None)
        bm25 = getattr(getattr(bundle, "bm25", None), "bm25", None)
        raw_idf = getattr(bm25, "idf", None) or {}
        if raw_idf:
            values = [float(v) for v in raw_idf.values() if float(v) > 0]
            mean = (sum(values) / len(values)) if values else 1.0
            idf = {str(k): float(v) / mean for k, v in raw_idf.items() if float(v) > 0}
    except Exception:  # noqa: BLE001 —— IDF 不可用不得阻断生成（退回等权）
        idf = {}
    weights: list[float] = []
    for term in terms:
        weight = idf.get(str(term))
        if weight is None:
            # 词表外（如未登录的专名/英文缩写）按「稀有」处理：给偏高的固定权重（但别太夸张）
            weight = 1.2 if len(str(term)) >= 3 else 0.5
        weights.append(max(min(float(weight), 3.0), 0.2))
    return weights or [fallback] * len(terms)


def _fallback_excerpt(chunks: Sequence[Mapping[str, Any]], *, limit: int = 320) -> str:
    """无草稿时的兜底：返回首个证据块的正文片段（不编造）。"""
    for chunk in chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("content") or "")).strip()
        if text:
            return text[:limit]
    return "不清楚"


def _question_terms(question: str, *, limit: int = 24, min_terms: int = 6) -> list[str]:
    """问句实词（用于草稿句子打分）：**jieba 分词** + 标点切分兜底。

    ⚠️ 两条实测踩坑（都必须记住）：
        1. 中文**没有词边界**：用标点切分得到的片段长达 16~20 字（「提到该行盈利增长的关键因素有哪些」），
           长度过滤会把**所有**片段滤掉 ⇒ 实词表为空 ⇒ 草稿为空 ⇒ 每题退回「证据原文」兜底；
        2. 但 2 字词（平安/银行/增长/业务…）在每份年报里到处都是：一旦放开到 2 字，
           「资产重组承诺」表格里的长句就靠命中「平安」「业务」挤进草稿，把答案带偏。
    折中策略：**先按 ≥3 字筛（中文）**，若实词不足 ``min_terms`` 个，再放宽到 ≥2 字补充。
    这样既不会空表，也不会被高频 2 字词主导。
    """
    stop = {"哪些", "哪个", "什么", "多少", "怎么", "如何", "是否", "以及", "分别", "包括",
            "请问", "说明", "分析", "结合", "关于", "对于", "年度报告", "年报", "报告", "公司",
            "集团", "银行", "根据", "以上", "如下", "该行", "本行", "我们", "他们", "什么",
            "情况", "方面", "内容", "问题", "主要", "进行", "通过", "相关", "表示", "上述",
            "what", "were", "which", "with", "that", "this", "from", "according", "annual",
            "report", "ratio", "bank", "group", "the", "and", "for"}
    terms: list[str] = []

    def _push(token: str, min_len: int) -> None:
        token = str(token or "").strip()
        if not (min_len <= len(token) <= 8) or token in terms or token.lower() in stop:
            return
        if re.fullmatch(r"[\d.,%]+", token):
            return
        terms.append(token)

    words: list[str] = []
    try:
        import jieba  # 已装（工单7 的 BM25 同源依赖）；不可用时走下面的标点切分兜底

        words = [str(w) for w in jieba.lcut(str(question or ""))]
    except Exception:  # noqa: BLE001 —— 分词不可用不得阻断问答
        for piece in re.split(r"[\s，,。、；;：:！!？?（）()【】\[\]]+", str(question or "")):
            if piece.strip():
                words.append(piece.strip())
    for word in words:
        is_ascii = bool(re.fullmatch(r"[A-Za-z]+", word))
        _push(word, 2 if is_ascii else 3)
    if len(terms) < int(min_terms):                        # 实词太少 ⇒ 放宽到 2 字补充
        for word in words:
            _push(word, 2)
    terms.sort(key=len, reverse=True)
    return terms[:limit]
