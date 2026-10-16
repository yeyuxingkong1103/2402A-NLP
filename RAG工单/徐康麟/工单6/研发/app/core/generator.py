# -*- coding: utf-8 -*-
"""工单4 生成器：可插拔后端（ollama / openai 兼容 / extractive）+ **原子保全护栏**。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化
工单6 增补：人工智能NLP-RAG-混合检索任务
    —— ``generate(..., use_cache=False)`` 新增形参：跳过生成级缓存读取，
    使「响应时间测试」量到真实 LLM 耗时而非缓存命中的约 1 ms。

设计要点（与任务书 §生成后端 / §★A / §★B 对齐）：

    1. **先确定性抽取、后 LLM 润色**：
       ``answer_composer.compose`` 先把「事实骨架」（数值 / 逐项枚举 / 结论句）从证据里抽出来；
    2. **LLM 只做润色**：把证据块（Top-K=3，见 ``RAG_RETRIEVAL__PROMPT_TOP_K``）与草稿交给 LLM，
       要求「逐项完整枚举、禁止用『等』概括省略、只用证据里的事实」；
    3. **原子保全校验（硬闸门）**：LLM 候选答案必须包含草稿的**全部原子**
       （带单位数值 + 枚举项 + 字段词）。缺任何一个 ⇒ 丢弃候选、回落到确定性草稿，
       并把缺失原子写进日志与响应（``llm_rejected``）——这是「禁止编造 / 禁止缩写枚举项」的落地方式；
    4. **extractive 永远可用**：无 LLM（探测失败 / 超时 / 后端不可用）时答案就是确定性草稿，
       行为与有 LLM 时**答案事实完全一致**，只是少一层润色。

生成后端事实（设计/环境事实.md §8）：本地 Ollama ``qwen2.5:3b`` 预热后首字 23.5~25.2 ms；
冷启动首字 3430.8 ms（必违约）⇒ 启动/问答前必须预热（``llm_client.warmup_generate``）。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from dataclasses import replace as dataclasses_replace

from .answer_composer import ComposedAnswer, atoms_preserved, compose, squash
from .cache import get_cache
from .citation import MODALITY_LABEL, MODALITY_SUB_LABEL
from .config import AppConfig, get_config
from .prompts_loader import load_prompt, render

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

UNKNOWN_TEXT = "不清楚"

#: 证据块标签（写进提示词，让 LLM 知道每块来自哪份文件哪一页、什么模态）
def _chunk_label(chunk: Mapping[str, Any]) -> str:
    modality = str(chunk.get("type") or chunk.get("modality") or "text")
    sub = chunk.get("modality_sub")
    label = MODALITY_LABEL.get(modality, modality)
    if modality == "image" and sub:
        label = MODALITY_SUB_LABEL.get(str(sub), label)
    return f"{chunk.get('file_name')} 第{int(chunk.get('page') or 0)}页（{label}）"


def build_evidence_block(chunks: Sequence[Mapping[str, Any]], *, max_chars: int = 2400,
                         max_chunks: int = 3) -> str:
    """把 Top-K 证据块格式化成提示词里的「证据」段（**只取前 ``max_chunks`` 条 = 生成口径 K=3**）。"""
    lines: list[str] = []
    used = 0
    for chunk in chunks[:max_chunks]:
        body = str(chunk.get("content") or chunk.get("text") or "")
        piece = f"【证据 {len(lines) + 1}】{_chunk_label(chunk)}\n{body}"
        if used + len(piece) > max_chars and lines:
            break
        lines.append(piece)
        used += len(piece)
    return "\n\n".join(lines)


def _lazy_logger(logger: Any, module: str = "generator") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class GenerationOutcome:
    """一次生成的结果（含后端、耗时、护栏结论与确定性草稿）。"""

    text: str
    backend: str
    model: str
    method: str                      # rule_composer | llm_grounded | llm_rejected
    first_token_ms: float
    total_ms: float
    llm_candidate: str | None = None
    llm_rejected_reason: str | None = None
    missing_atoms: list[str] = field(default_factory=list)
    composer: dict[str, Any] = field(default_factory=dict)
    language: str = "zh"
    #: T7：**英文路径禁止静默回落**——护栏两次未过时置 True，并在答案中显式提示（用户可见）
    language_degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prompt(question: str, chunks: Sequence[Mapping[str, Any]], draft: ComposedAnswer, *,
                 language: str = "zh", max_chunks: int = 3, logger: Any = None) -> str:
    """从 ``研发/app/prompts/qa_prompt.txt`` 渲染提示词（逐项枚举要求写在提示词文件里）。"""
    template = load_prompt("qa_prompt", logger=logger)
    evidence = build_evidence_block(chunks, max_chunks=max_chunks)
    draft_text = draft.text or "（确定性抽取未得到草稿，请完全依据证据作答）"
    return render(template, {
        "question": question,
        "evidence": evidence,
        "draft": draft_text,
        "language": "中文" if language != "en" else "英文",
    })


def generate(question: str, analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
             cfg: AppConfig | None = None, llm_client: Any = None, language: str = "zh",
             logger: Any = None, use_cache: bool = True) -> GenerationOutcome:
    """生成答案：确定性草稿 + （可选）LLM 润色 + 原子保全护栏。

    ``cfg.llm.backend`` 决定后端：

    * ``extractive``：只用确定性草稿（**无 LLM 依赖**，答案事实不变）；
    * ``ollama`` / ``openai``：LLM 润色；护栏不过即回落草稿；
    * ``auto``：由 ``llm_client`` 决定（引擎层已解析并留痕）。

    ``use_cache=False``（工单6 新增）：跳过**生成级缓存**的读取。工单4 的生成缓存原本
    不受调用方开关管辖，导致「响应时间测试」量到的是缓存命中的约 1 ms，而不是真实的 LLM 耗时。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger)
    started = time.perf_counter()
    with log.enter("generate", {"question_digest": question[:70], "chunks": len(chunks),
                                "backend_cfg": config.llm.backend, "language": language}) as span:
        # ① 确定性草稿（T7：层级题先做「证据驱动的问句补全」，避免翻译导致的父节点名不匹配而漏项）
        analysis_for_draft = augment_hierarchy_analysis(analysis, chunks, logger=log)
        draft = compose(analysis_for_draft, chunks, cfg=config, logger=log)
        # ①b **草稿一致性复核**（T7 新增，两步，都是「证据驱动」的，不含任何语料专有常量）：
        #    (1) 问句字段词覆盖：草稿必须落在问句问的那个字段上（如问「军用领域收入占比」时，
        #        草稿不能是「军种销售收入占比」）——不满足则用**最稀有字段词**过滤证据后重合成一次；
        #    (2) 数值/单位与证据一致性：草稿里的数值必须是证据原文的子串（防止「42.35% 被前导序号
        #        正则截成 35%」这类上游缺陷），缺单位时按证据里的单位声明补全（如「单位：万元」）。
        draft, repairs = verify_and_repair_draft(analysis_for_draft, draft, chunks, cfg=config, logger=log)
        # ② 生成级缓存（第三级）：同一「问句 + 证据块集合 + 语言 + 配置指纹」直接复用
        cache = get_cache(config, logger=log)
        chunk_ids = [str(c.get("chunk_id")) for c in chunks]
        gkey = cache.generation_key(question, chunk_ids=chunk_ids, language=language)
        if cache.enabled and use_cache:
            hit, cached = cache.generation.get(gkey)
            if hit and isinstance(cached, dict) and cached.get("text"):
                outcome = GenerationOutcome(
                    text=str(cached["text"]), backend=str(cached.get("backend", "cache")),
                    model=str(cached.get("model", "cache")), method="cache_hit",
                    first_token_ms=0.0, total_ms=round((time.perf_counter() - started) * 1000, 2),
                    llm_candidate=cached.get("llm_candidate"),
                    llm_rejected_reason=cached.get("llm_rejected_reason"),
                    missing_atoms=list(cached.get("missing_atoms") or []),
                    composer=draft.to_dict(), language=language,
                    language_degraded=bool(cached.get("language_degraded")))
                log.log_event("generation.cache_hit", key=gkey, chars=len(outcome.text))
                span.set_output({"method": "cache_hit", "chars": len(outcome.text), "cache": "hit"})
                return outcome
        backend = "extractive"
        model = "rule-based"
        method = "rule_composer"
        first_ms = 0.0
        candidate: str | None = None
        rejected: str | None = None
        missing: list[str] = []
        language_degraded = False
        text = draft.text

        if llm_client is not None and str(config.llm.backend).lower() != "extractive":
            backend = str(getattr(getattr(llm_client, "backend", None), "name", "llm"))
            model = str(getattr(getattr(llm_client, "backend", None), "model", ""))
            # ③ 最多两次尝试：第一次常规；英文路径或护栏拒绝时按失败原因重试一次
            for attempt in range(1, 3):
                try:
                    if language == "en":
                        prompt = _build_english_prompt(question, draft, attempt=attempt,
                                                       missing=missing, previous=candidate, logger=log)
                        template_name = f"qa_prompt_en#{attempt}"
                    else:
                        prompt = build_prompt(question, chunks, draft, language=language,
                                              max_chunks=int(getattr(config.retrieval, "prompt_top_k", 3)),
                                              logger=log)
                        template_name = f"qa_prompt#{attempt}"
                    log.log_event("generation.prompt", template=template_name, chars=len(prompt),
                                  attempt=attempt,
                                  evidence_chunks=min(len(chunks), int(getattr(config.retrieval, "prompt_top_k", 3))))
                    result = llm_client.generate_full(prompt, max_tokens=_max_tokens(config),
                                                      temperature=float(config.llm.temperature))
                    if attempt == 1:
                        first_ms = float(result.first_token_ms)
                    candidate = _postprocess(result.text)
                    if language == "en":
                        # 英文路径护栏：数值按基本单位等价保留（55.2 million ≡ 5,520.00万元）+ 英文主体
                        ok, missing = numbers_preserved(candidate, draft.atoms)
                        if ok and latin_ratio(candidate) < 0.4:
                            ok, missing = False, ["<英文主体>"]
                    else:
                        # 中文路径护栏：数值/计数/枚举项/字段词 + 关键子句
                        guard_atoms = list(dict.fromkeys(list(draft.atoms) + list(draft.key_phrases)))
                        ok, missing = atoms_preserved(candidate, guard_atoms, logger=log)
                    if not candidate:
                        rejected = "empty_llm_output"
                    elif not ok:
                        rejected = "atoms_missing"
                    else:
                        text = candidate if not draft.text else _merge_or_replace(candidate, draft)
                        method = "llm_grounded"
                        rejected = None
                    log.log_event("generation.llm_guard", accepted=(rejected is None), reason=rejected,
                                  attempt=attempt, missing_atoms=missing[:8], atoms=len(draft.atoms),
                                  candidate_chars=len(candidate or ""))
                    if rejected is None or attempt == 2:
                        break
                except Exception as exc:  # noqa: BLE001 —— LLM 失败必须显式降级到确定性草稿
                    rejected = f"{type(exc).__name__}: {exc}"
                    log.log_event("generation.llm_failed", level="WARNING", attempt=attempt, error=rejected,
                                  action="回落到确定性抽取草稿（答案事实不变）")
                    break
            # ④ 英文路径**禁止静默回落**：护栏两次都没过 ⇒ 显式降级标记 + 用户可见提示
            if language == "en" and (rejected is not None or method != "llm_grounded"):
                language_degraded = True
                degraded_note = _degraded_notice(rejected)
                base = text or draft.text or UNKNOWN_TEXT
                text = f"{base}\n{degraded_note}" if base and base != UNKNOWN_TEXT else base
                log.log_event("generation.language_degraded", level="WARNING", reason=rejected,
                              action="英文答案未通过护栏：已置 language_degraded 并在答案中显式提示")
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        outcome = GenerationOutcome(
            text=text or UNKNOWN_TEXT, backend=backend, model=model,
            method=("cache_hit" if method == "cache_hit" else
                    ("llm_grounded" if method == "llm_grounded" else "rule_composer")),
            first_token_ms=round(first_ms, 2), total_ms=total_ms, llm_candidate=candidate,
            llm_rejected_reason=rejected, missing_atoms=missing, composer=draft.to_dict(),
            language=language, language_degraded=language_degraded,
        )
        if cache.enabled and outcome.text and outcome.text != UNKNOWN_TEXT:
            try:
                cache.generation.put(gkey, {"text": outcome.text, "backend": outcome.backend,
                                            "model": outcome.model, "llm_candidate": outcome.llm_candidate,
                                            "llm_rejected_reason": outcome.llm_rejected_reason,
                                            "missing_atoms": outcome.missing_atoms,
                                            "language_degraded": outcome.language_degraded},
                                     saved_ms=total_ms)
            except Exception as exc:  # noqa: BLE001 —— 缓存写入失败不影响答案
                log.log_event("generation.cache_put_failed", level="WARNING",
                              error=f"{type(exc).__name__}: {exc}")
        log.log_event("generation.done", method=outcome.method, backend=backend, model=model,
                      chars=len(outcome.text), first_token_ms=outcome.first_token_ms,
                      total_ms=total_ms, rejected=rejected, language_degraded=language_degraded)
        span.set_output({"method": outcome.method, "backend": backend, "chars": len(outcome.text),
                         "rejected": rejected, "total_ms": total_ms,
                         "language_degraded": language_degraded})
        return outcome


def augment_hierarchy_analysis(analysis: Any, chunks: Sequence[Mapping[str, Any]], *,
                               logger: Any = None) -> Any:
    """层级题（``hierarchy_enum``）的**证据驱动问句补全**。

    根因（T7 定位，T6 §8.1 缺口2）：英文题经翻译后父节点名可能与语料不一致
    （语料「大客户销售部」 vs 译文「关键客户销售部门」），合成器只按「问句里出现过的父节点」
    取层级，于是只答出「销售部 4 个部门」而漏掉「6 个销售处」。
    修复：把**证据里真实出现的**「X 由 N 个Y构成」的父节点名并入问句，使层级枚举不受译名影响；
    非层级题原样返回（不改动其它形态的判定）。
    """
    if str(getattr(analysis, "answer_shape", "")) != "hierarchy_enum":
        return analysis
    question = str(getattr(analysis, "question", ""))
    parents: list[str] = []
    for chunk in chunks:
        for match in _HIER_PARENT.finditer(str(chunk.get("content") or "")):
            name, unit = match.group(1), match.group(3)
            if not name or name in parents:
                continue
            # **单位词过滤**（T7 实测教训）：只补与问句问的单位一致的层级。
            # 例：问「销售部有几个部门 / 大客户销售部有几个销售处」时，只补 unit=部门/销售处 的层级；
            # 「总经理 由 7 个下级构成：上海分公司、…」的 unit=下级 不在问句里 ⇒ 不补，
            # 否则答案会把「上海分公司」带进来（golden 明令禁止）。
            if unit and unit not in question:
                continue
            parents.append(name)
    if not parents:
        return analysis
    try:
        augmented = dataclasses_replace(analysis,
                                        question=str(getattr(analysis, "question", "")) + " " + " ".join(parents))
        log = _lazy_logger(logger)
        log.log_event("generation.hierarchy_augmented", parents=parents[:8])
        return augmented
    except Exception as exc:  # noqa: BLE001 —— 补全失败必须退回原分析对象（不阻断）
        _lazy_logger(logger).log_event("generation.hierarchy_augment_failed", level="WARNING",
                                       error=f"{type(exc).__name__}: {exc}")
        return analysis


#: 与 answer_composer 的 R1 同型的父节点提取（父节点名 / 数量 / **单位词**，用于问句补全）
_HIER_PARENT = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,20})\s*由\s*(\d+)\s*个([\u4e00-\u9fff]{1,8})构成")

#: 证据里的「单位声明」：表头 `（万元）` 或 `单位：万元`
_UNIT_DECL = re.compile(r"(?:单位[:：]\s*|[（(\[])(万元|亿元|万股|元|股|%|倍)(?:[）)\]]|\s|$)")
#: 带单位的数值 / 裸数值
_NUM_WITH_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)(万元|亿元|万股|元|股|%|倍)")
_NUM_PLAIN = re.compile(r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)(?![\d.,])")


def verify_and_repair_draft(analysis: Any, draft: ComposedAnswer,
                            chunks: Sequence[Mapping[str, Any]], *, cfg: Any = None,
                            logger: Any = None) -> tuple[ComposedAnswer, list[dict[str, Any]]]:
    """草稿一致性复核与修复（T7 新增）。

    返回 ``(修复后的草稿, 修复记录)``。两步：

    **第一步 · 字段词覆盖**：问句含字段词时，草稿必须命中其中至少一个；否则说明合成器选错了
    子句/序列（实测：问「军用领域收入占主营业务收入比重」，草稿却抽到「军种销售收入占比」）。
    处置：按「证据中出现次数最少」的字段词过滤证据块后**重合成一次**（只做一次，避免循环）。

    **第二步 · 数值/单位对齐证据**：
      * 草稿数值若在证据里只以**更长数值的后缀**形式出现（如草稿 `35%` ↔ 证据 `42.35%`），
        按证据原文补全——这类截断来自上游文本清洗，属「可证伪」的确定性修复；
      * 草稿出现裸数值而证据块声明了单位（`单位：万元` / `（万元）`）时补上单位
        （实测：`15,000.00` → `15,000.00万元`）。
    每一步都写结构化日志，修复记录进 ``diagnostics.generation.repairs``，绝不静默改数。
    """
    log = _lazy_logger(logger)
    repairs: list[dict[str, Any]] = []
    if not draft.ok:
        return draft, repairs
    evidence = "\n".join(str(c.get("content") or c.get("text") or "") for c in chunks)
    # ---- 第一步：字段词覆盖 ----
    terms = [squash(t) for t in (getattr(analysis, "field_terms", []) or []) if squash(t)]
    if terms:
        body = squash(draft.text)
        missing = [t for t in terms if t not in body]
        # 取**最稀有的缺失字段词**（最能锁定口径的那个）做一次受约束重合成；
        # 实测用例：问「军用领域收入占主营业务收入比重」而草稿只提到「主营业务收入」时，
        # 草稿其实抽错了主体（军种销售收入），必须按「军用领域」过滤证据后重合成。
        rarest = _rarest_term(missing, chunks) if missing else ""
        if rarest:
            filtered = [c for c in chunks
                        if rarest in squash(str(c.get("content") or c.get("text") or ""))]
            if filtered and len(filtered) < len(chunks):
                retry = compose(analysis, filtered, cfg=cfg, logger=log)
                if retry.ok and rarest in squash(retry.text):
                    repairs.append({"kind": "field_term_resynthesis", "term": rarest,
                                    "chunks_before": len(chunks), "chunks_after": len(filtered),
                                    "draft_before": draft.text[:80], "draft_after": retry.text[:80]})
                    log.log_event("generation.draft_resynthesized", term=rarest,
                                  chunks_before=len(chunks), chunks_after=len(filtered))
                    draft = retry
    # ---- 第二步：数值/单位对齐证据 ----
    evidence_flat = squash(evidence)
    fixed = draft.text
    for value, unit in _NUM_WITH_UNIT.findall(squash(draft.text)):
        token = f"{value}{unit}"
        # **完整数值**判定：token 必须是证据里某个数值的「整段」，不能只是更长数值的后缀
        # （实测：草稿 `35%` 实际是证据 `42.35%` 被上游清洗截断后的残片，直接子串匹配会误判为「已有证据」）
        if re.search(r"(?<![\d.])" + re.escape(token), evidence_flat):
            continue
        longer = [f"{v}{u}" for v, u in _NUM_WITH_UNIT.findall(evidence_flat)
                  if u == unit and v.endswith(value) and len(v) > len(value)]
        if longer:
            replacement = max(longer, key=len)
            fixed = fixed.replace(token, replacement)
            repairs.append({"kind": "number_truncation_fix", "from": token, "to": replacement})
            log.log_event("generation.number_repaired", frm=token, to=replacement)
    # 裸数值补单位：仅当证据块**声明了唯一单位**、该数值带千分位/小数（货币样式）、
    # 且确实出现在证据里 —— 避免把页码（479）或年份（2019）误补成「万元」。
    declared = {u for u in _UNIT_DECL.findall(evidence_flat)}
    if len(declared) == 1:
        unit = next(iter(declared))
        for value in _NUM_PLAIN.findall(squash(fixed)):
            if not re.search(r"[.,]", value):
                continue
            if value not in evidence_flat:
                continue
            if f"{value}{unit}" in evidence_flat or f"{value}{unit}" in squash(fixed):
                continue
            pattern = re.compile(re.escape(value) + r"(?![\d.,])(?![%倍股元万])")
            if pattern.search(squash(fixed)):
                fixed = pattern.sub(f"{value}{unit}", fixed, count=1)
                repairs.append({"kind": "unit_inherited", "value": value, "unit": unit})
                log.log_event("generation.unit_inherited", value=value, unit=unit)
    if fixed != draft.text:
        draft = dataclasses_replace(draft, text=fixed,
                                    notes={**draft.notes, "repairs": repairs})
    elif repairs:
        draft = dataclasses_replace(draft, notes={**draft.notes, "repairs": repairs})
    return draft, repairs


def _rarest_term(terms: Sequence[str], chunks: Sequence[Mapping[str, Any]]) -> str:
    """在候选字段词中挑「语料里出现次数最少」的那个（最能锁定主体/口径）。"""
    best, best_count = "", None
    for term in terms:
        count = sum(1 for c in chunks if term in squash(str(c.get("content") or c.get("text") or "")))
        if best_count is None or count < best_count:
            best, best_count = term, count
    return best


def _degraded_notice(reason: str | None) -> str:
    """语言降级提示（英文路径专用，**用户可见**，不静默）。"""
    detail = f"（{reason}）" if reason else ""
    return (f"[language_degraded{detail}: the English rendering did not pass the numeric/enumeration "
            f"guard, so the verbatim Chinese evidence answer is returned]")


def _max_tokens(config: AppConfig) -> int:
    """生成上限：枚举题需要更长输出，但不得超过配置（默认 512）。"""
    return int(getattr(config.llm, "max_tokens", 512) or 512)


def _postprocess(text: str) -> str:
    """LLM 输出清洗：去 ``` 代码块、去「答案：」前缀、压平空白。"""
    out = str(text or "").strip()
    if out.startswith("```"):
        out = out.strip("`")
        out = out.split("\n", 1)[-1] if "\n" in out else out
    for prefix in ("答案：", "答案:", "答：", "答:"):
        if out.startswith(prefix):
            out = out[len(prefix):]
    out = out.replace("\n", "").strip()
    return out


def _merge_or_replace(candidate: str, draft: ComposedAnswer) -> str:
    """护栏通过时的取舍：候选更长且包含草稿原子 ⇒ 采用候选；否则保留草稿。"""
    cand = squash(candidate)
    body = squash(draft.text)
    if not body:
        return candidate
    # 候选必须**包含草稿的全部原子**（已校验）且不是退化的一句话
    if len(cand) < max(12, int(len(body) * 0.5)):
        return draft.text
    return candidate


#: 英文路径第 2 次尝试用的**重译提示词**（内嵌常量）。
#: 说明：主提示词仍从 ``研发/app/prompts/qa_body_prompt.txt`` 读取（可评审、可迭代）；
#: 这条「失败重试」提示词只有失败路径才用到，T7 把它内嵌在本模块内，避免把改动扩散到
#: 提示词目录（T7 的改动范围只含本模块与 cache/retriever/query_understanding/answerability/language）。
_ENGLISH_RETRY_TEMPLATE = """You are translating a Chinese evidence-based answer into English.
Your previous translation was REJECTED because it dropped items or numbers.

Hard rules (each violation = rejection):
1. Translate the DRAFT sentence by sentence - every clause must appear in your output.
2. Keep EVERY number and unit exactly (4, 6, 5,520.00万元, 14.0%, -2.0% ...). Never drop or round one.
3. Enumerate EVERY item of every list. If the DRAFT lists 4 departments and 6 cities,
   your answer must list all 4 departments and all 6 cities, using their English names
   (e.g. 大客户销售部 -> Key Account Sales Department, 珠海 -> Zhuhai, 汽车 -> automotive, IC卡 -> IC card).
4. Translate Chinese organisation / department / city names into English.
5. Output the answer only, 1~4 sentences, no preamble, no citations, no markdown.

Original question (English): {question}

Chinese draft (authoritative facts, translate ALL of it):
{draft}

Your previous rejected translation (do NOT copy its omissions):
{previous}

Items/numbers that were missing last time: {missing}

Corrected English answer:
"""


def _build_english_prompt(question: str, draft: ComposedAnswer, *, attempt: int = 1,
                          missing: Sequence[str] | None = None, previous: str | None = None,
                          logger: Any = None) -> str:
    """英文路径：以确定性中文草稿为事实骨架，要求**逐项完整翻译**、数值原样保留。

    第 2 次尝试改用更严格的提示词，并显式告知上次缺失的数值/项目（见 ``_ENGLISH_RETRY_TEMPLATE``）。
    """
    if attempt <= 1:
        template = load_prompt("qa_body_prompt", logger=logger)
        return render(template, {"question": question, "draft": draft.text or "（无草稿）",
                                 "language": "English"})
    return render(_ENGLISH_RETRY_TEMPLATE, {
        "question": question,
        "draft": draft.text or "（无草稿）",
        "previous": (previous or "（空）")[:400],
        "missing": "、".join(list(missing or [])[:12]) or "（枚举项/数值不完整）",
    })


# ---------------------------------------------------------------------------
# 英文路径护栏：数值按**基本单位**等价比较（55.2 million ≡ 5,520.00 万元）
# ---------------------------------------------------------------------------
_EN_UNITS: tuple[tuple[str, float, str], ...] = (
    ("万元", 1e4, "money"), ("亿元", 1e8, "money"), ("元", 1.0, "money"), ("人民币", 1.0, "money"),
    ("万股", 1e4, "share"), ("亿股", 1e8, "share"), ("股", 1.0, "share"),
    ("million", 1e6, "money"), ("billion", 1e9, "money"), ("yuan", 1.0, "money"),
    ("万元", 1e4, "money"), ("%", 1.0, "percent"),
)
_NUM_TOKEN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(万元|亿元|元|万股|亿股|股|million|billion|yuan|%)?",
                        re.IGNORECASE)


def numeric_pairs(text: str) -> list[tuple[float, str]]:
    """抽取 ``(基本单位值, 量纲)``：中文/英文单位统一换算（万元/亿元/million/billion/%）。"""
    flat = re.sub(r"\s+", "", str(text or ""))
    out: list[tuple[float, str]] = []
    for m in _NUM_TOKEN.finditer(flat):
        raw, unit = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            value = float(raw)
        except ValueError:
            continue
        factor, kind = 1.0, "pure"
        for name, fac, k in _EN_UNITS:
            if unit == name.lower():
                factor, kind = fac, k
                break
        out.append((value * factor, kind))
    return out


def numbers_preserved(candidate: str, atoms: Sequence[str], *, tol: float = 1e-6) -> tuple[bool, list[str]]:
    """英文路径的数值护栏：草稿中的每个数值都要在候选里按基本单位找到等价写法。"""
    cand = numeric_pairs(candidate)
    missing: list[str] = []
    for atom in atoms:
        pairs = numeric_pairs(atom)
        if not pairs:
            continue
        for base, kind in pairs:
            ok = any(abs(base - cbase) <= tol * max(1.0, abs(base))
                     and (kind == ckind or "pure" in (kind, ckind))
                     for cbase, ckind in cand)
            if not ok:
                missing.append(str(atom))
                break
    return (not missing), missing


def latin_ratio(text: str) -> float:
    """拉丁字符占「字母+汉字」的比例（判断候选是否为英文主体）。"""
    latin = sum(1 for ch in str(text) if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in str(text) if "\u4e00" <= ch <= "\u9fff")
    total = latin + cjk
    return (latin / total) if total else 0.0
