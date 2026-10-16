# -*- coding: utf-8 -*-
"""工单4 问答引擎：从提问到「带引用答案」的完整链路（≤3 秒 / 可拒答 / 中英文 / 图像证据）。

工单编号：人工智能NLP-RAG 项目-金融问答系统部署
工单6 增补：人工智能NLP-RAG-混合检索任务
    —— ``answer(..., strategy=...)`` 接入检索策略层；策略指纹并入缓存键（``_strategy_fingerprint_of``）；
    策略诊断写入 ``diagnostics.strategy``；``use_cache=False`` 透传到检索与生成层，使三级缓存只写不读。

链路（设计/接口设计.md §3.3 / §4.2）：

    问句 → 会话指代消解 → 问句理解（语言/意图/答案形态/主体路由）
        → 混合检索（FAISS 向量 + BM25 → RRF → 轻量重排）Top-K=5
        → 可答性闸门（不达标 ⇒ 「不清楚」，HTTP 200 + refused=true）
        → 证据页补全（同页同类块）
        → 生成（确定性抽取草稿 + LLM 润色 + 原子保全护栏；后端可插拔）
        → 引用构造（1-based 物理页码 + 模态标注）
        → 缓存（问题级内存 + 答案级落盘）与结构化日志

性能策略（任务书「性能策略」）：
    * 问题级缓存（内存 LRU）+ 答案级缓存（``qa_runtime.sqlite3``，重启仍命中）；
    * 预热（``warmup()``：索引加载 + 分词器 + 嵌入 + 生成模型 ``keep_alive``）；
    * Top-K 检索（K=5 返回、K=3 进提示词）+ 轻量重排；
    * 延迟分列：``retrieve`` / ``generate`` / ``first_token`` / ``total``（毫秒，实测值，不估算）。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .answer_composer import squash
from .answerability import (GateInputs, evaluate_answerability, keyword_coverage_of,
                            mojibake_ratio, refusal_reason_text, refusal_text)
from .citation import Citation, build_citations, citation_marker, cited_pages as _cited_pages
from .config import AppConfig, get_config
from .conversation import ConversationStore, now_turn
from .embedder import warmup as embed_warmup
from .generator import GenerationOutcome, generate
from .language import english_variants
from .llm_client import LLMClient, resolve_backend, warmup_generate
from . import perf_hook
from .query_understanding import QueryAnalysis, analyze
from .retriever import (Hit, expand_with_page_context, load_bundle, out_of_vocabulary, retrieve,
                        warmup_entity_tagger)
from .text_utils import warmup_tokenizer

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

UNKNOWN_TEXT = "不清楚"

#: 问答链路版本号（写进缓存键）——检索/合成/护栏算法一变就升版，**旧缓存自动失效**，
#: 避免「改了算法却被旧缓存答案掩盖」（T5 实测踩坑：改完拒答闸门后仍命中旧的非拒答缓存）。
#: 工单6 升版到 ``t6-2026.10-v1``：检索策略层与缓存键（并入策略指纹）已变。
PIPELINE_VERSION = "t6-2026.10-v1"


def _lazy_logger(logger: Any, module: str = "qa_engine") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _strategy_fingerprint_of(strategy: Any) -> str:
    """检索策略指纹（工单6）：进缓存键，避免换策略后复用旧策略的答案。

    未传策略时返回 ``""`` —— 缓存键与 工单4 逐位一致，既有缓存继续有效。
    """
    if strategy is None:
        return ""
    if isinstance(strategy, Mapping):
        data = dict(strategy)
    elif hasattr(strategy, "to_dict"):
        data = dict(strategy.to_dict())
    else:  # pragma: no cover
        data = {"repr": repr(strategy)}
    keep = ("mode", "embedding_model", "vector_k", "fulltext_k", "top_k", "fusion",
            "vector_weight", "fulltext_weight", "rrf_k", "rerank_model", "score_threshold",
            "vector_min_similarity", "fulltext_fields", "fulltext_op", "enable_fuzzy",
            "use_image_channel")
    return json.dumps({k: data.get(k) for k in keep}, ensure_ascii=False, sort_keys=True,
                      default=str)


@dataclass(slots=True)
class Answer:
    """一次问答的结构化结果（对外 JSON 契约见 设计/接口设计.md §2.2 / §3.3）。"""

    question: str
    answer: str
    refused: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    backend: str = "extractive"
    model: str = "rule-based"
    generation_method: str = "rule_composer"
    cache: dict[str, Any] = field(default_factory=lambda: {"hit": False})
    trace_id: str = ""
    language: str = "zh"
    refusal_reason: str | None = None
    refusal_detail: str | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QAEngine:
    """问答引擎（进程内单例即可：索引与模型均有缓存，构造本身很快）。"""

    def __init__(self, cfg: AppConfig | None = None, *, logger: Any = None,
                 storage: Any = None, llm_client: Any = None, warmup: bool = False) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.storage = storage
        self._llm = llm_client
        self._llm_resolved = llm_client is not None
        # 工单17 F5：`_memory_cache` 原先是一个**裸 OrderedDict** —— 无锁、无 TTL。
        #   * 无锁：多线程并发 `move_to_end` / `popitem` 会撞坏内部链表（OrderedDict 非线程安全），
        #     表现为并发下偶发 KeyError / 500（压测实证见 优化/瓶颈排查与修复报告.md §2.3）；
        #   * 无 TTL：条目只在容量满时被 LRU 淘汰，长期运行下内存只涨不掉。
        # 现改为「(写入时刻, Answer) + RLock + LRU 上限 + TTL」，
        # 上限/TTL 由 PERF_ANSWER_CACHE_MAX / PERF_ANSWER_CACHE_TTL_S 配置；
        # PERF_ENABLED=0 时 limit 回退 cfg.retrieval.cache_size（512）、TTL 回退 0（永不过期）＝基座行为。
        self._memory_cache: "OrderedDict[str, tuple[float, Answer]]" = OrderedDict()
        self._cache_lock = threading.RLock()
        self._cache_evictions = 0
        self._cache_expired = 0
        self.conversations = ConversationStore(max_turns=int(self.cfg.answer.history_turns),
                                             storage=storage, logger=self.log)
        # 工单7：把「语料已知主体名」注入会话层，供指代消解使用。
        # 动机（实测缺陷，测试官 TC-M-A2/B3）：本工单语料的问句几乎都用**简称**
        # （「平安银行」「招商银行」），而 conversation 的形态正则原先只认「…股份有限公司」，
        # ⇒ 上文主体抽不到 ⇒ 「那它的净利润呢？」「该行…」完全不消解，落到别家公司。
        # 这里在构造期就把 ``bundle.corpus_issuers`` 交过去（只读引用，不复制数据）。
        self._inject_known_subjects()
        self._bundle = None
        self._stats = {"asked": 0, "refused": 0, "cache_hit": 0}
        if warmup:
            self.warmup()

    # ------------------------------------------------------------------ 预热
    def _inject_known_subjects(self) -> None:
        """把索引里的语料主体名灌进会话层（指代消解用）。索引不可用时静默跳过。"""
        try:
            known = getattr(self.bundle, "corpus_issuers", None)
            if known:
                self.conversations.set_known_subjects(known)
        except Exception as exc:  # noqa: BLE001 —— 注入失败不影响问答（消解退回形态匹配）
            self.log.log_event("qa.known_subjects_inject_failed", level="WARNING",
                               error=f"{type(exc).__name__}: {exc}")
    def warmup(self, *, with_generate: bool = True) -> dict[str, Any]:
        """预热链路（索引/分词/嵌入/生成模型），并把实测耗时写进日志。"""
        with self.log.enter("warmup", {"with_generate": with_generate}) as span:
            started = time.perf_counter()
            out: dict[str, Any] = {}
            self._bundle = load_bundle(self.cfg, logger=self.log)
            out["index_chunks"] = self._bundle.size
            # 工单7：索引就绪后把语料主体名灌进会话层（构造期 bundle 可能尚未加载）
            self._inject_known_subjects()
            out["tokenizer"] = warmup_tokenizer(logger=self.log)
            out["entity_tagger"] = warmup_entity_tagger(logger=self.log)
            out["embedding"] = embed_warmup(self.cfg, logger=self.log)
            # 工单6：预热多字段倒排索引（全文检索通道）。
            # 为什么必须预热：首次构建实测约 4 s（2676 块 × 4 字段分词），
            # 若不预热，用户第一次点「全文检索」就会吃掉 4 s，**直接违反「响应 ≤3 s」**。
            try:
                from .retrieval_strategy import warmup_strategy

                out["fulltext_index"] = warmup_strategy(cfg=self.cfg, logger=self.log)
            except Exception as exc:  # noqa: BLE001 —— 预热失败不得阻断启动，但必须记录
                self.log.log_event("qa.strategy_warmup_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
                out["fulltext_index"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if with_generate and self.llm is not None:
                out["generate"] = warmup_generate(self.cfg, logger=self.log)
            else:
                out["generate"] = {"ok": False, "reason": "无可用生成后端（extractive 路径无需预热）"}
            out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.log.log_event("qa.warmup", **{k: v for k, v in out.items()})
            span.set_output(out)
            return out

    # ------------------------------------------------------------------ 后端
    @property
    def llm(self) -> Any:
        """惰性解析生成后端（探测 ≤0.5 s、不重试；失败即返回 ``None`` → extractive 路径）。"""
        if self._llm_resolved:
            return self._llm
        self._llm_resolved = True
        try:
            backend = resolve_backend(self.cfg, logger=self.log)
            if backend.name == "extractive":
                self.log.log_event("qa.backend", backend="extractive",
                                   reason="无可用 LLM，走确定性抽取合成器")
                self._llm = None
            else:
                self._llm = LLMClient(cfg=self.cfg, backend=backend, logger=self.log)
        except Exception as exc:  # noqa: BLE001 —— 后端探测失败必须降级，不得阻断问答
            self.log.log_event("qa.backend_failed", level="WARNING",
                               error=f"{type(exc).__name__}: {exc}",
                               action="降级为 extractive（确定性合成器）")
            self._llm = None
        return self._llm

    @property
    def bundle(self) -> Any:
        if self._bundle is None:
            self._bundle = load_bundle(self.cfg, logger=self.log)
        return self._bundle

    # ------------------------------------------------------------------ 主流程
    def answer(self, question: str, *, lang: str = "auto", top_k: int | None = None,
               session_id: str | None = None, return_evidence: bool = False,
               use_cache: bool = True, strategy: Any = None) -> Answer:
        """端到端问答：返回带引用的答案；证据不足时 ``refused=True`` 且答案为「不清楚」。

        **工单6 新增**：``strategy`` —— 传入 ``retrieval_strategy.StrategyConfig``（或等价 dict）
        即可切换「向量检索 / 全文检索 / 混合检索」；不传则保持 工单4/5 既有混合链路，
        行为与基线逐位一致。策略指纹会进缓存键，避免换策略后命中旧策略的答案缓存。
        """
        trace_id = uuid.uuid4().hex[:12]
        log = self.log.bind(trace_id=trace_id)
        started = time.perf_counter()
        strategy_fp = _strategy_fingerprint_of(strategy)
        with log.enter("answer", {"question": question[:100], "lang": lang, "top_k": top_k,
                                  "session_id": session_id, "use_cache": use_cache,
                                  "strategy": strategy_fp[:60] or None}) as span:
            self._stats["asked"] += 1
            # 0) 会话指代消解（不改写事实，只补主体）
            resolved_question, anaphora_used = self.conversations.resolve(question, session_id)
            # 1) 问句理解（语言/意图/答案形态/主体路由）
            analysis = analyze(resolved_question, requested_lang=lang,
                               corpus_issuers=self.bundle.corpus_issuers, logger=log)
            language = analysis.language
            # 1b) 英文路径：先把问句译成中文检索式，再用中文问句理解（检索与判据都用中文）
            english_note: dict[str, Any] = {}
            retrieval_question = resolved_question
            if language == "en":
                variants = english_variants(resolved_question, llm_client=self.llm, cfg=self.cfg, logger=log)
                english_note = {"zh_query": variants.get("zh_query"), "backend": variants.get("backend"),
                                "error": variants.get("error")}
                if variants.get("zh_query"):
                    # 检索式换成中文译文（BM25 是中文分词器；向量路 bge-m3 中英均可），
                    # 作答语言仍是英文（analysis.language 保留 en，交给 generator 走英文提示词）。
                    retrieval_question = str(variants["zh_query"])
                    analysis = analyze(retrieval_question, requested_lang="zh",
                                       corpus_issuers=self.bundle.corpus_issuers, logger=log)
                    analysis.language = "en"   # 作答语言仍是英文
            # 2) 缓存（问题级内存 + 答案级落盘）
            key = self._cache_key(resolved_question, language, top_k, strategy_fp=strategy_fp)
            cached = self._cache_get(key) if use_cache else None
            if cached is not None:
                self._stats["cache_hit"] += 1
                cached = dict(cached)
                cached["cache"] = {"hit": True, "key": key, "layer": cached.get("cache", {}).get("layer", "memory")}
                answer = Answer(**cached)
                answer.trace_id = trace_id
                answer.latency_ms = {"retrieve": 0.0, "generate": 0.0, "first_token": 0.0,
                                     "total": round((time.perf_counter() - started) * 1000, 2)}
                log.log_event("qa.cache_hit", key=key, answer_chars=len(answer.answer),
                              total_ms=answer.latency_ms["total"])
                self._finish(answer, question, session_id, log, anaphora_used, english_note, from_cache=True)
                span.set_output({"refused": answer.refused, "chars": len(answer.answer),
                                 "cache_hit": True, "total_ms": answer.latency_ms["total"]})
                return answer
            # 3) 检索（默认：RRF 融合 + 轻量重排；工单6：可按策略走 向量/全文/混合）
            retrieve_started = time.perf_counter()
            strategy_diag: dict[str, Any] = {}
            hits: list[Hit] = retrieve(retrieval_question, top_k=top_k, analysis=analysis,
                                       cfg=self.cfg, logger=log, strategy=strategy,
                                       strategy_diag=(strategy_diag if strategy is not None else None),
                                       use_cache=use_cache)
            retrieve_ms = round((time.perf_counter() - retrieve_started) * 1000, 2)
            # 4) 可答性闸门
            coverage = keyword_coverage_of(analysis.keywords,
                                           [str(h.get("content") or "")[:1200] for h in hits[:3]])
            gate_inputs = GateInputs(
                question=resolved_question, keywords=list(analysis.keywords),
                top_score=float(hits[0].score) if hits else 0.0,
                max_vector_similarity=max([float(h.get("vector_score") or 0.0) for h in hits] or [0.0]),
                max_bm25_score=max([float(h.get("bm25_score") or 0.0) for h in hits] or [0.0]),
                coverage=coverage, corpus_hint=getattr(analysis, "corpus_hint", None),
                top_corpus=(self._corpus_of(hits[0]) if hits else None),
                corpus_hint_hit=self._corpus_hint_hit(analysis, hits),
                has_support_sentence=bool(hits),
                # 词表外实体闸门：只对中文路径启用（英文路径的实词本来就不在中文词表里）
                oov_keywords=(out_of_vocabulary(self.bundle, analysis.keywords,
                                                [str(h.get("content") or "")[:1500] for h in hits[:3]])
                              if language != "en" else []),
                # 工单7 新增：乱码闸门（问句过半汉字不在语料词表内 ⇒ 无法判断意图，拒答）
                mojibake_ratio=(mojibake_ratio(resolved_question, self.bundle.bm25)[0]
                                if language != "en" else 0.0),
            )
            gate = evaluate_answerability(gate_inputs, self.cfg, logger=log)
            diagnostics = {"gate": gate.to_dict(), "retrieve_ms": retrieve_ms,
                           "hits": [self._hit_digest(h) for h in hits],
                           "coverage": round(coverage, 4), "english": english_note,
                           # 工单6：检索策略全明细（模式/权重/融合/重排/各阶段耗时/说明）
                           "strategy": (strategy_diag or None) if strategy is not None else None}
            if not gate.answerable:
                total_ms = round((time.perf_counter() - started) * 1000, 2)
                self._stats["refused"] += 1
                answer = Answer(
                    question=question, answer=refusal_text(self.cfg), refused=True,
                    citations=[], cited_pages=[], evidence=[], chunks=[self._hit_payload(h) for h in hits],
                    latency_ms={"retrieve": retrieve_ms, "generate": 0.0, "first_token": retrieve_ms,
                                "total": total_ms},
                    backend=("extractive" if self.llm is None else str(getattr(self.llm.backend, "name", "llm"))),
                    model=("rule-based" if self.llm is None else str(getattr(self.llm.backend, "model", ""))),
                    generation_method="refused", cache={"hit": False, "key": key},
                    trace_id=trace_id, language=language, refusal_reason=gate.reason,
                    refusal_detail=refusal_reason_text(gate.reason),
                    analysis=analysis.to_dict(), diagnostics=diagnostics, session_id=session_id,
                )
                log.log_event("answerability.refused", reason=gate.reason, coverage=round(coverage, 4),
                              total_ms=total_ms)
                self._cache_put(key, answer)
                self._finish(answer, question, session_id, log, anaphora_used, english_note)
                span.set_output({"refused": True, "reason": gate.reason, "total_ms": total_ms})
                return answer
            # 5) 证据页补全（枚举完整性）
            prompt_top_k = int(getattr(self.cfg.retrieval, "prompt_top_k", 3))
            generation_hits = hits[:prompt_top_k]
            compose_hits = expand_with_page_context(
                self.bundle, generation_hits,
                max_extra=int(getattr(self.cfg.retrieval, "page_context_extra", 6)), logger=log)
            # 6) 生成（确定性草稿 + LLM 润色 + 原子保全护栏）
            outcome: GenerationOutcome = generate(
                resolved_question, analysis, compose_hits, cfg=self.cfg, llm_client=self.llm,
                language=language, logger=log, use_cache=use_cache)
            # 6b) **兜底形态复核**：若只落到「证据句选择」（无任何结构化证据支撑），
            #     且问句实词覆盖度低 / 向量相似度低 ⇒ 拒答而不是硬答一句无关句子。
            #     依据（T5 实测）：「今天北京的天气怎么样」只靠「天气」二字命中环保条款，
            #     句子选择器会把它当答案交出去（句面相关、语义无关）。
            if str(outcome.composer.get("method")) == "sentence_select":
                fallback_cov = float((outcome.composer.get("notes") or {}).get("coverage") or 0.0)
                # 工单7 修正：原实现拿**英文**余弦底线 min_vector_similarity_en(0.45) 去卡**中文**问句，
                # 尺度不对。中文路径改用按本工单语料标定的 min_vector_similarity（默认 0.60，
                # 实测正样本余弦 min 0.6626 / 语料外负样本 max 0.6113，见 测试/测试数据/阈值标定.json）。
                _sim_floor = float(
                    getattr(self.cfg.answer, "min_vector_similarity", 0.60)
                    if language != "en"
                    else getattr(self.cfg.answer, "min_vector_similarity_en", 0.45))
                if fallback_cov < 0.5 and float(gate_inputs.max_vector_similarity) < _sim_floor:
                    total_ms = round((time.perf_counter() - started) * 1000, 2)
                    self._stats["refused"] += 1
                    answer = Answer(
                        question=question, answer=refusal_text(self.cfg), refused=True,
                        citations=[], cited_pages=[], evidence=[],
                        chunks=[self._hit_payload(h) for h in hits],
                        latency_ms={"retrieve": retrieve_ms, "generate": round(float(outcome.total_ms), 2),
                                    "first_token": retrieve_ms, "total": total_ms},
                        backend=outcome.backend, model=outcome.model, generation_method="refused",
                        cache={"hit": False, "key": key}, trace_id=trace_id, language=language,
                        refusal_reason="low_similarity",
                        refusal_detail=refusal_reason_text("low_similarity"),
                        analysis=analysis.to_dict(),
                        diagnostics={**diagnostics, "generation": outcome.to_dict(),
                                     "fallback_coverage": round(fallback_cov, 4)},
                        session_id=session_id,
                    )
                    log.log_event("answerability.refused", reason="low_similarity",
                                  stage="post_composition", fallback_coverage=round(fallback_cov, 4),
                                  max_vector=round(float(gate_inputs.max_vector_similarity), 4))
                    self._cache_put(key, answer)
                    self._finish(answer, question, session_id, log, anaphora_used, english_note)
                    span.set_output({"refused": True, "reason": "low_similarity", "total_ms": total_ms})
                    return answer
            # 7) 引用（1-based 物理页码；只引用真正取用的块）
            used = list(outcome.composer.get("used_chunk_ids") or [])
            if not used and generation_hits:
                used = [generation_hits[0].chunk_id]
            citations: list[Citation] = build_citations(hits, used_chunk_ids=used,
                                                        max_citations=max(prompt_top_k, 3), logger=log)
            if not citations and hits:
                citations = build_citations(hits, used_chunk_ids=[hits[0].chunk_id], logger=log)
            body = outcome.text
            marker = citation_marker(citations)
            text = f"{body}\n{marker}" if marker else body
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            first_token_ms = round(retrieve_ms + float(outcome.first_token_ms), 2)
            answer = Answer(
                question=question, answer=text, refused=False,
                citations=[c.to_dict() for c in citations], cited_pages=_cited_pages(citations),
                evidence=[self._hit_payload(h, with_text=return_evidence) for h in generation_hits],
                chunks=[self._hit_payload(h) for h in hits],
                latency_ms={"retrieve": retrieve_ms,
                            "generate": round(float(outcome.total_ms), 2),
                            "first_token": first_token_ms, "total": total_ms},
                backend=outcome.backend, model=outcome.model, generation_method=outcome.method,
                cache={"hit": False, "key": key}, trace_id=trace_id, language=language,
                analysis=analysis.to_dict(),
                diagnostics={**diagnostics, "generation": outcome.to_dict(),
                             "citations": [c.to_dict() for c in citations]},
                session_id=session_id,
            )
            self._cache_put(key, answer)
            self._finish(answer, question, session_id, log, anaphora_used, english_note)
            log.log_event("qa.answer", refused=False, chars=len(text), cited_pages=answer.cited_pages,
                          method=outcome.method, backend=outcome.backend,
                          retrieve_ms=retrieve_ms, generate_ms=outcome.total_ms,
                          first_token_ms=first_token_ms, total_ms=total_ms)
            span.set_output({"refused": False, "chars": len(text), "pages": answer.cited_pages,
                             "total_ms": total_ms, "cache_hit": False})
            return answer

    # ------------------------------------------------------------------ 内部
    def _corpus_of(self, hit: Mapping[str, Any]) -> str | None:
        """证据块所属语料键。

        工单7 修正：必须用 ``bundle.corpus_of_file``（来自 manifest 的 ``corpus_keys``，
        与 chunk_id 前缀同源）。旧实现调 ``retriever.corpus_key_of``，那是以「文件名里首个数字段」
        作键 —— 本工单 9 份年报被它按年份并成 pdf2020/pdf2021/pdf2022 三个桶，
        于是「问句点名平安银行、Top-1 却来自中国平安年报」这类跨语料错答
        因两文件同属 pdf2020 而**看起来匹配**（主体闸门失效，实测 Q09 即此情形）。
        """
        from .retriever import corpus_key_of_file

        file_name = str(hit.get("file_name") or "")
        if not file_name:
            return None
        corpus_map = getattr(getattr(self, "bundle", None), "corpus_of_file", None)
        return corpus_key_of_file(file_name, corpus_map) or None

    def _corpus_hint_hit(self, analysis: QueryAnalysis, hits: Sequence[Mapping[str, Any]]) -> bool:
        """问句点名的发行人是否真的出现在 Top-K 证据里（跨文件串答的判定依据）。"""
        issuer = squash(getattr(analysis, "issuer_hint", "") or "")
        if not issuer:
            return True
        for hit in hits[:3]:
            if issuer in squash(str(hit.get("content") or "")):
                return True
        return False

    def _hit_digest(self, hit: Mapping[str, Any]) -> dict[str, Any]:
        return {"chunk_id": hit.get("chunk_id"), "page": hit.get("page"), "file": hit.get("file_name"),
                "type": hit.get("type"), "final_score": hit.get("final_score"),
                "vector_rank": hit.get("vector_rank"), "bm25_rank": hit.get("bm25_rank"),
                "bm25_score": hit.get("bm25_score"), "vector_score": hit.get("vector_score")}

    def _hit_payload(self, hit: Mapping[str, Any], *, with_text: bool = True) -> dict[str, Any]:
        payload = {
            "chunk_id": hit.get("chunk_id"), "page": hit.get("page"), "page_basis": "1-based-physical",
            "file_name": hit.get("file_name"), "type": hit.get("type"),
            "modality": hit.get("modality") or hit.get("type"),
            "modality_sub": hit.get("modality_sub"), "score": hit.get("score"),
            "final_score": hit.get("final_score"), "rrf_score": hit.get("rrf_score"),
            "vector_rank": hit.get("vector_rank"), "bm25_rank": hit.get("bm25_rank"),
            "bm25_score": hit.get("bm25_score"), "vector_score": hit.get("vector_score"),
        }
        if with_text:
            payload["text"] = str(hit.get("content") or hit.get("text") or "")
        else:
            payload["text"] = str(hit.get("content") or hit.get("text") or "")[:400]
        return payload

    def _cache_key(self, question: str, language: str, top_k: int | None,
                   strategy_fp: str = "") -> str:
        from ..storage.sqlite_manager import cache_key

        return cache_key(f"{PIPELINE_VERSION}|{question}", lang=language,
                         top_k=int(top_k or self.cfg.retrieval.top_k),
                         extra=(strategy_fp or None))

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        """两级缓存：先内存，再落盘（命中落盘时回填内存）。

        工单17 F5：内存层加锁 + TTL 过期检查（过期即当作未命中并删除）。
        """
        now = time.time()
        with self._cache_lock:
            item = self._memory_cache.get(key)
            if item is not None:
                stamp, cached = item
                ttl = perf_hook.answer_cache_ttl(0.0)
                if ttl > 0 and (now - float(stamp)) > ttl:
                    self._memory_cache.pop(key, None)
                    self._cache_expired += 1
                else:
                    self._memory_cache.move_to_end(key)
                    payload = cached.to_dict()
                    payload["cache"] = {"hit": True, "key": key, "layer": "memory"}
                    return payload
        if self.storage is not None:
            stored = self.storage.load_answer_cache(key)
            if stored:
                self.storage.bump_cache_hit(key)
                payload = dict(stored)
                payload["cache"] = {"hit": True, "key": key, "layer": "sqlite"}
                try:
                    rebuilt = Answer(**payload)
                except TypeError:  # 兼容旧结构：字段缺失时不进内存（仍可用）
                    return payload
                with self._cache_lock:
                    self._memory_cache[key] = (time.time(), rebuilt)
                    self._memory_cache.move_to_end(key)
                    self._trim_locked()
                return payload
        return None

    def _trim_locked(self) -> None:
        """按上限淘汰最旧条目（**调用方须已持锁**）。"""
        limit = perf_hook.answer_cache_limit(int(getattr(self.cfg.retrieval, "cache_size", 512)))
        while len(self._memory_cache) > max(1, limit):
            self._memory_cache.popitem(last=False)
            self._cache_evictions += 1

    @staticmethod
    def _slim(answer: Answer) -> Answer:
        """工单17 F5：缓存条目瘦身。

        实测动机：``chunks`` / ``evidence`` 里带的是**整块全文**（``_hit_payload`` 默认
        ``with_text=True``），单条 Answer 可达 30~60 KB；512 条就是 15~30 MB 的常驻内存，
        而缓存命中只需要「答案 + 引用 + 关键诊断」，不需要全文。
        这里把这两个字段里的 ``text`` 截断到 ``PERF_CACHE_CHUNK_CHARS``（默认 400 字符）。
        优化关闭（``PERF_ENABLED=0``）时阈值为 0 = 不截断，行为与基座逐位一致。
        """
        limit = perf_hook.cache_chunk_chars(0)
        if limit <= 0:
            return answer
        slim = Answer(**answer.to_dict())
        for bucket in (slim.chunks, slim.evidence):
            for item in bucket:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and len(text) > limit:
                        item["text"] = text[:limit]
        return slim

    def _cache_put(self, key: str, answer: Answer) -> None:
        with self._cache_lock:
            self._memory_cache[key] = (time.time(), self._slim(answer))
            self._memory_cache.move_to_end(key)
            self._trim_locked()
        if self.storage is not None:
            payload = answer.to_dict()
            payload["cache"] = {"hit": False, "key": key}
            try:
                self.storage.save_answer_cache(key, answer.question, payload)
            except Exception as exc:  # noqa: BLE001 —— 缓存落盘失败不影响问答
                self.log.log_event("qa.cache_persist_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")

    def _finish(self, answer: Answer, question: str, session_id: str | None, log: Any,
                anaphora_used: bool, english_note: Mapping[str, Any], *, from_cache: bool = False) -> None:
        """收尾：会话记录 + 历史落盘（两者失败都只 WARNING）。"""
        if session_id:
            self.conversations.add(session_id, now_turn(question, answer.answer, trace_id=answer.trace_id,
                                                        cited_pages=answer.cited_pages))
        if self.storage is not None:
            try:
                self.storage.record_qa(
                    trace_id=answer.trace_id, question=question, answer=answer.answer,
                    refused=answer.refused, session_id=session_id, refusal_reason=answer.refusal_reason,
                    citations=answer.citations, cited_pages=answer.cited_pages,
                    latency_ms=float(answer.latency_ms.get("total", 0.0)),
                    first_token_ms=float(answer.latency_ms.get("first_token", 0.0)),
                    backend=answer.backend, model=answer.model, cache_hit=bool(answer.cache.get("hit")),
                    language=answer.language, top_k=int(self.cfg.retrieval.top_k),
                )
            except Exception as exc:  # noqa: BLE001 —— 历史落盘失败不影响问答
                log.log_event("qa.history_failed", level="WARNING", error=f"{type(exc).__name__}: {exc}")
        log.log_event("qa.done", refused=answer.refused, language=answer.language,
                      anaphora_used=anaphora_used, cache_hit=bool(answer.cache.get("hit")),
                      from_cache=from_cache, english=dict(english_note),
                      cited_pages=answer.cited_pages)

    # ------------------------------------------------------------------ 诊断
    def retrieve_only(self, question: str, *, top_k: int = 10, modality: str | None = None) -> dict[str, Any]:
        """只检索不生成（``/retrieve`` 诊断口径；用于检索精度截图留痕）。"""
        started = time.perf_counter()
        log = self.log
        analysis = analyze(question, corpus_issuers=self.bundle.corpus_issuers, logger=log)
        hits = retrieve(question, top_k=top_k, modality=modality, analysis=analysis, cfg=self.cfg, logger=log)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        log.log_event("retrieval.only", question_digest=question[:60], hits=len(hits),
                      modality=modality, elapsed_ms=elapsed)
        return {"question": question, "hits": [self._hit_payload(h) for h in hits],
                "hit_count": len(hits), "modality": modality,
                "latency_ms": {"total": elapsed}}

    def health(self) -> dict[str, Any]:
        """健康检查（不伪装未就绪能力：每项都给 ``ok`` 与原因）。"""
        import sys

        log = self.log
        checks: dict[str, Any] = {}
        checks["interpreter"] = {"ok": True, "python": sys.version.split()[0], "executable": sys.executable}
        resolved_index_dir = str(getattr(self.cfg.paths, "index_dir", ""))
        try:
            bundle = self.bundle
            by_modality = bundle.manifest.get("chunk_by_modality") or {}
            resolved_index_dir = bundle.index_dir          # ← **已解析**路径（含 bge-m3_1024 子目录）
            checks["index"] = {"ok": True, "chunks": bundle.size, "dim": bundle.vector_index.dim,
                               "faiss": bundle.vector_index.faiss_index is not None,
                               "image_chunks": int(by_modality.get("image") or 0),
                               "path": bundle.index_dir, "bm25_vocab": bundle.bm25.vocab_size()}
        except Exception as exc:  # noqa: BLE001 —— 如实报告不可用
            checks["index"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        client = self.llm
        if client is None:
            checks["llm"] = {"ok": True, "backend": "extractive", "model": "rule-based",
                             "note": "无 LLM：走确定性抽取合成器（答案事实一致，仅少一层润色）"}
        else:
            checks["llm"] = {"ok": True, **client.health()}
        try:
            from .vector_store import describe_index

            # ⚠️ 必须传**已解析**的索引目录（bundle.index_dir），不能用 cfg.paths.index_dir：
            #    后者是父目录（研发/data/index），ids.json/vectors.npy 实际在 bge-m3_1024/ 下，
            #    传父目录会让 health 误报 degraded（captain 2026-10 抽检定位）。
            checks["index_meta"] = {"ok": True, "path": resolved_index_dir,
                                    **describe_index(resolved_index_dir, logger=log)}
        except Exception as exc:  # noqa: BLE001
            checks["index_meta"] = {"ok": False, "path": resolved_index_dir,
                                    "reason": f"{type(exc).__name__}: {exc}"}
        out = {"status": "ok" if all(c.get("ok") for c in checks.values()) else "degraded",
               "checks": checks, "stats": dict(self._stats),
               "config": {"top_k": self.cfg.retrieval.top_k,
                          "prompt_top_k": int(getattr(self.cfg.retrieval, "prompt_top_k", 3)),
                          "embed_model": self.cfg.llm.ollama_embed_model,
                          "gen_model": self.cfg.llm.ollama_gen_model,
                          "backend_cfg": self.cfg.llm.backend}}
        log.log_event("qa.health", status=out["status"], stats=out["stats"])
        return out

    def stats(self) -> dict[str, Any]:
        """引擎运行统计（工单17：补上缓存上限/TTL/淘汰/过期计数，供 /api/v1/perf）。"""
        with self._cache_lock:
            entries = len(self._memory_cache)
        return {**self._stats, "memory_cache": entries,
                "memory_cache_max": perf_hook.answer_cache_limit(
                    int(getattr(self.cfg.retrieval, "cache_size", 512))),
                "memory_cache_ttl_s": perf_hook.answer_cache_ttl(0.0),
                "memory_cache_evictions": self._cache_evictions,
                "memory_cache_expired": self._cache_expired,
                "storage": (self.storage.stats() if self.storage is not None else None)}
