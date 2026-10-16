# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# （本文件复用自 工单7/工单10 基座；工单17 只补头部编号，未改动其逻辑）
"""工单：人工智能NLP-RAG-Query 理解优化
工单6 增补：人工智能NLP-RAG-混合检索任务
    —— ``ask(..., strategy=...)`` 把检索策略透传给底层问答引擎；策略指纹并入对话级缓存键；
    ``MultiTurnAnswer.base`` 白名单新增 ``diagnostics``（供界面「三模式对比」与评测读取策略明细）。

多轮问答引擎（``MultiTurnEngine``）—— 工单5 对外的统一入口。

## 职责
把「**工单5 的多轮 Query 理解**」套在「**工单4 的单轮 RAG 内核**」外面：

    用户问句
        │
        ├─ DialogueStateStore   取本会话状态（焦点实体 / 上一轮槽位 / 最近 3 轮）
        │
        ├─ QueryRewriter        指代消解 + 省略补全 → **独立可检索问句**
        │
        ├─ QAEngine.answer()    工单4 内核：混合检索 → RRF → 可答性闸门 → 生成 → 引用
        │
        └─ 回写状态 + 返回 MultiTurnAnswer（含改写前后对照，便于界面展示与留痕）

## 与工单4 的关系（**不退化**保证）
* 单轮调用（无历史 / 无指代 / 无省略）时，``strategy == "none"`` 且改写后问句 == 原问句，
  链路与工单4 **逐字一致** —— 因此工单4 的 16 题单轮能力原样保留（实测 16/16）。
* 底层 ``QAEngine`` 收到的是**已消解**的问句，故不再需要它自己的 ``ConversationStore``
  （这里传 ``session_id=None``，避免两套会话状态互相干扰）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"


def _strategy_fingerprint(strategy: Any) -> str:
    """检索策略指纹（工单6）：进对话级缓存键，避免换策略后复用旧策略的答案。

    未传策略时返回 ``""`` —— 缓存键与 工单5 逐位一致，既有会话缓存继续有效。
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

#: 空输入 / 无内容时的友好提示（对外可见文案）
EMPTY_INPUT_HINT = (
    "请输入您想问的问题。例如：「武汉兴图新科电子股份有限公司的法定代表人是谁？」"
    "或者先问一家公司，再用「他」「这个公司」「那另一家公司呢？」来追问。"
)


def _lazy_logger(logger: Any, module: str = "multiturn_engine") -> Any:
    if logger is not None:
        return logger
    try:
        from .logging_conf import get_logger

        return get_logger(module)
    except Exception:  # noqa: BLE001
        return _NullLogger()


class _NullLogger:
    """日志不可用时的空实现（对话链路永不因日志失败）。"""

    def log_event(self, *_a: Any, **_kw: Any) -> None:
        return None

    def bind(self, **_kw: Any) -> "_NullLogger":
        return self

    class _Span:
        def set_output(self, *_a: Any, **_kw: Any) -> None:
            return None

        def __enter__(self) -> "_NullLogger._Span":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

    def enter(self, *_a: Any, **_kw: Any) -> "_NullLogger._Span":
        return _NullLogger._Span()


@dataclass(slots=True)
class MultiTurnAnswer:
    """一次多轮问答的结果（界面 / HTTP / 报告共用）。"""

    question: str                    # 用户原句
    rewritten_query: str             # 改写后送入检索的独立问句
    answer: str
    refused: bool = False
    strategy: str = "none"
    focus_entity: str | None = None
    turn_index: int = 1
    citations: list[dict[str, Any]] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    language: str = "zh"
    rewrite: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiTurnEngine:
    """多轮问答引擎（进程内单例即可）。"""

    def __init__(self, cfg: Any = None, *, logger: Any = None, storage: Any = None,
                 engine: Any = None, warmup: bool = False,
                 persist_dir: Path | str | None = None) -> None:
        from .core.config import get_config

        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        if engine is not None:
            self.engine = engine
        else:
            from .core.qa_engine import QAEngine

            self.engine = QAEngine(self.cfg, logger=self.log, storage=storage)
        #: 最近 3 轮（工单「速度优先：对话历史只保留最近 3 轮」）
        self.max_turns = 3
        from .core.dialogue_state import DialogueStateStore

        if persist_dir is None:
            try:
                persist_dir = Path(self.cfg.paths.repo_root) / "研发" / "data" / "runtime" / "sessions"
            except Exception:  # noqa: BLE001 —— 配置异常时不落盘，纯内存运行
                persist_dir = None
        self.states = DialogueStateStore(max_turns=self.max_turns, storage=storage,
                                         logger=self.log, persist_dir=persist_dir)
        #: 工单5 优化层：**对话级缓存**（LRU + TTL）。
        #: 键含 session_id —— 同一问句在不同会话里上下文/焦点可能不同，**不得跨会话复用**；
        #: 因此这里刻意**不落盘**（底层 QAEngine 已有答案级 SQLite 缓存负责跨进程命中），
        #: 只用进程内 LRU，避免多一层持久化带来的陈旧风险。
        from .core.dialogue_cache import DialogueCache

        self.cache = DialogueCache(logger=self.log, persist_path=None)
        self.rewriter = None            # 需等索引加载后才能拿到 corpus_issuers
        self._stats = {"asked": 0, "rewritten": 0, "refused": 0, "empty": 0, "cache_hit": 0}
        if warmup:
            self.warmup()

    # ------------------------------------------------------------------ 预热
    def _ensure_rewriter(self) -> Any:
        """惰性构造改写器（实体表来自索引内容自动统计，**不硬编码公司名**）。"""
        if self.rewriter is None:
            from .core.query_rewriter import QueryRewriter

            issuers = {}
            try:
                issuers = getattr(self.engine.bundle, "corpus_issuers", {}) or {}
            except Exception as exc:  # noqa: BLE001 —— 拿不到实体表也要能工作（退化为纯正则）
                self.log.log_event("multiturn.issuers_unavailable", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
            self.rewriter = QueryRewriter(corpus_issuers=issuers, cfg=self.cfg, logger=self.log)
        return self.rewriter

    def warmup(self, *, with_generate: bool = True) -> dict[str, Any]:
        """预热底层链路并构造改写器，返回实测耗时。"""
        started = time.perf_counter()
        out: dict[str, Any] = {}
        out["base"] = self.engine.warmup(with_generate=with_generate)
        rw = self._ensure_rewriter()
        out["entities"] = sum(len(v) for v in (rw.corpus_issuers or {}).values())
        out["corpora"] = sorted((rw.corpus_issuers or {}).keys())
        out["max_turns"] = self.max_turns
        out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.log.log_event("multiturn.warmup", entities=out["entities"],
                           corpora=out["corpora"], elapsed_ms=out["elapsed_ms"])
        return out

    # ------------------------------------------------------------------ 主流程
    def ask(self, question: str, *, session_id: str = "default", lang: str = "auto",
            use_cache: bool = True, strategy: Any = None) -> MultiTurnAnswer:
        """多轮问答：改写 → 检索 → 生成 → 状态回写。

        **工单6 新增**：``strategy`` —— 把检索策略（向量 / 全文 / 混合）透传给底层问答引擎；
        不传时行为与 工单5 基线一致。指代消解与查询改写发生在检索之前，与检索策略**正交**，
        因此换策略不影响多轮能力（回归测试据此设计）。
        """
        sid = session_id or "default"
        t0 = time.perf_counter()
        self._stats["asked"] += 1
        state = self.states.get(sid)
        rewriter = self._ensure_rewriter()
        rewrite = rewriter.rewrite(question, state)
        rewrite_ms = round((time.perf_counter() - t0) * 1000, 2)

        # R0：空 / 无内容 —— 直接友好提示，不进入检索（也绝不崩溃）
        if rewrite.strategy == "empty_input":
            self._stats["empty"] += 1
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            out = MultiTurnAnswer(
                question=question if isinstance(question, str) else "",
                rewritten_query="", answer=EMPTY_INPUT_HINT, refused=False,
                strategy="empty_input", focus_entity=state.focus_entity,
                turn_index=int(state.total_turns) + 1, citations=[], cited_pages=[],
                latency_ms={"rewrite": rewrite_ms, "retrieve": 0.0, "generate": 0.0,
                            "total": total_ms},
                language="zh", rewrite=rewrite.to_dict(), base={"skipped": "empty_input"},
                session_id=sid,
            )
            self._record(sid, state, out)
            return out

        # 工单5 优化层：对话级缓存查表（键含 session_id；``use_cache=False`` 时整层关闭）
        # 工单6：策略指纹并入对话级缓存键，避免换策略后命中旧策略的答案
        strategy_fp = _strategy_fingerprint(strategy)
        cache_key: str | None = None
        if use_cache:
            cache_key = self.cache.make_key(rewrite.rewritten, lang, sid, strategy_fp=strategy_fp)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._from_cache(cached, question, sid, state, rewrite, rewrite_ms, t0)

        base = self.engine.answer(rewrite.rewritten, lang=lang, use_cache=use_cache,
                                  strategy=strategy)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        if rewrite.changed:
            self._stats["rewritten"] += 1
        if getattr(base, "refused", False):
            self._stats["refused"] += 1

        lat = dict(getattr(base, "latency_ms", {}) or {})
        lat["rewrite"] = rewrite_ms
        # total 以**本引擎**的墙钟为准（含改写），不用底层 total（不含改写）
        lat["total"] = total_ms
        lat.setdefault("retrieve", 0.0)
        lat.setdefault("generate", 0.0)

        out = MultiTurnAnswer(
            question=question if isinstance(question, str) else "",
            rewritten_query=rewrite.rewritten,
            answer=str(getattr(base, "answer", "") or ""),
            refused=bool(getattr(base, "refused", False)),
            strategy=rewrite.strategy,
            focus_entity=rewrite.focus_entity_after,
            turn_index=int(state.total_turns) + 1,
            citations=list(getattr(base, "citations", []) or []),
            cited_pages=list(getattr(base, "cited_pages", []) or []),
            latency_ms=lat,
            language=str(getattr(base, "language", "zh") or "zh"),
            rewrite=rewrite.to_dict(),
            base={
                "trace_id": getattr(base, "trace_id", ""),
                "generation_method": getattr(base, "generation_method", ""),
                "backend": getattr(base, "backend", ""),
                "model": getattr(base, "model", ""),
                "refusal_reason": getattr(base, "refusal_reason", None),
                "answer_shape": (getattr(base, "analysis", {}) or {}).get("answer_shape"),
                "cache": getattr(base, "cache", {}) or {},
                # 工单6：把底层问答的**诊断明细**透传给界面/评测。
                # 里面含 ``diagnostics.hits``（Top-K 的 chunk_id/page/final_score）
                # 与 ``diagnostics.strategy``（检索模式/权重/融合/重排/各阶段耗时/说明），
                # 是「三模式对比」与检索策略留痕的唯一数据来源。
                # 注意：这是**白名单**字典（不是 base.to_dict()），漏加字段界面就会显示空值。
                "diagnostics": getattr(base, "diagnostics", {}) or {},
            },
            session_id=sid,
        )
        self._record(sid, state, out)
        if cache_key is not None:
            self.cache.put(cache_key, out.to_dict())
        self.log.log_event("multiturn.ask", session_id=sid, turn=out.turn_index,
                           strategy=out.strategy, focus=out.focus_entity,
                           refused=out.refused, total_ms=total_ms,
                           question_digest=out.question[:60],
                           rewritten_digest=out.rewritten_query[:80])
        return out

    def _record(self, sid: str, state: Any, out: MultiTurnAnswer) -> None:
        """把本轮写回会话状态（焦点实体 / 槽位 / 轮次）。"""
        from .core.dialogue_state import TurnRecord

        state.push_entity(out.focus_entity or "")
        self.states.append(sid, TurnRecord(
            question=out.question, rewritten_query=out.rewritten_query, answer=out.answer,
            trace_id=str(out.base.get("trace_id") or ""), cited_pages=list(out.cited_pages),
            focus_entity=out.focus_entity, strategy=out.strategy, refused=out.refused,
            latency_ms=float(out.latency_ms.get("total") or 0.0),
            slot=(out.rewrite or {}).get("slot"),
        ))

    # ------------------------------------------------------------------ 辅助
    def _from_cache(self, payload: dict[str, Any], question: str, sid: str, state: Any,
                    rewrite: Any, rewrite_ms: float, t0: float) -> MultiTurnAnswer:
        """命中**对话级缓存**：用当前轮次/会话号重建答案，其余字段照抄缓存。

        缓存里的证据、引用与答案文本与首次计算时完全一致（键含 session_id 与改写后问句），
        因此这里只更新「本次请求」才有意义的字段，并把检索/生成耗时记 0（确实没执行）。
        """
        self._stats["cache_hit"] += 1
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        data = dict(payload)
        lat = dict(data.get("latency_ms") or {})
        lat.update({"rewrite": rewrite_ms, "retrieve": 0.0, "generate": 0.0, "total": total_ms})
        data.update({
            "question": question if isinstance(question, str) else "",
            "rewritten_query": rewrite.rewritten,
            "strategy": rewrite.strategy,
            "focus_entity": rewrite.focus_entity_after,
            "turn_index": int(state.total_turns) + 1,
            "latency_ms": lat,
            "rewrite": rewrite.to_dict(),
            "session_id": sid,
        })
        base = dict(data.get("base") or {})
        base["cache"] = {"hit": True, "layer": "dialogue"}
        data["base"] = base
        out = MultiTurnAnswer(**{k: v for k, v in data.items() if k in MultiTurnAnswer.__slots__})
        self._record(sid, state, out)
        self.log.log_event("multiturn.cache_hit", session_id=sid, turn=out.turn_index,
                           total_ms=total_ms, rewritten_digest=out.rewritten_query[:80])
        return out

    def reset(self, session_id: str = "default") -> None:
        """清空某个会话的多轮状态**与其对话级缓存条目**。"""
        sid = session_id or "default"
        self.states.reset(sid)
        removed = self.cache.invalidate_session(sid)
        self.log.log_event("multiturn.reset", session_id=sid, cache_removed=removed)

    def history_digest(self, session_id: str = "default", *, keep_recent: int = 3,
                       max_chars: int = 240) -> dict[str, Any]:
        """把较早的对话轮次压成一段紧凑摘要（工单5 优化层：对话历史压缩）。

        返回 ``CompressedHistory.to_dict()``；无历史时 ``digest`` 为空串。
        """
        from .core.history_compress import compress_history

        state = self.states.get(session_id or "default")
        turns = [t.to_dict() for t in state.recent(self.max_turns)]
        out = compress_history(turns, keep_recent=keep_recent, max_chars=max_chars,
                               focus_entity=state.focus_entity)
        return out.to_dict()

    def history(self, session_id: str = "default") -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.states.history(session_id or "default")]

    def state(self, session_id: str = "default") -> dict[str, Any]:
        return self.states.get(session_id or "default").to_dict()

    def stats(self) -> dict[str, Any]:
        return {**self._stats, "dialogue_cache": self.cache.stats(),
                "sessions": self.states.stats()}

    def health(self) -> dict[str, Any]:
        """健康检查（供 Web/API 的 ``/health`` 使用）。"""
        out: dict[str, Any] = {"status": "ok", "work_order": WORK_ORDER,
                               "max_turns": self.max_turns, "stats": self.stats()}
        try:
            out["index"] = self.engine.health()
            out["entities"] = sorted((self._ensure_rewriter().corpus_issuers or {}).keys())
        except Exception as exc:  # noqa: BLE001 —— 健康检查本身不许抛
            out.update({"status": "degraded", "error": f"{type(exc).__name__}: {exc}"})
        return out


__all__ = ["WORK_ORDER", "EMPTY_INPUT_HINT", "MultiTurnAnswer", "MultiTurnEngine"]
