# -*- coding: utf-8 -*-
"""工单6 检索策略层 —— **向量检索 / 全文检索 / 混合检索** 三种模式的统一配置与调度。

工单编号：人工智能NLP-RAG-混合检索任务

本模块是工单6 的**核心**：把 工单4/5 既有的「向量通道 + BM25 通道 + RRF 融合」这一条写死的
链路，重构为**可配置、可切换、可解释**的检索策略。

三种检索模式
------------
============  ==================================================================
模式           链路
============  ==================================================================
``vector``    嵌入模型（bge-m3 / m3e-base 可切）→ 余弦召回 Top-K → **重排**
``fulltext``  倒排索引（布尔 / 短语 / 模糊 / 多字段）→ BM25 打分 → **重排**
``hybrid``    两路并行 → **融合算法**（加权平均 / RRF / 投票）→ **重排**
============  ==================================================================

可配置项（工单原文「检索策略配置」逐条对应）
--------------------------------------------
* **向量检索**：``embedding_model``（嵌入模型）、``vector_k``（召回 Top-K）、
  ``score_threshold`` / ``vector_min_similarity``（Score 阈值）、``rerank_model``（重排模型）
* **全文检索**：``fulltext_fields``（正文/标题/关键词/摘要 多字段）、``fulltext_op``（默认布尔运算符）、
  ``enable_fuzzy``（模糊匹配）；布尔与短语由查询串自身语法决定
* **混合检索**：``vector_weight`` / ``fulltext_weight``（权重）、``fusion``（融合算法）、``rrf_k``

重排算法（``rerank_model``）
----------------------------
``none`` 不重排｜``rule`` 规则特征重排（工单4 八维特征，默认）｜
``tfidf`` TF-IDF 余弦重排｜``cross_encoder`` CrossEncoder（缺模型时**显式降级为 rule 并记录原因**，不静默）

设计纪律
--------
* **不破坏既有行为**：**不传 ``strategy`` 时**（``retrieve(..., strategy=None)``）走 工单4 原生的
  ``hybrid_search``，行为与基线逐位一致 —— 这是「工单4/5 能力不退化」的保证。
  一旦传了 ``strategy``，就走本模块的链路，它与 ``hybrid_search`` **并不等价**，已知差异：
  ① 不做查询富化（``analysis.field_terms`` 不再拼进查询）；
  ② 无「单路 rank1 救援分」（``rank_rescue_weight``）；
  ③ 无重排预算裁剪（``RERANK_BUDGET=60`` / ``RERANK_MIN_RELATIVE``）；
  ④ 候选池口径不同（本模块按融合分取 Top-K，``hybrid_search`` 先裁剪再重排）。
  因此**不得声称「策略层默认 == 基线」**；需要基线一致时请显式不传策略。
  本模块 ``StrategyConfig.fusion`` 默认是 ``weighted_average``（工单要求的最低需求算法），
  不是 工单4 的 RRF；要复现 RRF 请用 ``preset("hybrid_rrf")``。
* **不同嵌入模型的向量不可混用**：``bundle_for_model`` 按「模型 → 索引目录」取向量索引，
  找不到就显式报错，绝不拿 bge-m3 的索引去跑 m3e 的查询向量。
* 融合算法沿用 ``fusion.py`` 的纪律：跨语义空间的原始分**必须先归一化再相加**。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import fusion as fusion_mod
from .embedding_models import EmbeddingModelSpec, embed_query_with_model, resolve_model
from .fusion import ALGORITHM_LABELS, ALGORITHMS, DEFAULT_RRF_K
from .fulltext_query import (
    DEFAULT_FIELDS,
    FIELD_LABELS,
    search_fulltext,
)
from .text_utils import tokenize
from .vector_store import search_vectors

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

# ---------------------------------------------------------------------------
# 枚举（用字面量常量而非 Enum，便于 JSON/界面直接往返）
# ---------------------------------------------------------------------------
MODE_VECTOR = "vector"
MODE_FULLTEXT = "fulltext"
MODE_HYBRID = "hybrid"
MODES: tuple[str, ...] = (MODE_VECTOR, MODE_FULLTEXT, MODE_HYBRID)
MODE_LABELS: dict[str, str] = {
    MODE_VECTOR: "向量检索（嵌入召回 + 重排）",
    MODE_FULLTEXT: "全文检索（倒排索引 + 布尔/短语/模糊）",
    MODE_HYBRID: "混合检索（向量 + 全文 融合）",
}

RERANK_NONE = "none"
RERANK_RULE = "rule"
RERANK_TFIDF = "tfidf"
RERANK_CROSS_ENCODER = "cross_encoder"
RERANK_MODELS: tuple[str, ...] = (RERANK_NONE, RERANK_RULE, RERANK_TFIDF, RERANK_CROSS_ENCODER)
RERANK_LABELS: dict[str, str] = {
    RERANK_NONE: "不重排（按召回分）",
    RERANK_RULE: "规则重排（八维特征，默认，最快）",
    RERANK_TFIDF: "TF-IDF 余弦重排",
    RERANK_CROSS_ENCODER: "CrossEncoder 重排（缺模型时降级为规则重排）",
}

#: CrossEncoder 候选模型（本机无网络下载权重时不可用 —— 会显式降级）
CROSS_ENCODER_CANDIDATES: tuple[str, ...] = (
    "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


def _coerce_bool(value: Any) -> bool:
    """把 JSON / 表单 / 配置里的「布尔」稳妥地转成 ``bool``。

    ⚠️ **不可直接用 ``bool(value)``**（评审修正）：``bool("false")`` 是 ``True``，
    界面/接口把复选框传成字符串时会把用户的「关」静默变成「开」。这里显式识别常见字面量，
    无法识别时**回退到 ``bool(value)`` 但保持与 Python 真值语义一致**（非空字符串为真）。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on", "是", "开", "真"):
        return True
    if text in ("0", "false", "no", "n", "off", "否", "关", "假", ""):
        return False
    return bool(value)


#: 全文检索的隐式布尔运算符（``_Parser`` 只认这两个；其余值必须在校验期被拒）
FULLTEXT_OPS: tuple[str, ...] = ("OR", "AND")


# ---------------------------------------------------------------------------
# 策略配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """一次检索的全部可调参数（冻结，可安全跨线程共享）。"""

    # ---- 模式 ----
    mode: str = MODE_HYBRID
    # ---- 向量通道 ----
    embedding_model: str = "bge-m3"
    vector_k: int = 20                 # 向量召回 Top-K
    vector_min_similarity: float = 0.0  # 余弦相似度**绝对**下限（0 = 不过滤）
    # ---- 全文通道 ----
    fulltext_k: int = 20               # 全文召回 Top-K
    fulltext_fields: tuple[str, ...] = DEFAULT_FIELDS
    fulltext_op: str = "OR"            # 隐式运算符（OR / AND）
    enable_fuzzy: bool = True          # 模糊匹配（查询串写 ``词~`` 才真正触发）
    # ---- 混合融合 ----
    fusion: str = fusion_mod.ALGORITHM_WEIGHTED_AVERAGE
    vector_weight: float = 0.5
    fulltext_weight: float = 0.5
    rrf_k: int = DEFAULT_RRF_K
    # ---- 重排与输出 ----
    rerank_model: str = RERANK_RULE
    top_k: int = 5
    score_threshold: float = 0.0       # **归一化**分数下限（相对本次最高分，0 = 不过滤）
    use_image_channel: bool = True     # 保留工单4 图像证据权重校正

    # -- 序列化 --
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "embedding_model": self.embedding_model,
            "vector_k": self.vector_k, "vector_min_similarity": self.vector_min_similarity,
            "fulltext_k": self.fulltext_k, "fulltext_fields": list(self.fulltext_fields),
            "fulltext_op": self.fulltext_op, "enable_fuzzy": self.enable_fuzzy,
            "fusion": self.fusion, "vector_weight": self.vector_weight,
            "fulltext_weight": self.fulltext_weight, "rrf_k": self.rrf_k,
            "rerank_model": self.rerank_model, "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "use_image_channel": self.use_image_channel,
            # 派生展示字段（只读、不参与构造）
            "mode_label": MODE_LABELS.get(self.mode, self.mode),
            "fusion_label": ALGORITHM_LABELS.get(self.fusion, self.fusion),
            "rerank_label": RERANK_LABELS.get(self.rerank_model, self.rerank_model),
            "fulltext_field_labels": [FIELD_LABELS.get(f, f) for f in self.fulltext_fields],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "StrategyConfig":
        """从 dict 构造（忽略未知键与派生展示键，便于界面直接回传整个 dict）。"""
        if not data:
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in allowed or value is None:
                continue
            if key == "fulltext_fields":
                if isinstance(value, str):
                    value = tuple(p for p in value.replace("，", ",").split(",") if p.strip())
                kwargs[key] = tuple(str(v) for v in value)
            elif key in ("vector_k", "fulltext_k", "top_k", "rrf_k"):
                kwargs[key] = int(value)
            elif key in ("vector_min_similarity", "vector_weight", "fulltext_weight",
                         "score_threshold"):
                kwargs[key] = float(value)
            elif key in ("enable_fuzzy", "use_image_channel"):
                kwargs[key] = _coerce_bool(value)
            elif key == "fulltext_op":
                kwargs[key] = str(value).strip().upper() or "OR"
            else:
                kwargs[key] = str(value)
        return cls(**kwargs)

    def replace(self, **changes: Any) -> "StrategyConfig":
        """返回替换若干字段后的新策略（``None`` 值被忽略，便于界面「不修改」语义）。"""
        clean = {k: v for k, v in changes.items() if v is not None}
        return replace(self, **clean)

    def label(self) -> str:
        """一行中文摘要（日志/报告用）。"""
        if self.mode == MODE_VECTOR:
            return f"向量检索｜{self.embedding_model}｜召回{self.vector_k}｜重排={self.rerank_model}"
        if self.mode == MODE_FULLTEXT:
            fields = "+".join(FIELD_LABELS.get(f, f) for f in self.fulltext_fields)
            return (f"全文检索｜字段={fields}｜默认{self.fulltext_op}｜"
                    f"模糊={'开' if self.enable_fuzzy else '关'}｜重排={self.rerank_model}")
        return (f"混合检索｜融合={self.fusion}｜向量权重{self.vector_weight}/"
                f"全文权重{self.fulltext_weight}｜{self.embedding_model}｜重排={self.rerank_model}")

    def validate(self) -> list[str]:
        """返回配置错误列表（空列表 = 合法）。"""
        problems: list[str] = []
        if self.mode not in MODES:
            problems.append(f"未知检索模式：{self.mode!r}（可选：{', '.join(MODES)}）")
        if self.fusion not in ALGORITHMS:
            problems.append(f"未知融合算法：{self.fusion!r}（可选：{', '.join(ALGORITHMS)}）")
        if self.rerank_model not in RERANK_MODELS:
            problems.append(f"未知重排模型：{self.rerank_model!r}（可选：{', '.join(RERANK_MODELS)}）")
        if int(self.vector_k) < 1:
            problems.append(f"vector_k 必须 ≥1（当前 {self.vector_k}）")
        if int(self.fulltext_k) < 1:
            problems.append(f"fulltext_k 必须 ≥1（当前 {self.fulltext_k}）")
        if int(self.top_k) < 1:
            problems.append(f"top_k 必须 ≥1（当前 {self.top_k}）")
        if float(self.vector_weight) < 0 or float(self.fulltext_weight) < 0:
            problems.append("权重不得为负")
        if float(self.vector_weight) == 0.0 and float(self.fulltext_weight) == 0.0:
            problems.append("向量权重与全文权重不能同时为 0")
        if not 0.0 <= float(self.score_threshold) <= 1.0:
            problems.append(f"score_threshold 必须落在 [0,1]（当前 {self.score_threshold}）")
        # 实测陷阱：vector_min_similarity 曾不做校验，传 1.5 会**静默裁光全部向量候选**
        # （余弦相似度上限为 1），传负数则完全无效 —— 两者都该在入口就报错。
        if not 0.0 <= float(self.vector_min_similarity) <= 1.0:
            problems.append(f"vector_min_similarity 必须落在 [0,1]（余弦相似度上限为 1；"
                            f"当前 {self.vector_min_similarity}）")
        for name in self.fulltext_fields:
            if name not in FIELD_LABELS:
                problems.append(f"未知全文检索字段：{name!r}（可选：{', '.join(FIELD_LABELS)}）")
        if not self.fulltext_fields:
            problems.append("全文检索字段不能为空")
        # 实测陷阱：``fulltext_op="XOR"`` 曾能通过校验，运行时被 ``_Parser`` 静默当 OR 处理。
        # 非法运算符必须在校验期被拒，而不是悄悄降级。
        if str(self.fulltext_op).strip().upper() not in FULLTEXT_OPS:
            problems.append(f"未知布尔运算符：{self.fulltext_op!r}（可选：{', '.join(FULLTEXT_OPS)}）")
        return problems


#: 预设策略（界面下拉框直接取用；工单要求的三种模式各自有默认预设）
PRESETS: dict[str, StrategyConfig] = {
    "vector": StrategyConfig(mode=MODE_VECTOR, rerank_model=RERANK_RULE),
    "fulltext": StrategyConfig(mode=MODE_FULLTEXT, rerank_model=RERANK_RULE),
    "hybrid": StrategyConfig(mode=MODE_HYBRID, fusion=fusion_mod.ALGORITHM_WEIGHTED_AVERAGE,
                             vector_weight=0.5, fulltext_weight=0.5, rerank_model=RERANK_RULE),
    "hybrid_rrf": StrategyConfig(mode=MODE_HYBRID, fusion=fusion_mod.ALGORITHM_RRF,
                                 vector_weight=0.5, fulltext_weight=0.5, rerank_model=RERANK_RULE),
    "hybrid_voting": StrategyConfig(mode=MODE_HYBRID, fusion=fusion_mod.ALGORITHM_VOTING,
                                    vector_weight=0.5, fulltext_weight=0.5, rerank_model=RERANK_RULE),
}


def preset(name: str) -> StrategyConfig:
    """取预设策略；未命中抛 ``ValueError``（不静默回退到默认）。"""
    key = str(name or "").strip().lower()
    if key not in PRESETS:
        raise ValueError(f"未知预设：{name!r}（可选：{', '.join(PRESETS)}）")
    return PRESETS[key]


def describe_modes() -> list[dict[str, Any]]:
    """给界面用的模式清单。"""
    return [{"id": m, "label": MODE_LABELS[m]} for m in MODES]


def describe_rerank_models() -> list[dict[str, Any]]:
    """给界面用的重排模型清单。"""
    return [{"id": m, "label": RERANK_LABELS[m]} for m in RERANK_MODELS]


def describe_fulltext_fields() -> list[dict[str, Any]]:
    """给界面用的全文检索字段清单（含默认加成）。"""
    from .fulltext_query import describe_fields

    return describe_fields()


def fulltext_syntax_help() -> str:
    """给界面用的全文查询语法提示。"""
    from .fulltext_query import query_syntax_help

    return query_syntax_help()


def list_models_for_ui(*, index_root: Path | str | None = None,
                       base_url: str = "http://127.0.0.1:11434",
                       probe: bool = True) -> list[dict[str, Any]]:
    """给界面用的嵌入模型清单（含「是否已装」「是否已建索引」）。"""
    from .embedding_models import describe_models

    return describe_models(index_root=index_root, base_url=base_url, probe=probe)


# ---------------------------------------------------------------------------
# 模型 → 向量索引（bundle 视图）
# ---------------------------------------------------------------------------
_BUNDLE_VIEW_CACHE: dict[str, Any] = {}
_SPEC_CACHE: dict[str, EmbeddingModelSpec] = {}


def resolve_spec_cached(model_id: str, *, index_root: Path | str | None = None,
                        base_url: str = "http://127.0.0.1:11434") -> EmbeddingModelSpec:
    """带缓存的模型解析（避免每次查询都探测 Ollama）。"""
    key = f"{model_id}|{index_root}|{base_url}"
    cached = _SPEC_CACHE.get(key)
    if cached is not None:
        # index_built 会随建索引变化，这里刷新一次状态（只查文件，不联网）
        if index_root is not None:
            spec_dir = Path(index_root) / cached.slug
            cached.index_dir = str(spec_dir)
            cached.index_built = (spec_dir / "vectors.npy").is_file()
        return cached
    spec = resolve_model(model_id, index_root=index_root, base_url=base_url, probe=False)
    _SPEC_CACHE[key] = spec
    return spec


def bundle_for_model(bundle: Any, spec: EmbeddingModelSpec, cfg: Any, *, logger: Any = None) -> Any:
    """返回把**向量索引换成 ``spec`` 对应索引**的 bundle 视图（块表 / BM25 / 页索引复用）。

    * ``spec.slug`` 与当前 bundle 一致 → 直接返回原 bundle（零开销）；
    * 否则加载 ``{index_dir}/{slug}/`` 的向量索引并 ``dataclasses.replace`` 出视图；
    * 该模型的索引不存在 → 抛 ``IndexMissingError`` 并提示构建命令，**绝不跨模型混用**。
    """
    current = Path(str(getattr(bundle, "index_dir", ""))).name
    if current == spec.slug:
        return bundle
    key = f"{getattr(bundle, 'index_dir', '')}::{spec.slug}"
    cached = _BUNDLE_VIEW_CACHE.get(key)
    if cached is not None:
        return cached
    from .errors import IndexMissingError
    from .vector_store import load_vector_index

    index_dir = Path(str(cfg.paths.index_dir)) / spec.slug
    if not (index_dir / "vectors.npy").is_file():
        raise IndexMissingError(
            f"嵌入模型 {spec.id}（{spec.ollama_model}）的向量索引不存在：{index_dir}",
            detail={"model": spec.id, "index_dir": str(index_dir),
                    "build_command": f"pwsh -NoProfile -File run_py.ps1 "
                                     f"研发/scripts/build_index_multi.py --model {spec.id}"},
        )
    vector_index = load_vector_index(index_dir, expected_dim=int(spec.dim),
                                     use_faiss=bool(getattr(cfg.vector_store, "use_faiss", True)),
                                     logger=logger)
    view = replace(bundle, vector_index=vector_index, vectors=vector_index.vectors)
    _BUNDLE_VIEW_CACHE[key] = view
    return view


def clear_strategy_cache() -> None:
    """清空模型/bundle 视图缓存（测试用）。"""
    _BUNDLE_VIEW_CACHE.clear()
    _SPEC_CACHE.clear()


# ---------------------------------------------------------------------------
# 重排算法
# ---------------------------------------------------------------------------
def _tfidf_cosine(query_tokens: Sequence[str], doc_tokens: Sequence[str],
                  idf: Mapping[str, float]) -> float:
    """TF-IDF 加权向量的余弦相似度（查询侧 1+log(tf)，文档侧 1+log(tf)，均乘 idf）。"""
    if not query_tokens or not doc_tokens:
        return 0.0

    def weights(tokens: Sequence[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return {term: (1.0 + math.log(freq)) * float(idf.get(term, 1.0))
                for term, freq in counts.items()}

    q_vec, d_vec = weights(query_tokens), weights(doc_tokens)
    dot = 0.0
    # 只遍历较短一侧，避免无谓的 hash 查询
    short, long_ = (q_vec, d_vec) if len(q_vec) <= len(d_vec) else (d_vec, q_vec)
    for term, value in short.items():
        other = long_.get(term)
        if other is not None:
            dot += value * other
    if dot <= 0.0:
        return 0.0
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
    if q_norm <= 0.0 or d_norm <= 0.0:
        return 0.0
    return dot / (q_norm * d_norm)


_CROSS_ENCODER_CACHE: dict[str, Any] = {}


def _load_cross_encoder(logger: Any = None) -> tuple[Any, str | None]:
    """尝试加载 CrossEncoder；返回 ``(模型, 失败原因)``。"""
    if "model" in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE["model"], _CROSS_ENCODER_CACHE.get("error")
    model, error = None, None
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        for name in CROSS_ENCODER_CANDIDATES:
            try:
                model = CrossEncoder(name)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 —— 逐个候选尝试，失败原因如实记录
                error = f"{name}: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = f"sentence_transformers 不可用：{type(exc).__name__}: {exc}"
    _CROSS_ENCODER_CACHE["model"] = model
    _CROSS_ENCODER_CACHE["error"] = error
    if logger is not None:
        logger.log_event("rerank.cross_encoder", available=model is not None, error=error)
    return model, error


def apply_rerank(question: str, candidates: list[dict[str, Any]], strategy: StrategyConfig, *,
                 analysis: Any = None, bundle: Any = None, logger: Any = None
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按 ``strategy.rerank_model`` 重排候选，返回 ``(重排后候选, 重排说明)``。

    * ``rule``：复用 工单4 ``reranker.rerank`` 的八维特征（词法覆盖/字段/答案形态/模态/
      主体路由/极性/数值邻近/身份簇）；
    * ``tfidf``：查询与候选正文的 TF-IDF 余弦；
    * ``cross_encoder``：加载不到模型时**显式降级为 rule**，并在 ``note`` 中写明原因；
    * ``none``：按融合分原序。

    统一后的 ``final_score`` 口径：所有算法都输出 **``[0,1]`` 归一化分 + 特征分** 的加权和，
    便于跨模式比较与阈值裁剪。
    """
    note: dict[str, Any] = {"requested": strategy.rerank_model, "applied": strategy.rerank_model,
                            "degraded": False, "reason": None}
    if not candidates:
        return [], note
    idf = None
    if bundle is not None:
        bm25 = getattr(getattr(bundle, "bm25", None), "bm25", None)
        idf = getattr(bm25, "idf", None)

    if strategy.rerank_model == RERANK_NONE:
        ordered = sorted(candidates, key=lambda c: (-float(c.get("rrf_score") or 0.0),
                                                    str(c.get("chunk_id"))))
        max_score = max((float(c.get("rrf_score") or 0.0) for c in ordered), default=0.0) or 1.0
        out: list[dict[str, Any]] = []
        for item in ordered:
            row = dict(item)
            row["rerank"] = {"method": RERANK_NONE}
            row["final_score"] = round(float(item.get("rrf_score") or 0.0) / max_score, 6)
            out.append(row)
        return out, note

    if strategy.rerank_model == RERANK_TFIDF:
        query_tokens = tokenize(question)
        idf_map = idf or {}
        for item in candidates:
            doc_tokens = tokenize(str(item.get("content") or ""))
            item["tfidf"] = round(_tfidf_cosine(query_tokens, doc_tokens, idf_map), 6)
        rrf_values = [float(c.get("rrf_score") or 0.0) for c in candidates]
        max_rrf = max(rrf_values) or 1.0
        for item in candidates:
            item["rrf_norm"] = round(float(item.get("rrf_score") or 0.0) / max_rrf, 6)
            item["rerank"] = {"method": RERANK_TFIDF, "tfidf": item["tfidf"]}
            item["final_score"] = round(0.5 * item["rrf_norm"] + 0.5 * item["tfidf"], 6)
        candidates.sort(key=lambda c: (-float(c.get("final_score") or 0.0), str(c.get("chunk_id"))))
        return candidates, note

    if strategy.rerank_model == RERANK_CROSS_ENCODER:
        model, error = _load_cross_encoder(logger)
        if model is not None:
            try:
                pairs = [(question, str(c.get("content") or "")[:1500]) for c in candidates]
                scores = [float(s) for s in model.predict(pairs)]
                lo, hi = min(scores), max(scores)
                span = (hi - lo) or 1.0
                for item, score in zip(candidates, scores):
                    item["cross_encoder"] = round(score, 6)
                    item["rerank"] = {"method": RERANK_CROSS_ENCODER, "score": round(score, 6)}
                    item["final_score"] = round((score - lo) / span, 6)
                candidates.sort(key=lambda c: (-float(c.get("final_score") or 0.0),
                                               str(c.get("chunk_id"))))
                note["applied"] = RERANK_CROSS_ENCODER
                return candidates, note
            except Exception as exc:  # noqa: BLE001 —— 推理失败同样降级，不阻断检索
                error = f"CrossEncoder 推理失败：{type(exc).__name__}: {exc}"
        note.update({"applied": RERANK_RULE, "degraded": True,
                     "reason": f"CrossEncoder 不可用，已显式降级为规则重排：{error}"})
        if logger is not None:
            logger.log_event("rerank.degraded", level="WARNING", requested=RERANK_CROSS_ENCODER,
                             applied=RERANK_RULE, reason=note["reason"])

    # RERANK_RULE（默认）—— 复用 工单4 的八维特征重排
    from .reranker import rerank as rule_rerank

    ranked = rule_rerank(analysis, candidates, idf=idf, rrf_weight=1.0, feature_weight=1.0,
                         logger=logger)
    for item in ranked:
        item.setdefault("rerank", {})
        if isinstance(item.get("rerank"), dict):
            item["rerank"]["method"] = RERANK_RULE
    return ranked, note


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def search_by_strategy(bundle: Any, question: str, strategy: StrategyConfig, *,
                       analysis: Any = None, cfg: Any = None,
                       allowed_ids: set[str] | None = None,
                       logger: Any = None) -> dict[str, Any]:
    """按 ``strategy`` 执行一次检索，返回结构化结果（含各通道明细，供界面/报告解释）。

    返回 dict 字段：
        ``hits``：重排 + 阈值 + Top-K 后的候选列表（每项含 final_score / 通道名次 / 重排明细）；
        ``vector_hits`` / ``fulltext_hits``：两路召回原始结果（``[(chunk_id, score)]``）；
        ``fusion``：融合算法与生效权重；
        ``rerank``：重排算法与降级说明；
        ``timings``：各阶段耗时（ms）；
        ``notes``：全流程的显式说明（降级、阈值裁剪、模糊扩展等）。
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    notes: list[str] = []
    # 入参兼容（评审修正）：曾写成 ``StrategyConfig.from_dict(strategy.to_dict())``，
    # 传纯 dict（界面/HTTP 回传 to_dict 结果）时会抛 ``AttributeError: 'dict' object has no
    # attribute 'to_dict'``。这里显式区分「已是配置对象」与「等价 dict」两条路径。
    if not isinstance(strategy, StrategyConfig):
        if not isinstance(strategy, Mapping):
            raise TypeError(
                f"strategy 必须是 StrategyConfig 或等价 dict，收到 {type(strategy).__name__}")
        strategy = StrategyConfig.from_dict(strategy)
    problems = strategy.validate()
    if problems:
        raise ValueError("检索策略配置非法：" + "；".join(problems))

    if cfg is None:
        from .config import get_config

        cfg = get_config()
    index_root = Path(str(cfg.paths.index_dir))

    # ---------- ① 向量通道 ----------
    vector_hits: list[tuple[str, float]] = []
    vector_note: dict[str, Any] = {}
    if strategy.mode in (MODE_VECTOR, MODE_HYBRID):
        t0 = time.perf_counter()
        spec = resolve_spec_cached(strategy.embedding_model, index_root=index_root,
                                   base_url=str(getattr(cfg.llm, "ollama_base_url",
                                                        "http://127.0.0.1:11434")))
        view = bundle_for_model(bundle, spec, cfg, logger=logger)
        query_vector = embed_query_with_model(
            question, spec,
            base_url=str(getattr(cfg.llm, "ollama_base_url", "http://127.0.0.1:11434")),
            logger=logger)
        vector_hits = search_vectors(view.vector_index, query_vector, int(strategy.vector_k),
                                     allowed_ids=allowed_ids, logger=logger)
        if float(strategy.vector_min_similarity) > 0:
            before = len(vector_hits)
            vector_hits = [(cid, s) for cid, s in vector_hits
                           if float(s) >= float(strategy.vector_min_similarity)]
            if len(vector_hits) != before:
                notes.append(f"向量相似度阈值 {strategy.vector_min_similarity} 裁掉 "
                             f"{before - len(vector_hits)} 条")
        timings["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        vector_note = {"model": spec.id, "ollama_model": spec.ollama_model, "dim": spec.dim,
                       "index_dir": str(getattr(view.vector_index, "index_dir", "")),
                       "recalled": len(vector_hits),
                       "top_similarity": round(float(vector_hits[0][1]), 6) if vector_hits else None}

    # ---------- ② 全文通道 ----------
    fulltext_hits: list[tuple[str, float]] = []
    fulltext_note: dict[str, Any] = {}
    if strategy.mode in (MODE_FULLTEXT, MODE_HYBRID):
        t0 = time.perf_counter()
        ft_diag: dict[str, Any] = {}
        fulltext_hits = search_fulltext(
            bundle, question, top_k=int(strategy.fulltext_k),
            fields=strategy.fulltext_fields, default_op=strategy.fulltext_op,
            enable_fuzzy=bool(strategy.enable_fuzzy), allowed_ids=allowed_ids,
            diag=ft_diag, logger=logger)
        timings["fulltext_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        fulltext_note = {"fields": list(strategy.fulltext_fields),
                         "default_op": strategy.fulltext_op,
                         "fuzzy": bool(strategy.enable_fuzzy),
                         "recalled": len(fulltext_hits),
                         "top_score": round(float(fulltext_hits[0][1]), 6) if fulltext_hits else None,
                         "query_ast": ft_diag.get("query_ast"),
                         "boolean_hits": ft_diag.get("boolean_hits"),
                         "positive_terms": (ft_diag.get("positive_terms") or [])[:12],
                         "fuzzy_expansions": ft_diag.get("fuzzy_expansions") or {},
                         "oov_terms": ft_diag.get("oov_terms") or []}
        oov = fulltext_note["oov_terms"]
        if oov:
            notes.append(
                "全文通道词表外词项：" + "、".join(str(t) for t in oov[:8])
                + "（分词停用词或语料中不存在；若该词参与 AND 约束会导致 0 命中）")
        if not fulltext_hits:
            notes.append("全文通道 0 召回")

    # ---------- ③ 融合 ----------
    t0 = time.perf_counter()
    if strategy.mode == MODE_VECTOR:
        if not vector_hits:
            fused = fusion_mod.FusionResult(algorithm=fusion_mod.SINGLE_CHANNEL)
            notes.append("向量通道 0 召回")
        else:
            top = float(vector_hits[0][1]) or 1.0
            fused = fusion_mod.FusionResult(
                algorithm=fusion_mod.SINGLE_CHANNEL,
                scores={cid: round(float(s) / top, 8) for cid, s in vector_hits},
                detail={cid: {"vector_score": float(s), "vector_rank": r}
                        for r, (cid, s) in enumerate(vector_hits, 1)})
    elif strategy.mode == MODE_FULLTEXT:
        if not fulltext_hits:
            fused = fusion_mod.FusionResult(algorithm=fusion_mod.SINGLE_CHANNEL)
            notes.append("全文通道 0 召回")
        else:
            top = float(fulltext_hits[0][1]) or 1.0
            fused = fusion_mod.FusionResult(
                algorithm=fusion_mod.SINGLE_CHANNEL,
                scores={cid: round(float(s) / top, 8) for cid, s in fulltext_hits},
                detail={cid: {"fulltext_score": float(s), "fulltext_rank": r}
                        for r, (cid, s) in enumerate(fulltext_hits, 1)})
    else:
        fused = fusion_mod.fuse(strategy.fusion, vector_hits, fulltext_hits,
                                vector_weight=float(strategy.vector_weight),
                                fulltext_weight=float(strategy.fulltext_weight),
                                rrf_k=int(strategy.rrf_k))
        if not fused.scores:
            notes.append("两路均 0 召回")
    timings["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    notes.extend(fused.notes)

    # ---------- ④ 候选装配 ----------
    candidates: list[dict[str, Any]] = []
    for cid, score in fused.ranked():
        row = bundle.chunks.get(cid)
        if row is None:
            continue
        item = dict(row)
        detail = fused.detail.get(cid, {})
        item.update({
            "chunk_id": cid,
            "rrf_score": score,                    # 兼容工单4 rerank 的入参名（此处为「融合分」）
            "fusion_score": score,
            "vector_rank": detail.get("vector_rank"),
            "vector_score": detail.get("vector_score"),
            "fulltext_rank": detail.get("fulltext_rank"),
            "fulltext_score": detail.get("fulltext_score"),
            "fusion_detail": detail,
        })
        candidates.append(item)

    # ---------- ⑤ 重排 ----------
    t0 = time.perf_counter()
    ranked, rerank_note = apply_rerank(question, candidates, strategy, analysis=analysis,
                                       bundle=bundle, logger=logger)
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if rerank_note.get("degraded"):
        notes.append(str(rerank_note.get("reason")))

    # ---------- ⑥ 图像证据权重校正（工单4 口径：picture=0.4 弱证据） ----------
    if strategy.use_image_channel:
        for item in ranked:
            weight = float(item.get("evidence_weight") or 1.0)
            if weight != 1.0:
                item["evidence_weight_applied"] = weight
                item["final_score"] = round(float(item.get("final_score") or 0.0) * weight, 6)
        ranked.sort(key=lambda c: (-float(c.get("final_score") or 0.0), str(c.get("chunk_id"))))

    # ---------- ⑦ 归一化阈值 + Top-K ----------
    max_final = max((float(c.get("final_score") or 0.0) for c in ranked), default=0.0)
    if float(strategy.score_threshold) > 0 and max_final > 0:
        cut = max_final * float(strategy.score_threshold)
        kept = [c for c in ranked if float(c.get("final_score") or 0.0) >= cut]
        if len(kept) != len(ranked):
            notes.append(f"归一化分数阈值 {strategy.score_threshold} 裁掉 "
                         f"{len(ranked) - len(kept)} 条（cut={cut:.4f}）")
        ranked = kept
    hits = ranked[: max(1, int(strategy.top_k))]
    for rank, item in enumerate(hits, 1):
        item["rank"] = rank
        item["score"] = item.get("final_score")
        item["page_basis"] = "1-based-physical"

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    timings["total_ms"] = total_ms
    # 说明去重（保序）：同一原因可能被通道与融合两处记录，报告里不应重复出现
    deduped: list[str] = []
    for item in notes:
        if item and item not in deduped:
            deduped.append(item)
    if logger is not None:
        logger.log_event("strategy.search", mode=strategy.mode, fusion=strategy.fusion,
                         rerank=strategy.rerank_model, question=question[:60],
                         vector_hits=len(vector_hits), fulltext_hits=len(fulltext_hits),
                         returned=len(hits), timings=timings,
                         top=[c.get("chunk_id") for c in hits[:5]])
    return {
        "strategy": strategy.to_dict(),
        "strategy_label": strategy.label(),
        "hits": hits,
        "vector_hits": vector_hits,
        "fulltext_hits": fulltext_hits,
        "vector": vector_note,
        "fulltext": fulltext_note,
        "fusion": fused.to_dict(),
        "rerank": rerank_note,
        "timings": timings,
        "notes": deduped,
        "candidate_count": len(candidates),
    }


def warmup_strategy(*, cfg: Any = None, logger: Any = None) -> dict[str, Any]:
    """预热策略层：模型解析 + 多字段倒排索引 + 默认 bundle 视图。"""
    from .config import get_config
    from .fulltext_query import get_fulltext_index
    from .retriever import load_bundle

    config = cfg or get_config()
    started = time.perf_counter()
    bundle = load_bundle(config, logger=logger)
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    index = get_fulltext_index(bundle, logger=logger)
    out["fulltext_index"] = {"chunks": index.size(), "fields": {f: len(index.vocab.get(f, ()))
                                                               for f in index.postings},
                             "ms": round((time.perf_counter() - t0) * 1000, 2)}
    t0 = time.perf_counter()
    models = []
    for mid in ("bge-m3", "m3e-base"):
        try:
            spec = resolve_spec_cached(mid, index_root=Path(str(config.paths.index_dir)),
                                       base_url=str(config.llm.ollama_base_url))
            models.append({"id": spec.id, "dim": spec.dim, "index_built": spec.index_built})
        except ValueError as exc:
            models.append({"id": mid, "error": str(exc)})
    out["models"] = models
    out["models_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    if logger is not None:
        logger.log_event("strategy.warmup", **out)
    return out


__all__ = [
    "WORK_ORDER", "MODE_VECTOR", "MODE_FULLTEXT", "MODE_HYBRID", "MODES", "MODE_LABELS",
    "RERANK_NONE", "RERANK_RULE", "RERANK_TFIDF", "RERANK_CROSS_ENCODER", "RERANK_MODELS",
    "RERANK_LABELS", "StrategyConfig", "PRESETS", "preset", "describe_modes",
    "describe_rerank_models", "search_by_strategy", "apply_rerank", "bundle_for_model",
    "resolve_spec_cached", "clear_strategy_cache", "warmup_strategy",
    "describe_fulltext_fields", "fulltext_syntax_help", "list_models_for_ui", "fusion_mod",
]
