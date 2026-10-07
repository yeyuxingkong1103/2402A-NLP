"""
重排器（Reranker）—— 召回之后的精排
工单编号：人工智能NLP-RAG-混合检索任务

对应工单功能需求（工单6「2 功能详细需求 →（1）向量检索（召回+重排）」）：

    需求描述：应用重排算法（如基于LLM的重排器）对召回结果进行优化排序。
    技术要求：提供至少 3 种重排算法（如基于 LLM 的重排器、基于 TF-IDF 的重排器、
             基于用户反馈的自适应重排器）。

—— 为什么「召回」之后还要「重排」——

召回（retrieval）与重排（rerank）是两件不同的事：
  * **召回**要的是「别漏」，允许粗糙 —— 所以它是双塔结构：query 与 doc 各编码一次，
    可以预先算好向量、用一次矩阵乘法比完全库，快到毫秒级；代价是 query 与 doc
    在编码时**互不可见**，对「措辞不同但意思相同」「否定与肯定」「数字对不上」这类
    细粒度差异不敏感。
  * **重排**要的是「别错」，允许慢 —— 它把 (query, doc) **放在一起**看，
    可以建模交互（attention 级别的词对词匹配，或直接让 LLM 读一遍）。
    代价是每个候选都要现算一遍，成本是召回的几十倍，所以只能作用于
    前 N 个候选（这里的 N = `RERANK_CANDIDATES`，默认 12）。

本文件实现 5 种重排器，用一个统一接口暴露，由 `RERANKER` 配置选择：

    none      不重排（对照组，用来量化"重排到底带来了多少提升"）
    tfidf     基于 TF-IDF 的重排器        —— 工单点名②
    llm       基于 LLM 的重排器            —— 工单点名①
    feedback  基于用户反馈的自适应重排器   —— 工单点名③
    cross     交叉编码器（bge-reranker-v2-m3），工单没要求，作为第 4 种可选实现

所有重排器都遵守同一条纪律：**只做微调，不做推翻**。
最终分 = `RERANK_ALPHA · 重排分 + (1-RERANK_ALPHA) · 原始归一化分`。
原因见 config.RERANK_ALPHA 的说明 —— 单个重排器出错时不至于把正确答案踢出上下文。
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import (
    CROSS_ENCODER_DEVICE,
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_MAX_PAIRS,
    CROSS_ENCODER_PATH,
    FEEDBACK_ENABLED,
    FEEDBACK_LR,
    FEEDBACK_PATH,
    FEEDBACK_PRIORS,
    LLM_RERANK_BATCH,
    LLM_RERANK_MAX_TOKENS,
    RERANK_ALPHA,
    TFIDF_RERANK_NGRAM,
    WORK_ORDER_NOS,  # noqa: F401
)
from .fulltext import tokenize

logger = logging.getLogger(__name__)


# ============================================================ 数据结构

@dataclass
class Candidate:
    """送进重排器的一条候选。"""

    idx: int                  # chunk 在 kb.chunks 里的下标
    text: str
    base_score: float         # 融合后的原始分（未归一）
    norm_score: float = 0.0   # 归一化到 0~1 的原始分，用于与重排分插值
    meta: dict = field(default_factory=dict)   # page/section/type/doc_key/image...
    sources: list[str] = field(default_factory=list)  # 被哪几路召回（vector/fulltext）


@dataclass
class RerankOutcome:
    """重排结果。"""

    idx: int
    rerank_score: float
    final_score: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {"idx": self.idx, "rerank_score": round(self.rerank_score, 4),
                "final_score": round(self.final_score, 4), "reason": self.reason}


# ============================================================ 基类

class BaseReranker:
    """
    重排器基类。

    子类只需实现 `score(query, candidates) -> list[float]`（与 candidates 等长的分数，
    约定返回 0~1），基类负责：归一化插值、排序、trace、异常兜底。
    """

    key: str = "base"
    name: str = "基础重排器"
    desc: str = ""
    # 是否会调用外部服务（前端据此提示"这一档会慢"）
    external: bool = False

    def __init__(self, kb=None):
        self.kb = kb

    # -------------------------------------------------- 子类实现
    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        raise NotImplementedError

    # -------------------------------------------------- 统一入口
    def rerank(self, query: str, candidates: list[Candidate],
               top_k: int | None = None,
               alpha: float | None = None) -> tuple[list[RerankOutcome], dict]:
        t0 = time.perf_counter()
        a = RERANK_ALPHA if alpha is None else alpha
        raw: list[float] = []
        err = ""
        if not candidates:
            return [], {"reranker": self.key, "ms": 0.0, "candidates": 0}
        try:
            raw = self.score(query, candidates)
            if len(raw) != len(candidates):
                raise ValueError(f"{self.key} 返回分数条数 {len(raw)} != 候选数 {len(candidates)}")
        except Exception as exc:  # noqa: BLE001 —— 重排是**可选增强**，失败必须降级而不是让请求 500
            err = f"{type(exc).__name__}: {str(exc)[:120]}"
            logger.warning("重排器 %s 失败，降级为按原始分排序：%s", self.key, err)
            raw = [c.norm_score for c in candidates]

        lo, hi = (min(raw), max(raw)) if raw else (0.0, 0.0)
        span = hi - lo
        norm = [(v - lo) / span if span > 1e-9 else 0.5 for v in raw]

        out: list[RerankOutcome] = []
        for c, r, rn in zip(candidates, raw, norm):
            out.append(RerankOutcome(
                idx=c.idx,
                rerank_score=float(r),
                final_score=float(a * rn + (1.0 - a) * c.norm_score),
                reason=self.explain(c, r),
            ))
        out.sort(key=lambda x: -x.final_score)
        if top_k:
            out = out[:top_k]
        trace = {
            "reranker": self.key,
            "name": self.name,
            "candidates": len(candidates),
            "alpha": a,
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "score_range": [round(lo, 4), round(hi, 4)],
        }
        if err:
            trace["degraded"] = err
        return out, trace

    def explain(self, c: Candidate, score: float) -> str:
        return ""


# ============================================================ ① 不重排

class NoReranker(BaseReranker):
    key = "none"
    name = "不重排（对照）"
    desc = "直接按融合分排序。作为消融对照，用来量化重排带来的净收益。"

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        return [c.norm_score for c in candidates]


# ============================================================ ② TF-IDF 重排器

class TFIDFReranker(BaseReranker):
    """
    基于 TF-IDF 的重排器。

    与「全文检索」里的 BM25 不是同一件事，两者刻意做成互补的：

      | | 全文路（BM25F） | 本重排器（TF-IDF） |
      |---|---|---|
      | 词频饱和 | 有（k1） | 无（线性 tf） |
      | 长度归一 | 有（b） | L2 向量归一 |
      | 多字段 | 有（BM25F 加权合并） | 合并成一整篇 |
      | **词组/邻近度** | 只对显式 `"短语"` 生效 | **对所有查询词自动算** |
      | 词序 | 位置存了但只用于短语 | **bigram**：`军用领域` ≠ `领域军用` |

    真正的差异点是最后两行：BM25 把查询当成**词的集合**（bag of words），
    丢掉了词序和词距；TF-IDF 重排器在这里额外做了两件事来补：
      1) **bigram 特征**：把相邻两词也当成一个特征入向量，
         「军用领域收入」这种连续串会被单独奖励；
      2) **邻近度**：查询词在候选里出现的平均间距越小，得分越高
         （同一个句子里的两个词，比隔了 500 字的两处各出现一次更相关）。

    实现上不用 sklearn：候选只有十几条、词表几千个，
    用 dict 手算稀疏点积比 TfidfVectorizer 快且零依赖。
    全库 IDF 从知识库现算一次并缓存（3000 块，约 0.4s，只在首次重排时付）。
    """

    key = "tfidf"
    name = "TF-IDF 重排器"
    desc = ("对候选重算 TF-IDF 稀疏向量余弦，并叠加 bigram 词序特征与查询词邻近度。"
            "纯本地计算，单次约 5~20ms。")

    _IDF_CACHE: dict[int, tuple[dict[str, float], int]] = {}

    def _idf(self) -> tuple[dict[str, float], int]:
        """
        全库 IDF（按知识库 id 缓存）。

        ★ 优先**复用倒排索引已经算好的 IDF**，而不是自己再遍历一遍 1879 个块分词。
        实测差别：自己重算一遍要 2.9 秒（首次重排时全部计在用户头上），
        复用倒排索引的统计量只要几十毫秒 —— 因为倒排索引构建时本来就已经
        按字段建好了 postings，df 是顺手就能读出来的。这是"两套组件共享同一份
        语料统计"带来的性能红利，也顺带保证了两处的 IDF 定义**完全一致**。
        """
        if self.kb is None:
            return {}, 0
        key = id(self.kb)
        hit = TFIDFReranker._IDF_CACHE.get(key)
        if hit is not None:
            return hit
        n = len(self.kb.chunks)
        try:
            from .fulltext import get_index

            idx = get_index(self.kb)
            idf = {t: idx.idf(t) for t in idx._terms_sorted}
        except Exception as exc:  # noqa: BLE001 —— 倒排索引不可用时退回自算
            logger.warning("复用倒排 IDF 失败，改为自算：%s", exc)
            df: dict[str, int] = {}
            for c in self.kb.chunks:
                for t in set(tokenize(c.get("text") or "")):
                    df[t] = df.get(t, 0) + 1
            idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
        TFIDFReranker._IDF_CACHE[key] = (idf, n)
        return idf, n

    def _vec(self, text: str, idf: dict[str, float]) -> dict[str, float]:
        toks = tokenize(text)
        tf: dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        # bigram：词序特征的载体
        gram = TFIDF_RERANK_NGRAM
        if gram >= 2:
            for k in range(gram - 1, 0, -1):
                for i in range(len(toks) - k):
                    bg = "\u0001".join(toks[i:i + k + 1])
                    tf[bg] = tf.get(bg, 0.0) + 0.5 * k   # 词序越长，权重越低
        vec = {t: (1.0 + math.log(v)) * idf.get(t, 1.0) for t, v in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        idf, _ = self._idf()
        qv = self._vec(query, idf)
        qtoks = [t for t in tokenize(query) if t]
        out: list[float] = []
        for c in candidates:
            dv = self._vec(c.text, idf)
            # 稀疏点积 = 余弦（两边都 L2 归一）
            cos = sum(v * dv.get(t, 0.0) for t, v in qv.items())
            prox = self._proximity(c.text, qtoks)
            out.append(0.8 * cos + 0.2 * prox)
        return out

    @staticmethod
    def _proximity(text: str, qtoks: list[str], window: int = 30) -> float:
        """
        查询词在文本里的邻近度（0~1）：取所有查询词的位置，
        若全部落在一个 window 大小的窗内，得分最高；越分散越低。
        """
        if len(qtoks) < 2:
            return 1.0 if qtoks and qtoks[0] in text else 0.0
        toks = tokenize(text)
        pos: list[int] = []
        for i, t in enumerate(toks):
            if t in set(qtoks):
                pos.append(i)
        if len(pos) < 2:
            return 0.0
        pos.sort()
        best = None
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                span = pos[j] - pos[i]
                if best is None or span < best:
                    best = span
        if best is None:
            return 0.0
        return 1.0 / (1.0 + best / window)

    def explain(self, c: Candidate, score: float) -> str:
        return f"tfidf+bigram+proximity={score:.3f}"


# ============================================================ ③ LLM 重排器

_LLM_RERANK_SYSTEM = """你是检索结果重排器。给定一个问题与若干候选资料片段，\
请判断每个片段对回答该问题的价值，逐条给出 0~10 的整数分：
  10 = 直接包含答案
  7~9 = 包含答案的关键依据（数据、名称、条款）
  4~6 = 主题相关但不足以直接回答
  1~3 = 仅出现个别关键词，无实质帮助
  0 = 完全无关

只输出 JSON：{"scores": [{"id": 编号, "score": 分数, "why": "不超过15字的理由"}, ...]}
不得输出任何其他内容。"""


class LLMReranker(BaseReranker):
    """
    基于 LLM 的重排器。

    它和前面几种的本质区别：**它读得懂语义，而不是数词**。
    招股书里最容易出错的恰恰是「数词算不出来」的那类问题：
      * 「哪个行业的毛利率同比下降最多」—— 需要在多个表格之间做**比较**，
        关键词全都命中，向量也都相似，只有读懂了数字才能排序；
      * 「他参与的哪个工程荣获国家科技进步一等奖」—— 「他」指代谁，得看懂上下文。

    代价也很实在：一次 LLM 往返 ~600~1500ms，且候选越多越慢。
    因此这里做了三件事把开销压在预算内：
      1) **只精排前 N 条**（`RERANK_CANDIDATES`，默认 12）；
      2) **分批**送（`LLM_RERANK_BATCH`，默认 6 条/次，两批并行不了就串行）；
      3) **截断**：每条候选只给前 220 字 —— 招股书的答案句几乎都在块首，
         给全文只会把 token 烧在无关的表格行上。

    任何异常（超时、限流、JSON 解析失败）都不会让请求失败：
    记 trace 里的 `degraded`，并自动退回按原始分排序。
    """

    key = "llm"
    name = "LLM 重排器"
    desc = "把 (问题, 候选) 一起交给大模型逐条打分。语义理解最好，但每次约 0.6~1.5s。"
    external = True

    SNIPPET_CHARS = 220

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        from . import llm

        scores: list[float] = [0.0] * len(candidates)
        batch = max(1, LLM_RERANK_BATCH)
        for start in range(0, len(candidates), batch):
            chunk = candidates[start:start + batch]
            lines = []
            for k, c in enumerate(chunk):
                snippet = (c.text or "").replace("\n", " ")[:self.SNIPPET_CHARS]
                head = f"第{c.meta.get('page', '?')}页"
                if c.meta.get("section"):
                    head += f"（{c.meta['section']}）"
                lines.append(f"[{k}] 来源{head}：{snippet}")
            user = f"【问题】\n{query}\n\n【候选片段】\n" + "\n".join(lines)
            data = llm.chat_json(
                [{"role": "system", "content": _LLM_RERANK_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=LLM_RERANK_MAX_TOKENS,
            )
            got = (data or {}).get("scores") or []
            for rec in got:
                try:
                    k = int(rec.get("id"))
                    s = float(rec.get("score"))
                except (TypeError, ValueError):
                    continue
                if 0 <= k < len(chunk):
                    scores[start + k] = max(0.0, min(10.0, s)) / 10.0
        return scores

    def explain(self, c: Candidate, score: float) -> str:
        return f"llm_score={score:.2f}"


# ============================================================ ④ 用户反馈自适应重排器

class FeedbackReranker(BaseReranker):
    """
    基于用户反馈的自适应重排器。

    「自适应」体现在它是**唯一会跨请求改变自己**的重排器：
    别的重排器每次跑都得到同样的结果，它的权重会随用户点击「这条有用 / 没用」而
    在线更新，并把权重落盘（`data/feedback.jsonl` 记反馈事件，
    `data/feedback_weights.json` 记学到的权重）。

    为什么用「浅层特征 + 线性权重」而不是训一个模型：
      招股书问答的反馈样本量级是几十条，任何高容量模型都会过拟合；
      而「哪一类块更可能是答案」本身有很强的先验（表格块多含数据、
      图像块多含结构图、标题命中的块更可能切题）。
      线性模型每条反馈都能**立刻、可解释地**体现出来 ——
      演示视频里能看到"给表格点两次赞，表格类块的排序就上去了"。

    特征（全部可从 chunk 元数据零成本算出，不引入任何模型调用）：
        keyword_coverage  查询词覆盖率（用倒排索引的分词器）
        type_table        是不是表格块
        type_image        是不是图像块
        section_hit       章节标题里是否命中查询词
        doc_key           问题点名的那份文档（工单3 的多文档消歧信号）

    更新规则是**感知机式**的一步梯度：
        w_i ← w_i + lr · (y - ŷ) · f_i
    其中 y∈{0,1} 是用户反馈，ŷ∈[0,1] 是当前预测。点「有用」把命中特征往上推，
    点「没用」往下拉。权重做 L2 收缩（乘 0.999）防止无界增长。
    """

    key = "feedback"
    name = "用户反馈自适应重排器"
    desc = "用于反馈数据在线调整特征权重（关键词覆盖 / 表图偏好 / 标题命中）。会随使用变好。"
    external = False

    WEIGHTS_PATH = FEEDBACK_PATH.parent / "feedback_weights.json"
    _weights: dict[str, float] | None = None

    # -------------------------------------------------- 权重存取
    @classmethod
    def weights(cls) -> dict[str, float]:
        if cls._weights is None:
            w = dict(FEEDBACK_PRIORS)
            try:
                if cls.WEIGHTS_PATH.exists():
                    w.update({k: float(v) for k, v in
                              json.loads(cls.WEIGHTS_PATH.read_text(encoding="utf-8")).items()
                              if k in FEEDBACK_PRIORS})
            except Exception as exc:  # noqa: BLE001
                logger.warning("反馈权重读取失败，使用先验：%s", exc)
            cls._weights = w
        return cls._weights

    @classmethod
    def save_weights(cls) -> None:
        if cls._weights is None:
            return
        try:
            cls.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            cls.WEIGHTS_PATH.write_text(
                json.dumps(cls._weights, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("反馈权重落盘失败：%s", exc)

    # -------------------------------------------------- 特征
    def features(self, query: str, c: Candidate) -> dict[str, float]:
        terms = {t for t in tokenize(query) if t and len(t) > 1}
        text = c.text or ""
        cover = 0.0
        if terms:
            toks = set(tokenize(text))
            cover = sum(1 for t in terms if t in toks) / len(terms)
        sec = c.meta.get("section") or ""
        return {
            "keyword_coverage": cover,
            "type_table": 1.0 if c.meta.get("type") == "table" else 0.0,
            "type_image": 1.0 if c.meta.get("type") == "image" else 0.0,
            "section_hit": 1.0 if (sec and any(t in sec for t in terms)) else 0.0,
            "doc_key": 0.0,   # 由外部（问句点名文档）在 meta 里置 hint_hit 时补
        }

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        w = self.weights()
        out: list[float] = []
        for c in candidates:
            f = self.features(query, c)
            # 问题点名了某份文档时，检索层会在 meta 里放 _hint 标记
            f["doc_key"] = 1.0 if c.meta.get("hint_hit") else 0.0
            raw = sum(w.get(k, 0.0) * v for k, v in f.items())
            out.append(1.0 / (1.0 + math.exp(-3.0 * (raw - 0.5))))   # 压到 0~1
        return out

    def explain(self, c: Candidate, score: float) -> str:
        return f"feedback(w={{{', '.join(f'{k}:{v:.2f}' for k, v in self.weights().items())}}})"


def record_feedback(query: str, chunk_idx: int, useful: bool, kb=None,
                    extra: dict | None = None) -> dict:
    """
    记录一条用户反馈，并**立即**更新自适应重排器的权重。

    返回更新后的权重与本次更新的增量，便于界面上直接展示"系统学到了什么"。
    反馈事件按 jsonl 追加落盘（一行一条，永不覆盖），权重单独存一份快照。
    """
    from .index_store import KnowledgeBase

    kb = kb or KnowledgeBase.get()
    c = kb.chunks[chunk_idx] if 0 <= chunk_idx < len(kb.chunks) else {}
    cand = Candidate(
        idx=chunk_idx, text=c.get("text", ""), base_score=0.0, norm_score=0.0,
        meta={"section": c.get("section", ""), "type": c.get("type", ""),
              "page": c.get("page", 0), "doc_key": c.get("doc_key", ""),
              "hint_hit": bool((extra or {}).get("hint_hit"))},
    )
    r = FeedbackReranker(kb)
    feats = r.features(query, cand)
    feats["doc_key"] = 1.0 if cand.meta.get("hint_hit") else 0.0
    w = FeedbackReranker.weights()
    raw = sum(w.get(k, 0.0) * v for k, v in feats.items())
    pred = 1.0 / (1.0 + math.exp(-3.0 * (raw - 0.5)))
    y = 1.0 if useful else 0.0
    delta = {k: round(FEEDBACK_LR * (y - pred) * v, 5) for k, v in feats.items()}
    for k, dv in delta.items():
        w[k] = round(w.get(k, 0.0) + dv, 5)
    # L2 收缩：反馈会一直累积，不做收缩权重会缓慢发散
    for k in list(w):
        w[k] = round(w[k] * 0.999, 5)
    FeedbackReranker._weights = w
    FeedbackReranker.save_weights()

    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query, "chunk_idx": chunk_idx, "useful": useful,
        "page": cand.meta.get("page"), "type": cand.meta.get("type"),
        "pred": round(pred, 4), "features": feats,
    }
    try:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("反馈事件落盘失败：%s", exc)

    return {"weights": w, "delta": delta, "pred": round(pred, 4),
            "features": {k: round(v, 4) for k, v in feats.items()},
            "total_feedback": feedback_count()}


def feedback_count() -> int:
    try:
        if FEEDBACK_PATH.exists():
            return sum(1 for ln in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines() if ln.strip())
    except Exception:  # noqa: BLE001
        pass
    return 0


def reset_feedback() -> None:
    """清空反馈（演示前复位用）。反馈数据属于运行期产物，可以直接删。"""
    FeedbackReranker._weights = dict(FEEDBACK_PRIORS)
    FeedbackReranker.save_weights()
    try:
        if FEEDBACK_PATH.exists():
            FEEDBACK_PATH.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("清空反馈失败：%s", exc)


# ============================================================ ⑤ 交叉编码器重排器（可选）

class CrossEncoderReranker(BaseReranker):
    """
    基于交叉编码器（Cross-Encoder）的重排器。

    结构上它是「重排」这件事最正统的实现：把 query 与 doc 拼成一个序列
    `[CLS] query [SEP] doc [SEP]` 送进 Transformer，让两者在每一层都互相 attend，
    最后用一个回归头输出相关性分数。相比之下：
      * 双塔（向量检索）：query、doc 各自独立编码，可预计算 —— 快，但交互为 0；
      * 本类：必须现算 —— 慢，但交互是 token 级的。

    这里用本机已有的 `D:\\models\\bge-reranker-v2-m3`（已转 fp16）。
    默认**关闭**（`CROSS_ENCODER_ENABLED=False`），原因有二：
      1) 加载要占 1~2GB 内存，而本机内存本来就紧张（见 user 环境说明）；
      2) CPU 上 12 对短文本前向约 1.0~1.5s，会明显挤压「响应 ≤3 秒」的预算。
    在 GPU 上跑（`CROSS_ENCODER_DEVICE=cuda`）才建议默认打开。
    权重不存在时，加载失败会被基类捕获 → 自动降级为原始分排序，不影响主链路。
    """

    key = "cross"
    name = "交叉编码器重排器（bge-reranker-v2-m3）"
    desc = "query 与 doc 拼一条序列过一次 Transformer，交互最充分；CPU 上约 1~1.5s。"
    external = False

    _model = None

    @classmethod
    def available(cls) -> tuple[bool, str]:
        p = Path(CROSS_ENCODER_PATH)
        if not p.is_dir():
            return False, f"权重目录不存在：{p}"
        if not any(p.glob("*.safetensors")):
            return False, f"目录里没有 safetensors 权重：{p}"
        return True, str(p)

    @classmethod
    def load(cls):
        if cls._model is None:
            from sentence_transformers import CrossEncoder

            cls._model = CrossEncoder(CROSS_ENCODER_PATH, device=CROSS_ENCODER_DEVICE,
                                      max_length=512)
        return cls._model

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        ok, msg = self.available()
        if not ok:
            raise RuntimeError(f"交叉编码器不可用：{msg}")
        model = self.load()
        pairs = [(query, (c.text or "")[:512]) for c in candidates[:CROSS_ENCODER_MAX_PAIRS]]
        if not pairs:
            return []
        logits = model.predict(pairs, batch_size=4, show_progress_bar=False)
        # bge-reranker 输出的是未过 sigmoid 的 logit，用 sigmoid 归一到 0~1
        vals = [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
        vals.extend([0.0] * (len(candidates) - len(vals)))
        return vals

    def explain(self, c: Candidate, score: float) -> str:
        return f"cross_encoder={score:.3f}"


# ============================================================ 注册表

RERANKERS: dict[str, type[BaseReranker]] = {
    "none": NoReranker,
    "tfidf": TFIDFReranker,
    "llm": LLMReranker,
    "feedback": FeedbackReranker,
    "cross": CrossEncoderReranker,
}


def get_reranker(key: str, kb=None) -> BaseReranker:
    """按 key 取重排器实例；未知 key 退回 tfidf（而不是报错）。"""
    cls = RERANKERS.get((key or "").strip().lower()) or TFIDFReranker
    return cls(kb)


def list_rerankers() -> list[dict]:
    """列出全部重排器及其当前可用性（前端下拉用）。"""
    out = []
    for k, cls in RERANKERS.items():
        avail, note = True, ""
        if k == "cross":
            avail, note = CrossEncoderReranker.available()
        if k == "llm":
            from .config import LLM_API_KEY

            avail = bool(LLM_API_KEY)
            note = "" if avail else "未配置 LLM_API_KEY"
        out.append({
            "key": k, "name": cls.name, "desc": cls.desc,
            "external": cls.external, "available": bool(avail), "note": note,
        })
    return out
