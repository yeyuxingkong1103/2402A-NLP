# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：Reranker 精排注册表（工单硬性要求2 + PDF 工单「≥3 种重排算法」）

工单要求
--------
1. 精排必须**真实加载并推理 Reranker 模型**，不允许用规则打分冒充；
2. Reranker 模型 **≥2 种且可切换**（BGE-Reranker-Large/Base 或 Cross-Encoder 系列）；
3. （PDF 工单）提供至少 3 种重排算法：基于 LLM 的重排器、基于 TF-IDF 的重排器、
   基于用户反馈的自适应重排器 —— 本模块全部实现，其中后两者与 LLM 重排器
   作为**用户可选的对比基线**，生产默认 `model`（真实 Reranker 模型）。

注册的 4 种精排算法
-------------------
| 算法键   | 说明                                             | 是否生产默认 |
| -------- | ------------------------------------------------ | ------------ |
| model    | Cross-Encoder Reranker 模型（3 个模型可切换）      | ✅ 是        |
| llm      | Ollama qwen2.5:3b 逐条相关性打分（0~10）          | 对比基线     |
| tfidf    | TF-IDF 余弦相似度重排（规则基线）                  | 对比基线     |
| adaptive | 用户反馈自适应重排（点赞/点踩学习 + 词面相似迁移） | 对比基线     |

可切换的 Reranker 模型（config.RERANKER_MODELS）
------------------------------------------------
  bge-reranker-v2-m3（默认，Large 级，1024 隐层 / 本机 D:\\model\\reranker）
  bge-reranker-v2-m3-ms（同构副本，多路径容灾）
  bge-reranker-base（768 隐层轻量版，随本工程落盘 data/models/）

工程约束（沿用上游踩坑结论）
----------------------------
- `from_pretrained(..., local_files_only=True)`，禁止任何下载；
- 8G 显存机器上以 fp16 + device_map 直载，剩余显存 <2.5G 自动改用 CPU；
- 加载失败进入**冷却期**而非永久熔断（瞬时内存不足不该让进程此后都不精排）；
- 任何精排异常都不打断问答：降级为「按召回顺序截断」并如实记录。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from src import config

# ---------------------------------------------------------------------------
# 模型句柄缓存（按模型名分别缓存 tokenizer/model/device）
# ---------------------------------------------------------------------------
_handles: dict[str, tuple] = {}
_load_failed_until: dict[str, float] = {}
_LOAD_COOLDOWN_S = 120

_lock = threading.RLock()


def _model_spec(name: str) -> dict:
    if name not in config.RERANKER_MODELS:
        raise KeyError(f"未知 Reranker 模型 {name!r}；"
                       f"可选：{list(config.RERANKER_MODELS)}")
    return config.RERANKER_MODELS[name]


def _pick_device() -> str:
    """cuda（显存充足）→ cpu（显存不足自动降级）。

    为什么需要显存闸门（本机实测）：RTX 4060 Laptop 8G 上桌面应用常年占用
    ~3.8G，embedding 再占 ~1.5G；reranker 以 fp16 加载时若剩余显存逼近临界，
    进程会触发段错误（Windows/CUDA 层崩溃，Python 无法捕获）。
    判据 2.5G：低于此值改用 CPU 推理（20 个候选约 1~2 秒），稳定性优先。
    """
    try:
        import torch
        if config.EMBEDDING_DEVICE.startswith("cuda") and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free / (1024 * 1024) < 2500:
                print(f"[reranker] 显存不足（剩余 {free / 1024 / 1024:.0f}MB），"
                      f"改用 CPU 推理以保证稳定", flush=True)
                return "cpu"
            return config.EMBEDDING_DEVICE
    except Exception:
        pass
    return "cpu"


def _load_once(name: str):
    """真正执行一次加载。"""
    path = _model_spec(name)["path"]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"本地精排模型不存在: {path}（模型 {name}）")

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    device = _pick_device()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    kwargs = {"local_files_only": True}
    if device.startswith("cuda"):
        # 关键：按 fp16 直接加载，并用 device_map 直送显存。
        # fp32 先落内存再 .to(cuda) 的峰值 = 内存副本 + 显存副本，
        # 在本机会触发 "OSError 1455 页面文件太小" 甚至段错误。
        kwargs["torch_dtype"] = torch.float16
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                path, device_map=device, **kwargs)
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
            model.to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
        model.to(device)
    model.eval()
    print(f"[rerankers] 已加载 {name}（{device}，fp16="
          f"{device.startswith('cuda')}，{time.time() - t0:.1f}s）", flush=True)
    return tokenizer, model, device


def get_model(name: str | None = None):
    """懒加载指定 Reranker 模型（线程安全，失败冷却重试）。

    返回 (tokenizer, model, device)。
    """
    global _handles
    name = name or config.RERANKER_MODEL_NAME
    if name in _handles:
        return _handles[name]
    with _lock:
        if name in _handles:
            return _handles[name]
        if time.time() < _load_failed_until.get(name, 0.0):
            raise RuntimeError(f"精排模型 {name} 处于失败冷却期，暂不重试")
        try:
            handle = _load_once(name)
        except Exception:
            _load_failed_until[name] = time.time() + _LOAD_COOLDOWN_S
            raise
        _handles[name] = handle
        _load_failed_until[name] = 0.0
        return handle


def unload(name: str | None = None) -> None:
    """卸载精排模型释放显存（切模型/长跑时用）。"""
    with _lock:
        _handles.pop(name or config.RERANKER_MODEL_NAME, None)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_available(name: str | None = None) -> bool:
    """权重目录存在，且不在失败冷却期内。"""
    name = name or config.RERANKER_MODEL_NAME
    try:
        path = _model_spec(name)["path"]
    except KeyError:
        return False
    if not os.path.isdir(path):
        return False
    return time.time() >= _load_failed_until.get(name, 0.0)


# ---------------------------------------------------------------------------
# 算法 1：Reranker 模型精排（Cross-Encoder，生产默认）
# ---------------------------------------------------------------------------
def _sort_with_tiebreak(candidates: Sequence[dict], score_key: str) -> list[dict]:
    """按精排分降序排；分数「打平」时回退到融合名次（RRF）裁决。

    【为什么需要】上游实测：RRF 把"含金额的片段"排在"只含占比的片段"之前（正确），
    而 Cross-Encoder 给两者的分数差 0.0016（噪声量级）却把顺序翻了过来。
    精排分几乎相同时谁排前面本就没有统计意义，此时 RRF 名次聚合了
    稠密+稀疏两路独立信号，更可信。故以 epsilon 分桶，桶内用 RRF 分裁决。
    """
    eps = getattr(config, "RERANKER_TIE_EPSILON", 0.0) or 0.0
    if eps <= 0:
        return sorted(candidates, key=lambda c: c.get(score_key, 0.0), reverse=True)

    def key(c: dict):
        bucket = round(c.get(score_key, 0.0) / eps)
        return (-bucket, -c.get("rrf_score", 0.0), -c.get("vector_score", 0.0))

    return sorted(candidates, key=key)


def _rerank_model(query: str, cands: list[dict], model_name: str) -> list[dict]:
    """Cross-Encoder 逐对 (query, passage) 打分，sigmoid → [0,1] 相关性。"""
    import torch

    tokenizer, model, device = get_model(model_name)
    # 精排用"干净正文"：去掉【标题】前缀，避免前缀词干扰相关性判断
    pairs = [(query, c.get("raw_text") or c.get("text", "")) for c in cands]
    scores: list[float] = []
    bs = config.RERANKER_BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            enc = tokenizer([p[0] for p in batch], [p[1] for p in batch],
                            padding=True, truncation=True,
                            max_length=config.RERANKER_MAX_LEN,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    for c, s in zip(cands, scores):
        c["rerank_score"] = float(s)
    return _sort_with_tiebreak(cands, "rerank_score")


# ---------------------------------------------------------------------------
# 算法 2：基于 LLM 的重排器（对比基线，工单 PDF 要求的三算法之一）
# ---------------------------------------------------------------------------
_LLM_RERANK_PROMPT = (
    "你是检索结果相关性评审员。给定用户问题和一段候选文本，"
    "请只输出一个 0~10 的整数分数（10=完全回答了问题，0=完全无关），"
    "不要输出任何其它内容。\n\n"
    "用户问题：{q}\n候选文本：{p}\n分数："
)


def _llm_score_one(query: str, passage: str, timeout: float) -> float | None:
    from src import llm

    try:
        out = llm.chat(
            [{"role": "user", "content": _LLM_RERANK_PROMPT.format(
                q=query, p=passage[:220])}],
            temperature=0.0, num_predict=4, timeout=timeout)
        txt = (out or "").strip()
        import re
        m = re.search(r"\d+(?:\.\d+)?", txt)
        if not m:
            return None
        return max(0.0, min(10.0, float(m.group(0)))) / 10.0
    except Exception:
        return None


def _rerank_llm(query: str, cands: list[dict], deadline: float) -> list[dict]:
    """LLM 重排：并发逐条打分（控制在 LLM_RERANK_POOL 内），超时/失败保持原序。

    为什么是"逐条点式打分"而不是"列表式排序"：本机 3B 模型在列表式任务上
    容易漏项/改格式（上游同源实测结论）；点式打分是受约束子任务，可靠性高，
    且天然可并发。
    """
    pool = cands[: config.LLM_RERANK_POOL]
    rest = cands[config.LLM_RERANK_POOL:]
    budget = max(1.0, deadline - time.time())
    per = max(2.0, budget / 4)          # 4 路并发下的单条超时预算
    results: dict[int, float | None] = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_llm_score_one, query,
                              c.get("raw_text") or c.get("text", ""), per): i
                    for i, c in enumerate(pool)}
            for fut in as_completed(futs, timeout=budget + 2):
                try:
                    results[futs[fut]] = fut.result()
                except Exception:
                    results[futs[fut]] = None
    except Exception as exc:  # noqa: BLE001 —— 整体超时/异常：保持原序
        print(f"[rerankers] LLM 重排降级（保持召回序）：{exc}", flush=True)
        return cands

    ok = [v for v in results.values() if v is not None]
    if not ok:
        return cands
    for i, c in enumerate(pool):
        c["llm_rerank_score"] = results.get(i)
        c["rerank_score"] = results.get(i) if results.get(i) is not None else 0.0
    ranked = sorted(pool, key=lambda c: c.get("rerank_score") or 0.0, reverse=True)
    return ranked + rest


# ---------------------------------------------------------------------------
# 算法 3：基于 TF-IDF 的重排器（对比基线）
# ---------------------------------------------------------------------------
def _rerank_tfidf(query: str, cands: list[dict]) -> list[dict]:
    """TF-IDF 余弦相似度重排（规则基线，非生产）。"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from src.tokenizer import tokenize_query

        texts = [c.get("text", "") or "" for c in cands]
        vec = TfidfVectorizer(tokenizer=tokenize_query, lowercase=False,
                              token_pattern=None)
        mat = vec.fit_transform(texts + [query])
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
        for c, s in zip(cands, sims):
            c["tfidf_rerank_score"] = float(s)
            c["rerank_score"] = float(s)
        return sorted(cands, key=lambda c: c["rerank_score"], reverse=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[rerankers] TF-IDF 重排降级：{exc}", flush=True)
        return cands


# ---------------------------------------------------------------------------
# 算法 4：基于用户反馈的自适应重排器（对比基线）
# ---------------------------------------------------------------------------
_feedback_lock = threading.RLock()
_feedback_cache: list[dict] | None = None
_feedback_doc_votes: dict[str, float] | None = None


def _feedback_path() -> str:
    return getattr(config, "FEEDBACK_STORE",
                   os.path.join(config.DATA_DIR, "feedback", "feedback.jsonl"))


def load_feedback(force: bool = False) -> list[dict]:
    """读取用户反馈留痕（jsonl，一行一条）。"""
    global _feedback_cache, _feedback_doc_votes
    with _feedback_lock:
        if _feedback_cache is not None and not force:
            return _feedback_cache
        rows: list[dict] = []
        path = _feedback_path()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        votes: dict[str, float] = {}
        for r in rows:
            key = f"{r.get('source', '')}#{r.get('chunk_id', '')}"
            votes[key] = votes.get(key, 0.0) + (1.0 if r.get("helpful") else -1.0)
        _feedback_cache = rows
        _feedback_doc_votes = votes
        return rows


def record_feedback(query: str, hit: dict, helpful: bool,
                    strategy: str = "", reranker: str = "") -> dict:
    """记录一条用户反馈（供自适应重排器学习；界面点赞/点踩调用）。"""
    global _feedback_cache
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "source": hit.get("source", ""),
        "chunk_id": hit.get("chunk_id", ""),
        "page_idx": hit.get("page_idx", 0),
        "block_type": hit.get("block_type", ""),
        "helpful": bool(helpful),
        "strategy": strategy,
        "reranker": reranker,
    }
    path = _feedback_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _feedback_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        _feedback_cache = None          # 下次读取时重载
    return row


def feedback_stats() -> dict:
    rows = load_feedback()
    up = sum(1 for r in rows if r.get("helpful"))
    return {"n": len(rows), "helpful": up, "unhelpful": len(rows) - up,
            "store": _feedback_path()}


def _rerank_adaptive(query: str, cands: list[dict]) -> list[dict]:
    """自适应重排：融合「该块的历史票数」与「历史相似问题的反馈迁移」。

    排序分 = 基础精排序位分（无模型时用 RRF/向量分）+ α·直接票数 + β·相似反馈迁移。
    相似反馈迁移：对历史反馈中的 query 与本 query 做词面 Jaccard 相似度，
    相似度高（≥0.5）时把对其它块的正/负反馈按相似度折扣迁移过来 ——
    这正是"基于用户反馈"相对规则排序的价值：跨问题复用偏好。
    """
    from src.tokenizer import tokenize_query

    rows = load_feedback()
    if not rows:
        return cands
    votes = _feedback_doc_votes or {}
    q_terms = set(tokenize_query(query))

    def base_score(c: dict, idx: int) -> float:
        for k in ("rerank_score", "rrf_score", "vector_score", "fulltext_score"):
            if c.get(k) is not None:
                return float(c[k])
        return 1.0 / (idx + 1)

    # 历史相似问题的反馈迁移表：chunk_key -> 加权票
    transfer: dict[str, float] = {}
    for r in rows:
        hq = r.get("query", "")
        if not hq:
            continue
        hq_terms = set(tokenize_query(hq))
        if not hq_terms:
            continue
        inter = len(q_terms & hq_terms)
        union = len(q_terms | hq_terms) or 1
        sim = inter / union
        if sim >= 0.5:
            key = f"{r.get('source', '')}#{r.get('chunk_id', '')}"
            transfer[key] = transfer.get(key, 0.0) + sim * (1.0 if r.get("helpful") else -1.0)

    alpha, beta = 0.15, 0.10
    for idx, c in enumerate(cands):
        key = f"{c.get('source', '')}#{c.get('chunk_id', '')}"
        direct = votes.get(key, 0.0)
        mig = transfer.get(key, 0.0)
        c["adaptive_boost"] = round(alpha * math.tanh(direct) + beta * math.tanh(mig), 4)
        c["rerank_score"] = base_score(c, idx) + c["adaptive_boost"]
    return sorted(cands, key=lambda c: c.get("rerank_score") or 0.0, reverse=True)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def rerank_detailed(query: str, candidates: Sequence[dict],
                    algorithm: str | None = None,
                    model: str | None = None,
                    top_k: int | None = None,
                    score_threshold: float | None = None) -> dict:
    """精排统一入口。返回 {hits, info}。

    info 记录工单硬性要求2 的三项观测值：
      · 召回阶段候选数 n_in
      · 精排阶段耗时 elapsed_ms
      · 精排前后 Top-K 命中变化 rank_changes（块 → 精排前名次 → 精排后名次）
    """
    algorithm = algorithm or config.RERANK_ALGORITHM_DEFAULT
    model = model or config.RERANKER_MODEL_NAME
    top_k = top_k or config.RERANKER_TOP_K_OUT
    thresh = config.SCORE_THRESHOLD if score_threshold is None else score_threshold

    cands = [dict(c) for c in candidates]
    if not cands:
        return {"hits": [], "info": {"algorithm": algorithm, "n_in": 0,
                                     "elapsed_ms": 0.0, "degraded": False}}

    before_rank = {_key(c): i for i, c in enumerate(cands, start=1)}
    t0 = time.time()
    degraded = False
    err = ""

    if not config.RERANKER_ENABLED:
        ranked, degraded = cands, True
        err = "精排总开关关闭（RERANKER_ENABLED=False）"
    elif algorithm == "model":
        # 生产默认：真实加载 Reranker 模型推理；失败则降级为按召回序截断
        if is_available(model):
            try:
                ranked = _rerank_model(query, cands, model)
            except Exception as exc:  # pragma: no cover - 兜底路径
                ranked, degraded, err = cands, True, f"模型精排失败：{exc}"
                print(f"[rerankers] {err}（本次退化为按召回顺序截断）", flush=True)
        else:
            ranked, degraded = cands, True
            err = f"精排模型不可用（{model} 缺失或冷却中）"
    elif algorithm == "llm":
        ranked = _rerank_llm(query, cands, deadline=t0 + config.RERANK_TIMEOUT_S)
    elif algorithm == "tfidf":
        ranked = _rerank_tfidf(query, cands)
    elif algorithm == "adaptive":
        ranked = _rerank_adaptive(query, cands)
    else:
        ranked, degraded, err = cands, True, f"未知精排算法 {algorithm}"

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    hits = [c for c in ranked if (c.get("rerank_score") is None
                                  or c["rerank_score"] >= thresh)][:top_k]

    changes = []
    for i, c in enumerate(hits, start=1):
        k = _key(c)
        r0 = before_rank.get(k)
        if r0 is not None and r0 != i:
            changes.append({"chunk": k, "before": r0, "after": i,
                            "delta": r0 - i})

    info = {
        "algorithm": algorithm,
        "model": model if algorithm == "model" else algorithm,
        "model_path": (_model_spec(model)["path"] if algorithm == "model"
                       and model in config.RERANKER_MODELS else ""),
        "n_in": len(cands),
        "n_out": len(hits),
        "elapsed_ms": elapsed_ms,
        "score_threshold": thresh,
        "rank_changes": changes,
        "n_changed": len(changes),
        "degraded": degraded,
        "degrade_reason": err,
    }
    return {"hits": hits, "info": info}


def rerank(query: str, candidates: Sequence[dict],
           top_k: int | None = None) -> list[dict]:
    """兼容上游调用点：默认算法精排并返回 top_k。"""
    return rerank_detailed(query, candidates, top_k=top_k)["hits"]


def _key(c: dict) -> str:
    return f"{c.get('source', '')}#{c.get('chunk_id', '')}"


def info() -> dict:
    """当前精排配置与状态，供界面/文档展示。"""
    name = config.RERANKER_MODEL_NAME
    cooling = max(0.0, _load_failed_until.get(name, 0.0) - time.time())
    return {
        "enabled": config.RERANKER_ENABLED and is_available(name),
        "algorithm": config.RERANK_ALGORITHM_DEFAULT,
        "algorithms": list(config.RERANK_ALGORITHMS),
        "model": name,
        "models": {k: {"path": v["path"], "desc": v["desc"],
                       "exists": os.path.isdir(v["path"])}
                   for k, v in config.RERANKER_MODELS.items()},
        "loaded_models": list(_handles),
        "top_k_in": config.RERANKER_TOP_K_IN,
        "top_k_out": config.RERANKER_TOP_K_OUT,
        "max_len": config.RERANKER_MAX_LEN,
        "cooldown_s": round(cooling, 1),
    }
