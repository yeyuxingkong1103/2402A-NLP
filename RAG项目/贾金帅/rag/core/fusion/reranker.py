# -*- coding: utf-8 -*-
"""重排器：云端 SiliconFlow 优先 + 本地交叉编码兜底。

对融合后的 Top-N 结果按「查询-文档」相关性重排序，提升召回精度。
优先调用 SiliconFlow /v1/rerank 接口；云端不可达/超时/报错时回退到本地
sentence-transformers 交叉编码器；两者都失败则跳过重排、返回原序。
任何路径都有超时兜底，保证在线链路不被拖死。 
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import threading
from typing import Optional

from src import config
from src.core.retrieval.base.retriever_base import RetrievalResult

logger = logging.getLogger(__name__)

# 单次重排的候选数上限（避免大模型/云端请求过大）；可配，默认 20
_MAX_RERANK_ITEMS = int(os.getenv("RERANKER_MAX_ITEMS", "20"))
# 本地线程池超时（秒）；云端 HTTP 超时默认值见 _RERANK_CLOUD_TIMEOUT
_RERANK_TIMEOUT_SECONDS = 6.0
# 云端重排 HTTP 超时（秒）：可配，默认 3s；超时快速降级，避免可选步骤拖慢主链路
_RERANK_CLOUD_TIMEOUT = float(os.getenv("RERANKER_CLOUD_TIMEOUT", "3.0"))
# 本地兜底是否开启（默认关）：云端失败时避免懒加载 2.2GB 本地模型造成尖峰
_RERANK_LOCAL_FALLBACK = os.getenv("RERANKER_LOCAL_FALLBACK", "false").lower() \
    in {"1", "true", "yes", "on"}
# 本地兜底线程池（单 worker，串行执行避免 CPU 模型并发打满）
_LOCAL_THREADS = 1

# SiliconFlow 云端重排配置（经环境变量读取，避免硬编码密钥；不写入 config.py）
_SILICONFLOW_RERANK_URL = os.getenv(
    "SILICONFLOW_RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
_SILICONFLOW_RERANK_MODEL = os.getenv(
    "SILICONFLOW_RERANK_MODEL", "Pro/BAAI/bge-reranker-v2-m3")


def _lazy_import_sentence_transformers():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder
    except ImportError:
        return None


class Reranker:
    """重排器：云端 API 优先，本地交叉编码兜底。

    Parameters
    ----------
    model_name : str | None
        本地兜底模型名/路径，None 用 config.RERANKER_MODEL_NAME。
    top_k : int | None
        重排后保留的条数，None 用 config.RERANKER_TOP_K。
    enabled : bool | None
        是否启用；None 用 config.RERANKER_ENABLED。
    timeout : float | None
        云端 HTTP / 本地线程池超时，None 用 _RERANK_TIMEOUT_SECONDS。
    api_url / api_key / api_model : str | None
        云端重排配置，None 用默认环境变量值。
    """

    name = "reranker"

    def __init__(self, model_name: Optional[str] = None,
                 top_k: Optional[int] = None,
                 enabled: Optional[bool] = None,
                 timeout: Optional[float] = None,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_model: Optional[str] = None):
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.top_k = top_k if top_k is not None else config.RERANKER_TOP_K
        self.enabled = (
            enabled if enabled is not None else config.RERANKER_ENABLED)
        # 云端 HTTP 超时：可配，默认 3s（避免可选步骤拖慢主链路）。
        self.timeout = (
            float(timeout) if timeout is not None else _RERANK_CLOUD_TIMEOUT)
        # 本地推理超时：保持宽松（CPU 模型可能稍慢），默认 6s。
        self.local_timeout = _RERANK_TIMEOUT_SECONDS
        self.api_url = api_url or _SILICONFLOW_RERANK_URL
        self.api_key = api_key if api_key is not None else _SILICONFLOW_API_KEY
        self.api_model = api_model or _SILICONFLOW_RERANK_MODEL

        self._model = None  # 本地兜底模型（懒加载）
        self._error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._predict_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

        if self.enabled and not self.api_key:
            # 未配置云端密钥时加载本地兜底模型（本地为主路径）。
            # 若配置了云端 key，本地模型在云端失败时懒加载，避免启动加载 2.2GB。
            self._load()

    def close(self) -> None:
        """释放本地模型与线程池资源。"""
        if self._predict_pool is not None:
            with self._pool_lock:
                if self._predict_pool is not None:
                    self._predict_pool.shutdown(wait=False)
                    self._predict_pool = None
        self._model = None
        self._error = None

    # ------------------------------------------------------------------
    # 本地模型加载（兜底）
    # ------------------------------------------------------------------
    def _load(self) -> None:
        CrossEncoder = _lazy_import_sentence_transformers()
        if CrossEncoder is None:
            self._error = "sentence-transformers 未安装，本地兜底不可用"
            return
        try:
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            self._error = f"重排模型加载失败: {exc}"
            logger.warning("%s，本地兜底将不可用。", self._error)

    def _ensure_local_loaded(self) -> None:
        """云端失败时懒加载本地模型（线程安全，仅加载一次）。"""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None and self._error is None:
                self._load()

    def _get_predict_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._predict_pool is None:
            with self._pool_lock:
                if self._predict_pool is None:
                    self._predict_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=_LOCAL_THREADS)
        return self._predict_pool

    def is_available(self) -> bool:
        # 未启用时不触发任何加载；配置了云端 key 时直接视为可用（本地懒加载）。
        if not self.enabled:
            return False
        if self.api_key:
            return True
        if self._model is None and self._error is None:
            with self._load_lock:
                if self._model is None and self._error is None:
                    self._load()
        return self._model is not None

    # ------------------------------------------------------------------
    # 云端 / 本地打分
    # ------------------------------------------------------------------
    def _rerank_cloud(self, query: str,
                      documents: list[str]) -> list[float]:
        """调用 SiliconFlow /v1/rerank，返回与 documents 对齐的相关度列表。"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.api_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        response = requests.post(
            self.api_url, json=payload, headers=headers,
            timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        items = data.get("results", []) or []
        if not isinstance(items, list):
            raise ValueError("云端重排返回格式不正确")
        scores = [0.0] * len(documents)
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(scores):
                scores[idx] = float(
                    item.get("relevance_score", item.get("score", 0.0)))
        return scores

    def _rerank_local(self, pairs: list[tuple[str, str]]) -> list[float]:
        """本地 CrossEncoder 打分（线程池 + 超时保护）。"""
        pool = self._get_predict_pool()
        future = pool.submit(self._model.predict, pairs)
        try:
            scores = future.result(timeout=self.local_timeout)
        except concurrent.futures.TimeoutError as exc:
            logger.warning("本地重排超时（%.1fs）：%s", self.local_timeout, exc)
            raise
        return [float(score) for score in scores]

    # ------------------------------------------------------------------
    # 重排接口
    # ------------------------------------------------------------------
    def rerank(self, query: str,
               results: list[RetrievalResult]) -> list[RetrievalResult]:
        """按「查询-文档」相关度对结果重排，返回前 top_k 条。

        云端优先；失败回退本地；本地失败跳过重排返回原序。
        """
        if not results:
            return []
        if not self.is_available() or not query:
            return results[:self.top_k] or results

        candidates = results[:_MAX_RERANK_ITEMS]
        documents = [self._result_text(r) for r in candidates]
        scores: Optional[list[float]] = None

        # 1) 云端重排
        if self.api_key:
            try:
                scores = self._rerank_cloud(query, documents)
            except Exception as exc:
                logger.warning("云端重排失败: %s，尝试本地兜底。", exc)

        # 2) 本地兜底重排：仅在「无云端 key（本地为主路径）」或显式开启
        #    RERANKER_LOCAL_FALLBACK 时才懒加载本地模型，避免云端失败后
        #    每次加载 2.2GB 模型造成尖峰。
        if scores is None:
            should_try_local = (not self.api_key) or _RERANK_LOCAL_FALLBACK
            if should_try_local:
                try:
                    self._ensure_local_loaded()
                    if self._model is not None:
                        pairs = [(query, doc) for doc in documents]
                        scores = self._rerank_local(pairs)
                except Exception as exc:
                    logger.warning("本地重排失败: %s，跳过重排。", exc)
            else:
                logger.info(
                    "云端重排不可用且未启用本地兜底（RERANKER_LOCAL_FALLBACK），跳过重排。"
                )

        # 3) 都失败：返回原序（跳过重排）
        if scores is None:
            return results[:self.top_k] or results

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda t: float(t[1]), reverse=True)
        reranked = []
        for rank, (result, score) in enumerate(scored[:self.top_k], start=1):
            # 非破坏性：不修改入参 result，避免污染下游（引擎结果/融合结果复用）。
            reranked.append(dataclasses.replace(
                result,
                score=round(float(score), 6),
                rank=rank,
                reason=f"{result.reason or '重排'}|相关度{score:.3f}",
            ))
        return reranked

    @staticmethod
    def _result_text(result: RetrievalResult) -> str:
        """把检索结果转成重排用的文档文本（药名 + 结构化字段 + 文档正文/章节）。"""
        c = result.content
        parts = [str(c.get("name", ""))]
        for key in ("indication", "category", "otc", "drug_class",
                    "related_diseases", "usage", "adverse",
                    "contraindication", "interaction", "dosage",
                    "specification", "manufacturer"):
            val = c.get(key)
            if val:
                parts.append(str(val))
        # 向量 chunk 正文（vector 结果的关键内容在 document/text）
        doc = c.get("document") or c.get("text")
        if doc:
            parts.append(str(doc))
        section_path = c.get("section_path")
        if section_path:
            parts.append(
                " > ".join(str(i) for i in section_path)
                if isinstance(section_path, list) else str(section_path))
        return " ".join(p for p in parts if p)
