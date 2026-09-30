"""重排序（精排）抽象层。

- ScoreFusionReranker：MVP 默认，直接按检索阶段 RRF 融合分数排序（轻量、无模型）。
- BGEReranker：BGE-rerank 交叉编码精排，调用本地 rerank 服务（/v1/rerank，硅基流动兼容）。
"""
from __future__ import annotations

import httpx

from .config import Settings
from .logging_config import get_logger

log = get_logger("reranker")


class Reranker:
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


class ScoreFusionReranker(Reranker):
    """MVP 精排：按融合分数排序。

    RRF 已在检索阶段完成稠密/BM25 的排名融合；此处为扩展点，可叠加业务规则、
    加权 dense/bm25 分数，或引入文本质量特征。
    """

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)


class BGEReranker(Reranker):
    """BGE-rerank 交叉编码精排：调本地 rerank 服务（硅基流动兼容）。

    服务不可用时降级回融合分数排序，保证链路可用。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def _get_client(self) -> httpx.Client:
        headers = {}
        if self.settings.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.settings.rerank_api_key}"
        return httpx.Client(
            base_url=self.settings.rerank_base_url, headers=headers, timeout=60.0
        )

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        texts = [c.get("text", "") for c in candidates]
        try:
            # 每次请求 `with` 建/关 client：连接不跨请求复用（本地服务开销可忽略），
            # 换来的是退出时连接池一定被释放，不留悬空的 httpx 连接。
            with self._get_client() as client:
                resp = client.post(
                    "/rerank",
                    json={
                        "model": self.settings.rerank_model,
                        "query": query,
                        "documents": texts,
                        # 显式要全量分数。不传时由服务端的默认 top_n 决定返回几条——
                        # 有的服务商默认只回前 N 条，剩下的候选就没有重排分数了（见下面
                        # 未打分分支）。本仓 rerank_service 默认返回全部，传了也无副作用。
                        "top_n": len(texts),
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                scores = self._parse_scores(payload, len(candidates))
        except Exception as exc:  # noqa: BLE001
            log.warning("BGE-rerank 服务不可用（%s），降级为融合分数排序: %s", self.settings.rerank_base_url, exc)
            return ScoreFusionReranker().rerank(query, candidates)

        if not scores:
            # 200 + 错误信封（无 results 键）以前会静默走到下面的"全量退回 RRF"分支：
            # 服务端明明没重排，调用方与前端拿到的却还是"已精排"的分数，且一条日志都没有。
            # 兜底行为不变（链路可用优先），但必须留下痕迹。
            log.warning(
                "BGE-rerank 返回 200 但拿不到任何分数（响应键：%s），本次未精排，"
                "退回融合分数排序——下游看到的仍是 RRF 原分数",
                self._payload_keys(payload),
            )
            return ScoreFusionReranker().rerank(query, candidates)

        # BGE 的 relevance_score（0~1）与 RRF 融合分（≈0.01~0.13）**不是同一量纲**，不能一起
        # 排序：服务端只回部分 index 时（top_n 截断 / 丢项），未打分的候选带着 RRF 分可能
        # 排到"精排器明确打了 0 分"的候选前面，把精排结论整个盖掉。
        # 所以分两段：已打分的按精排分降序；未打分的**整体排在已打分之后**，内部保持
        # 融合分数的原顺序。平手时靠 sort 的稳定性保持原顺序，不额外挑。
        scored = [(i, c, scores[i]) for i, c in enumerate(candidates) if i in scores]
        unscored = [c for i, c in enumerate(candidates) if i not in scores]
        if unscored:
            log.warning(
                "BGE-rerank 只回了 %d/%d 条分数（服务端截断或丢项？），未打分的 %d 条"
                "按融合分数排在已打分结果之后——两段分数不同量纲，不混排",
                len(scored), len(candidates), len(unscored),
            )
        scored.sort(key=lambda t: t[2], reverse=True)
        # 不改原地：candidates 是调用方的列表，就地覆盖 score 会让上游拿到被改写的分数
        return [{**c, "score": s} for _, c, s in scored] + unscored

    @staticmethod
    def _parse_scores(payload: object, n_candidates: int) -> dict[int, float]:
        """响应 -> {候选下标: 精排分}。形状不对的条目直接跳过（由调用方按"未打分"处理）。"""
        if not isinstance(payload, dict):
            return {}
        raw = payload.get("results")
        if not isinstance(raw, list):
            return {}
        scores: dict[int, float] = {}
        for r in raw:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r["index"])
                val = float(r["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue
            # 越界下标直接丢：宁可当它没打分，也不能拿别人的分数去覆盖候选
            if 0 <= idx < n_candidates:
                scores[idx] = val
        return scores

    @staticmethod
    def _payload_keys(payload: object) -> str:
        if isinstance(payload, dict):
            return ",".join(sorted(payload)[:8]) or "(空对象)"
        return type(payload).__name__


def create_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "bge":
        log.info("重排序使用 BGE-rerank（%s）", settings.rerank_base_url)
        return BGEReranker(settings)
    log.info("重排序使用 ScoreFusion（RRF 融合分数）")
    return ScoreFusionReranker()
