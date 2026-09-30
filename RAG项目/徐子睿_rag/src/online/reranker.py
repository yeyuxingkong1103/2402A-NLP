"""src/online/reranker.py —— BGE 精排（cross-encoder 重排序）。

在链路中的位置：
    src/online/chain.py → 【本文件】 → 本地 BGE Reranker 模型

与 backend/retrieval.py 里那个"让 qwen2:7b 打分"的精排相比，这里的做法不同：
    backend   用生成式大模型给候选打 0-5 分（成本高、需要模型、输出要解析）
    本文件   用专门的 cross-encoder 精排模型（FlagReranker），直接输出相关性分数

两者原理上的共同点：都是"把查询和候选放在一起看"（cross-encoder），
比单纯比较两个独立向量（bi-encoder，即检索阶段的做法）准得多 ——
代价是慢，所以只能用在少量候选上，必须先粗排再精排。

同样的 fail-open 设计：模型不可用时回退到 RRF 顺序，不让精排失败中断整个链路。
"""
from __future__ import annotations

from typing import Any

from configs.settings import get_settings


class Reranker:
    """BGE cross-encoder 精排器（单例使用，见文件末尾）。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None  # 惰性加载：不精排就不占内存/显存

    def _load(self) -> Any:
        """惰性加载 FlagReranker 模型。

        返回：
            FlagReranker 实例。

        use_fp16 与 device 联动：
            非 CPU 设备才开半精度 —— CPU 不支持 fp16 运算，
            不判断的话在无 GPU 机器上会直接报错而不是自动降级。
        """
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.settings.rerank_model, use_fp16=self.settings.rerank_device != "cpu")
        return self._model

    def rerank(self, query: str, hits: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
        """对召回结果重排序。

        参数：
            query: 检索用的查询（应传改写后的 rewritten_query）
            hits: 召回结果列表
            limit: 保留条数，不传取配置默认值
        返回：
            按 rerank_score 降序排列的前 limit 条，每项多一个 rerank_score 字段。

        三层保护：
            1. hits 为空 -> 直接返回空，不做无意义的模型调用
            2. 模型分数不是列表 -> 包成单元素列表
               （单条 pair 时某些版本的 FlagReranker 返回标量而非列表）
            3. 抛任何异常 -> 回退到按 rrf_score 排序
               "精排失败不能中断整条链路"是本项目的一贯取舍：
               宁可这一次排序质量降级，也不让用户拿到一个错误页

        回退排序用 rrf_score 而不是原序：
            hits 进入这里时已经过 RRF 融合排序，按 rrf_score 降序正好还原那个顺序。
        """
        limit = limit or self.settings.rerank_top_k
        if not hits:
            return []
        try:
            # cross-encoder 的输入是 (查询, 候选文本) 成对列表，逐对打分
            pairs = [[query, _content(hit)] for hit in hits]
            scores = self._load().compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
            ranked = [hit | {"rerank_score": float(score)} for hit, score in zip(hits, scores)]
            return sorted(ranked, key=lambda item: -item["rerank_score"])[:limit]
        except Exception:
            # fail-open：模型不可用（没下载/内存不足）时退回 RRF 顺序
            return sorted(hits, key=lambda item: -float(item.get("rrf_score", 0)))[:limit]


def _content(hit: dict[str, Any]) -> str:
    """从召回记录里取出正文文本。

    参数：
        hit: 一条召回结果
    返回：
        正文文本；都取不到时返回空串。

    `hit.get("entity") or hit` 兼容两种返回结构：
        Milvus search 的结果把业务字段包在 entity 里，
        而本模块保持通用性 —— 若传入的已经是摊平的结构，就直接用 hit 本身。

    content / text 两个名字都试：
        新架构集合的字段叫 content（见 src/offline/milvus_store.py），
        backend 那条主线的字段叫 text。两者共用这个精排器时都能工作。
    """
    entity = hit.get("entity") or hit
    return str(entity.get("content") or entity.get("text") or "")


# 模块级单例：cross-encoder 模型加载很贵，全进程只应加载一次
reranker = Reranker()
