from functools import lru_cache

from sentence_transformers import CrossEncoder

from backend.app.core.config import get_settings


settings = get_settings()


class RerankerService:
    """负责使用 bge-reranker-large 对候选依据重新排序。"""

    def __init__(self) -> None:
        # 加载本地 CrossEncoder 重排模型。
        self.model = CrossEncoder(settings.reranker_model_path, device=settings.model_device, local_files_only=True)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        # 如果没有候选内容，直接返回空列表。
        if not candidates:
            return []
        # CrossEncoder 的输入是“问题 + 候选文本”的成对列表。
        pairs = [(query, item["content"]) for item in candidates]
        # predict 会输出每个候选和问题的相关性分数。
        scores = self.model.predict(pairs, batch_size=settings.reranker_batch_size)
        # 把分数写回候选对象。
        for item, score in zip(candidates, scores):
            item["score"] = float(score)
        # 按重排分数从高到低排序。
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[: settings.rerank_top_k]




@lru_cache
def get_reranker_service() -> RerankerService:
    return RerankerService()
