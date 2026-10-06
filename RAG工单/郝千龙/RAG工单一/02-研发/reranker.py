# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【重排组件 · reranker.py】bge-reranker 交叉编码器对召回候选精排，带超时自动降级
# 编写日期：2026-10-04
import time
from typing import List, Dict

from sentence_transformers import CrossEncoder

import config


class Reranker:
    """CrossEncoder 重排器单例：输入 (问题, 候选) 对，输出相关性精排"""

    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def __init__(self):
        if self._model is None and config.ENABLE_RERANK:
            print(f"[INFO] 加载 Rerank 模型: {config.RERANK_MODEL_NAME} on {config.EMBED_DEVICE}")
            self._model = CrossEncoder(
                config.RERANK_MODEL_NAME, device=config.EMBED_DEVICE, max_length=512
            )

    def rerank(self, question: str, candidates: List[Dict]) -> List[Dict]:
        """
        对候选片段按与问题的相关性重新打分排序
        candidates: [{text,page,doc_id,chunk_id,score}, ...]
        返回同样结构但按 rerank 分降序；超时/异常时原样返回（降级）
        """
        if not config.ENABLE_RERANK or self._model is None or not candidates:
            return candidates

        t0 = time.time()
        try:
            pairs = [(question, c["text"]) for c in candidates]
            raw_scores = self._model.predict(
                pairs, batch_size=config.RERANK_BATCH_SIZE
            )
            elapsed = (time.time() - t0) * 1000
            # 超时保护：超过阈值认为重排不可靠，降级使用原顺序
            if elapsed > config.RERANK_TIMEOUT_MS:
                print(f"[WARN] rerank 耗时 {elapsed:.0f}ms 超阈值，降级原序")
                return candidates
            for c, s in zip(candidates, raw_scores):
                c["rerank_score"] = float(s)
            ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            return ranked
        except Exception as e:
            print(f"[WARN] rerank 失败，降级原序: {e}")
            return candidates


if __name__ == "__main__":
    r = Reranker()
    cands = [
        {"text": "公司注册资本为5,520万元", "page": 22, "doc_id": "p1", "chunk_id": "a", "score": 0.8},
        {"text": "今天天气不错", "page": 1, "doc_id": "p1", "chunk_id": "b", "score": 0.7},
    ]
    for c in r.rerank("注册资本是多少？", cands):
        print(round(c.get("rerank_score", 0), 3), c["text"])

# ====================================================================
# 技术备注：
# 1. Transformer：CrossEncoder 将 (query, passage) 拼接后一次性输入 Transformer，
#    经全交叉注意力建模深层交互，精度高于双塔向量相似度，但计算更贵，
#    因此只对召回的少量候选（Top-20）精排。
# 2. RAG：重排提升 context_precision，把真正含答案的片段顶到前面。
# 3. Fine-tuning：reranker 同样可用领域标注对做交叉编码器微调。
# 4. 超时降级保证任何情况下端到端 SLA 不被重排拖垮（优化方案风险表）。
# ====================================================================
