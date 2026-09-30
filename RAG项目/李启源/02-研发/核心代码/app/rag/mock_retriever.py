"""Mock retriever for testing without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MockChunk:
    """Mock chunk result."""
    chunk_id: str
    text: str
    source: str
    score: float
    rerank_score: float
    metadata: dict[str, Any]


@dataclass
class MockRetrievalResult:
    """Mock retrieval result."""
    results: list[MockChunk]
    total_time: float


class MockHybridRetriever:
    """Mock hybrid retriever that returns predefined results."""

    def __init__(self):
        # 预定义的mock数据
        self.mock_data = [
            {
                "text": "我们的产品支持7天无理由退货，退货时需要保持商品完好并附带原始包装。",
                "metadata": {"source": "退货政策.pdf", "kb_id": 1}
            },
            {
                "text": "配送时间一般为下单后3-5个工作日，偏远地区可能需要7-10个工作日。",
                "metadata": {"source": "配送说明.pdf", "kb_id": 1}
            },
            {
                "text": "如需售后服务，请拨打客服热线400-888-9999或发送邮件至service@example.com。",
                "metadata": {"source": "售后服务.pdf", "kb_id": 1}
            },
            {
                "text": "产品质保期为一年，在质保期内出现非人为损坏可免费维修或更换。",
                "metadata": {"source": "质保条款.pdf", "kb_id": 1}
            },
            {
                "text": "支付方式包括支付宝、微信支付、银行卡支付等多种选择。",
                "metadata": {"source": "支付方式.pdf", "kb_id": 1}
            },
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        final_top_k: int = 5,
        filters: dict | None = None,
        bm25_keywords: list[str] | None = None,
    ) -> MockRetrievalResult:
        """模拟搜索，返回mock数据."""

        # 简单的关键词匹配
        results = []
        for i, item in enumerate(self.mock_data[:top_k]):
            # 计算简单的相似度分数
            score = 0.9 - i * 0.1  # 递减分数

            chunk = MockChunk(
                chunk_id=f"mock_chunk_{i}",
                text=item["text"],
                source=item["metadata"]["source"],
                score=score,
                rerank_score=score,
                metadata=item["metadata"]
            )
            results.append(chunk)

        return MockRetrievalResult(
            results=results,
            total_time=0.1
        )
