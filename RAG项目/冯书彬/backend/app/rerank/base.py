from typing import Protocol


class RerankClient(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        # Reranker 统一入口，返回顺序必须与 documents 输入顺序一致。
        ...
