from typing import Protocol


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 向量客户端统一入口，具体实现负责批量编码和归一化。
        ...
