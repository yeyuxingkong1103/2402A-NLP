from typing import AsyncIterator


class FakeLLM:
    async def chat(self, messages: list[dict], **opts) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"echo: {last}"

    async def chat_stream(self, messages: list[dict], **opts) -> AsyncIterator[str]:
        text = await self.chat(messages, **opts)
        for ch in text:
            yield ch


class FakeEmbedding:
    async def encode_dense(self, texts: list[str]) -> list[list[float]]:
        # 用文本长度生成确定性的 1024 维向量，便于测试断言
        return [[float(len(t)) % 7.0] * 1024 for t in texts]

    async def encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [{0: float(len(t))} for t in texts]

    async def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        return (await self.encode_dense([text]))[0], (await self.encode_sparse([text]))[0]


class FakeRerank:
    async def rerank(self, query: str, passages: list[str], top_m: int) -> list[tuple[int, float]]:
        # 简单的确定性重排：按原文长度降序
        ranked = sorted(range(len(passages)), key=lambda i: -len(passages[i]))
        return [(i, float(len(passages)) - idx) for idx, i in enumerate(ranked[:top_m])]
