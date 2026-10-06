from openai import AsyncOpenAI


class EmbeddingClient:
    def __init__(self, *, model: str, base_url: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
