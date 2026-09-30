import httpx

from ....config import get_settings


class BgeRerankHttp:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or get_settings().rerank_base_url).rstrip("/")

    async def rerank(self, query: str, passages: list[str], top_m: int) -> list[tuple[int, float]]:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{self.base_url}/rerank", json={"query": query, "passages": passages, "top_m": top_m})
            r.raise_for_status()
            data = r.json()
        return [(res["index"], res["score"]) for res in data["results"]]
