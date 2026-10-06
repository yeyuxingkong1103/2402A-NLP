import base64

import httpx


class VisionEmbeddingClient:
    def __init__(self, *, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def embed_image(self, image_path: str) -> list[float]:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(
                "/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": b64},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
