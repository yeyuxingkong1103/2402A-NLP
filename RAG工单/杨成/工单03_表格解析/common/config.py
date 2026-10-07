"""环境配置。"""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    rag_data_dir: Path = Path("data")
    rag_top_k: int = 5
    llm_backend: str = "local"

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY") or None
        raw_top_k = os.getenv("RAG_TOP_K", "5")
        try:
            rag_top_k = int(raw_top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("RAG_TOP_K must be a positive integer") from exc
        if rag_top_k <= 0:
            raise ValueError("RAG_TOP_K must be a positive integer")

        return cls(
            openai_api_key=api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            rag_data_dir=Path(os.getenv("RAG_DATA_DIR", "data")),
            rag_top_k=rag_top_k,
            llm_backend="openai" if api_key else "local",
        )
