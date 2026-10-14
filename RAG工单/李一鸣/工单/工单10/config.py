from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    models: tuple[str, ...] = ("bge-large:latest", "nomic-embed-text:latest", "m3e-large:latest")
    ollama_base_url: str = "http://127.0.0.1:11434"
    top_k: int = 3


config = EmbeddingConfig()
