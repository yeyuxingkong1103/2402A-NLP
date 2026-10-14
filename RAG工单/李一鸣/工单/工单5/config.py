"""Multi-provider LLM configuration for the revised exercise."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    base_url: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 2048


MODELS = {
    "deepseek-chat": ModelConfig(
        "deepseek-chat",
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        os.getenv("DEEPSEEK_API_KEY", ""),
    ),
    "deepseek-coder": ModelConfig(
        "deepseek-coder",
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        os.getenv("DEEPSEEK_API_KEY", ""),
    ),
}
