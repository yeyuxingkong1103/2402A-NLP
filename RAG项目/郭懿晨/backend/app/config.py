from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用配置。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    data_dir: Path = Path("data")
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "rag_documents"
    bge_m3_model_path: Path = Path(r"D:\八维学习\bge-m3")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"
    vlm_enabled: bool = False
    vlm_base_url: str = "http://localhost:11434"
    vlm_api_key: str = ""
    vlm_model: str = "qwen2.5-vl"
    vlm_timeout: int = Field(default=120, gt=0)
    max_upload_mb: int = Field(default=100, gt=0)
    rerank_enabled: bool = True
    reranker_model_path: Path = Path(r"D:\八维学习\bge-reranker-v2-m3")
    reranker_device: str | None = None  # None → 自动检测 CUDA
    rerank_batch_size: int = Field(default=8, gt=0)
    retrieval_candidate_k: int = Field(default=30, gt=0)
    coarse_top_k: int = Field(default=10, gt=0)
    final_top_k: int = Field(default=4, gt=0)
    coarse_scorer: str = "dense_cosine"  # dense_cosine | rrf_score | hybrid
    coarse_hybrid_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    min_retrieval_score: float = Field(default=0.2, ge=0.0)


def ensure_project_path(project_root: Path, target_path: Path) -> Path:
    """确认目标路径位于项目目录内。"""
    resolved_root = project_root.resolve()
    resolved_target = target_path.resolve()

    if resolved_root == resolved_target or resolved_root in resolved_target.parents:
        return resolved_target

    raise ValueError(f"路径位于项目目录之外: {resolved_target}")
