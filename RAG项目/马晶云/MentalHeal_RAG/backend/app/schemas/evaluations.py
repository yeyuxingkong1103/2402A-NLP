from datetime import datetime

from pydantic import BaseModel, Field


class RagEvaluationCreateRequest(BaseModel):
    dataset_path: str = Field(
        default="data/evaluation/ragas_dataset.json",
        min_length=1,
        max_length=512,
    )


class RagEvaluationResponse(BaseModel):
    run_id: str
    status: str
    dataset_path: str
    sample_count: int
    metrics: dict[str, float | None]
    result: dict
    error_message: str
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class RagEvaluationListResponse(BaseModel):
    runs: list[RagEvaluationResponse]
