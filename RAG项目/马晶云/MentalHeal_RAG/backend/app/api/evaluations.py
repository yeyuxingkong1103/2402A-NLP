from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import PROJECT_ROOT, get_settings
from app.db.session import get_db_session
from app.models.chat import RagEvaluationRun, User
from app.schemas.evaluations import (
    RagEvaluationCreateRequest,
    RagEvaluationListResponse,
    RagEvaluationResponse,
)
from app.security.auth import get_current_user
from app.services.ragas_evaluation import create_run, start_run

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])


def require_admin(user: User) -> None:
    if user.account_role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")


def serialize_run(run: RagEvaluationRun) -> RagEvaluationResponse:
    return RagEvaluationResponse(
        run_id=run.run_id,
        status=run.status,
        dataset_path=run.dataset_path,
        sample_count=run.sample_count,
        metrics=run.metrics_json or {},
        result=run.result_json or {},
        error_message=run.error_message or "",
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
    )


@router.post(
    "/ragas",
    response_model=RagEvaluationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_ragas_evaluation(
    request: RagEvaluationCreateRequest,
    current_user: User = Depends(get_current_user),
) -> RagEvaluationResponse:
    require_admin(current_user)
    if request.dataset_path != "data/evaluation/ragas_dataset.json":
        raise HTTPException(status_code=400, detail="当前仅允许使用项目标准评测集")
    if not (PROJECT_ROOT / request.dataset_path).is_file():
        raise HTTPException(status_code=404, detail="评测数据集不存在")
    run = create_run(request.dataset_path)
    start_run(run.run_id, get_settings())
    return serialize_run(run)


@router.get("/ragas", response_model=RagEvaluationListResponse)
def list_ragas_evaluations(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> RagEvaluationListResponse:
    require_admin(current_user)
    runs = db.scalars(
        select(RagEvaluationRun).order_by(RagEvaluationRun.created_at.desc()).limit(50)
    ).all()
    return RagEvaluationListResponse(runs=[serialize_run(run) for run in runs])


@router.get("/ragas/{run_id}", response_model=RagEvaluationResponse)
def get_ragas_evaluation(
    run_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> RagEvaluationResponse:
    require_admin(current_user)
    run = db.scalar(select(RagEvaluationRun).where(RagEvaluationRun.run_id == run_id))
    if not run:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return serialize_run(run)
