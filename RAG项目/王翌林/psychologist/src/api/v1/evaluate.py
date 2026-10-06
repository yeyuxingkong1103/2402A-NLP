"""评测接口：RAGAS 指标评测。

RAGAS 是一套专门评估 RAG 系统质量的指标体系（Faithfulness 忠实度、
Relevancy 相关性、Precision/Recall 检索精确率/召回率）。评测是离线、耗时的
后台任务，因此这里限制为管理员权限，并直接把参数转交给 eval_service 执行。
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_current_admin
from src.db.mysql import get_db
from src.core.exceptions import ok
from src.models import User
from src.schemas import RagasEvalRequest
from src.services import eval_service

router = APIRouter(prefix="/eval", tags=["评测"])


@router.post("/ragas", summary="RAGAS 评测（Faithfulness/Relevancy/Precision/Recall）")
def ragas(payload: RagasEvalRequest, admin: User = Depends(get_current_admin),
          db: Session = Depends(get_db)):
    # 只做转发：RagasEvalRequest 携带 persona_id / limit / dataset_path 三个参数，
    # 真正的评测逻辑（生成问答、跑指标计算）都在 eval_service.run_eval 里。
    return ok(eval_service.run_eval(payload.persona_id, payload.limit, payload.dataset_path))