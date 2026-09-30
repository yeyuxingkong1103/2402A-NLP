from datetime import datetime
from pathlib import Path
import sys
from threading import Thread
from uuid import uuid4

from app.core.config import PROJECT_ROOT, Settings
from app.db.session import get_session_factory
from app.models.chat import RagEvaluationRun


DATASET_PATH = PROJECT_ROOT / "data/evaluation/ragas_dataset.json"


def _load_runner():
    scripts_root = str(PROJECT_ROOT)
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from scripts.evaluation.run_ragas import build_samples, run_ragas

    return build_samples, run_ragas


def create_run(dataset_path: str = "data/evaluation/ragas_dataset.json") -> RagEvaluationRun:
    db = get_session_factory()()
    try:
        run = RagEvaluationRun(
            run_id=uuid4().hex,
            status="queued",
            dataset_path=dataset_path,
            sample_count=0,
            metrics_json={},
            result_json={},
            error_message="",
            created_at=datetime.utcnow(),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()


def execute_run(run_id: str, settings: Settings) -> None:
    _update_run(run_id, status="running", started_at=datetime.utcnow(), error_message="")
    try:
        if not DATASET_PATH.is_file():
            raise FileNotFoundError(f"评测数据集不存在: {DATASET_PATH.name}")
        import json

        payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        if not isinstance(samples, list) or not samples:
            raise ValueError("评测数据集为空或格式无效")
        build_samples, run_ragas = _load_runner()
        rows = build_samples(settings, samples)
        evaluation = run_ragas(rows, settings)
        result = {
            "dataset": "data/evaluation/ragas_dataset.json",
            "sample_count": len(rows),
            "metrics": evaluation["averages"],
            "samples": evaluation["rows"],
        }
        _update_run(
            run_id,
            status="success",
            sample_count=len(rows),
            metrics_json=evaluation["averages"],
            result_json=result,
            finished_at=datetime.utcnow(),
            error_message="",
        )
    except Exception as exc:
        _update_run(
            run_id,
            status="failed",
            error_message=str(exc)[:4000],
            finished_at=datetime.utcnow(),
        )


def start_run(run_id: str, settings: Settings) -> Thread:
    thread = Thread(target=execute_run, args=(run_id, settings), daemon=True)
    thread.start()
    return thread


def _update_run(run_id: str, **values: object) -> None:
    db = get_session_factory()()
    try:
        run = db.query(RagEvaluationRun).filter(RagEvaluationRun.run_id == run_id).first()
        if not run:
            return
        for key, value in values.items():
            setattr(run, key, value)
        db.commit()
    finally:
        db.close()
