from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.app.config import AppSettings
from backend.app.embeddings import BgeM3Embedder
from backend.app.mineru import MinerUParser
from backend.app.pipeline import BuildPipeline
from backend.app.storage import JsonStateStore
from backend.app.mineru_vlm import MinerUVLMPostProcessor
from backend.app.vision import OpenAIVisionClient
from backend.app.vector_store import QdrantVectorStore


router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def get_task_store() -> JsonStateStore:
    """获取任务存储。"""
    return JsonStateStore(Path("data/state.json"))


def build_pipeline() -> BuildPipeline:
    """构造任务流水线。"""
    settings = AppSettings()
    store = JsonStateStore(Path("data/state.json"))
    parser = MinerUParser()
    embedder = BgeM3Embedder(settings.bge_m3_model_path)
    vector_store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection)
    post_processor = None
    if settings.vlm_enabled:
        vision_client = OpenAIVisionClient(
            settings.vlm_base_url,
            settings.vlm_api_key,
            settings.vlm_model,
            settings.vlm_timeout,
        )
        post_processor = MinerUVLMPostProcessor(vision_client)
    return BuildPipeline(store, parser, embedder, vector_store, post_processor=post_processor)


@router.post("/{task_id}/start")
def start_task(task_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    """启动构建任务。"""
    store = get_task_store()
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    background_tasks.add_task(build_pipeline().run, task_id)
    return {"task_id": task_id, "status": "scheduled"}


@router.get("/{task_id}")
def get_task(task_id: str) -> dict:
    """查询构建任务。"""
    task = get_task_store().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.model_dump(mode="json")
