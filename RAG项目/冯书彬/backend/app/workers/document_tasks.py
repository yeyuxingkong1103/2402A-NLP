import logging

from backend.app.ingestion.index_writer import index_materials
from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def index_material_payloads(material_payloads: list[dict]) -> int:
    # Celery 任务入口接收纯 JSON payload，转换为领域对象后复用同步索引函数。
    materials = [KnowledgeMaterial(**payload) for payload in material_payloads]
    return index_materials(materials)


if celery_app is not None:
    @celery_app.task(name="documents.index_materials")
    def index_materials_task(material_payloads: list[dict]) -> int:
        logger.info("收到知识库索引任务", extra={"material_count": len(material_payloads)})
        return index_material_payloads(material_payloads)
else:
    def index_materials_task(material_payloads: list[dict]) -> int:
        # 本地未安装 Celery 时保留可测试同步入口，生产由 requirements 启用真实 worker。
        logger.info("同步执行知识库索引任务", extra={"material_count": len(material_payloads)})
        return index_material_payloads(material_payloads)
