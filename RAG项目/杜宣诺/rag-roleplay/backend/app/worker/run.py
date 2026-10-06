from arq.connections import RedisSettings

from ..config import get_settings
from .tasks import extract_memory_task, ingest_document_task


class WorkerSettings:
    functions = [extract_memory_task, ingest_document_task]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
