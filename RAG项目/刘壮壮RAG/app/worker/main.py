from arq.connections import RedisSettings
from arq.worker import Worker

from app.config import get_settings
from app.worker.tasks import build_vector_store, extract_memory, process_file


async def startup(ctx):
    build_vector_store().ensure_collections()


async def shutdown(ctx):
    pass


if __name__ == "__main__":
    settings = get_settings()
    worker = Worker(
        functions=[process_file, extract_memory],
        redis_settings=RedisSettings(
            host=settings.redis_host, port=settings.redis_port
        ),
        on_startup=startup,
        on_shutdown=shutdown,
    )
    worker.run()
