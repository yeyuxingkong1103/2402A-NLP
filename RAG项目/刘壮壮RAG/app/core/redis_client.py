import redis.asyncio as aioredis

from app.config import get_settings


def get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.Redis(
        host=settings.redis_host, port=settings.redis_port, decode_responses=True
    )
