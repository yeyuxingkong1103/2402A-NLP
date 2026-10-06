"""Redis 客户端管理：创建并复用 Redis 客户端，供认证与短期记忆共用。"""

from redis import Redis, ConnectionPool

# 全局连接池，进程内复用
_pool: ConnectionPool | None = None
_client: Redis | None = None


def create_redis_client(redis_url: str) -> Redis:
    """创建 Redis 客户端。

    统一设置 decode_responses=True，避免调用方各处手动解码；
    使用连接池，进程内复用同一个客户端实例。

    参数：
    - redis_url: Redis 连接串，如 "redis://127.0.0.1:6379/0"

    返回：
    - Redis 客户端实例
    """
    global _pool, _client

    if _client is None:
        _pool = ConnectionPool.from_url(redis_url, decode_responses=True)
        _client = Redis(connection_pool=_pool)

    return _client


def get_redis_client() -> Redis:
    """获取全局 Redis 客户端单例。

    必须先调用 create_redis_client() 初始化，否则抛出 RuntimeError。
    测试代码可以通过注入替身绕过此检查。
    """
    if _client is None:
        raise RuntimeError("Redis 客户端未初始化，请先调用 create_redis_client()")
    return _client
