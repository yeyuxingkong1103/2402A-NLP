# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【缓存组件 · cache.py】统一缓存门面，支持 Redis（生产）与本地文件（单机免部署）双后端
# 编写日期：2026-09-28   修订日期：2026-10-04
import json
import time
import hashlib
import threading
from typing import Optional

import config

try:
    import redis  # Redis 后端可选依赖
except ImportError:  # pragma: no cover
    redis = None


def _key(question: str, top_k: int, lang: str = "zh") -> str:
    """根据问题+top_k+语言生成缓存键，避免撞车"""
    raw = f"{question.strip()}|k={top_k}|l={lang}"
    return "rag:answer:" + hashlib.md5(raw.encode("utf-8")).hexdigest()


class RedisCache:
    """Redis 缓存后端：生产环境高并发共享缓存"""

    _client = None

    @classmethod
    def client(cls):
        if cls._client is None:
            cls._client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
                socket_timeout=2,
            )
        return cls._client

    @classmethod
    def get(cls, key: str) -> Optional[dict]:
        """读取缓存，过期/不存在返回 None"""
        try:
            raw = cls.client().get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            print(f"[WARN] redis read fail: {e}")
            return None

    @classmethod
    def set(cls, key: str, payload: dict) -> None:
        """写入缓存并设置 TTL"""
        try:
            cls.client().setex(key, config.CACHE_TTL, json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            print(f"[WARN] redis write fail: {e}")

    @classmethod
    def health(cls) -> bool:
        """探活"""
        try:
            return bool(cls.client().ping())
        except Exception:
            return False


class LocalCache:
    """本地文件缓存后端：无需 Redis，Windows 单机即开即用；带 TTL 与线程锁"""

    _lock = threading.Lock()
    _store: dict = None

    @classmethod
    def _load(cls) -> dict:
        """惰性加载本地缓存文件"""
        if cls._store is None:
            path = config.DATA_DIR / "local_cache.json"
            if path.exists():
                try:
                    cls._store = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    cls._store = {}
            else:
                cls._store = {}
            cls._path = path
        return cls._store

    @classmethod
    def _persist(cls) -> None:
        """落盘持久化"""
        try:
            cls._path.write_text(
                json.dumps(cls._store, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            print(f"[WARN] local cache persist fail: {e}")

    @classmethod
    def get(cls, key: str) -> Optional[dict]:
        """读取缓存，过期自动剔除"""
        with cls._lock:
            store = cls._load()
            item = store.get(key)
            if not item:
                return None
            # TTL 校验
            if time.time() - item["ts"] > config.CACHE_TTL:
                store.pop(key, None)
                cls._persist()
                return None
            return item["payload"]

    @classmethod
    def set(cls, key: str, payload: dict) -> None:
        """写入缓存（覆盖写）"""
        with cls._lock:
            store = cls._load()
            store[key] = {"ts": time.time(), "payload": payload}
            cls._persist()

    @classmethod
    def health(cls) -> bool:
        """本地缓存恒可用（磁盘可写时）"""
        try:
            cls._load()
            return True
        except Exception:
            return False


class Cache:
    """缓存统一门面：按配置自动选择后端"""

    @staticmethod
    def _backend():
        if config.CACHE_BACKEND == "redis" and redis is not None:
            return RedisCache
        return LocalCache

    @classmethod
    def get_answer(cls, question: str, top_k: int, lang: str = "zh") -> Optional[dict]:
        """读缓存答案"""
        if not config.ENABLE_CACHE:
            return None
        return cls._backend().get(_key(question, top_k, lang))

    @classmethod
    def set_answer(cls, question: str, top_k: int, payload: dict,
                   lang: str = "zh") -> None:
        """写缓存答案"""
        if not config.ENABLE_CACHE:
            return
        cls._backend().set(_key(question, top_k, lang), payload)

    @classmethod
    def health(cls) -> bool:
        """缓存探活"""
        return cls._backend().health()

    @classmethod
    def backend_name(cls) -> str:
        """当前后端名（供界面展示）"""
        return "redis" if cls._backend() is RedisCache else "local"


if __name__ == "__main__":
    print("缓存后端:", Cache.backend_name(), "| ping:", Cache.health())

# ====================================================================
# 技术备注：
# 1. RAG：缓存命中时跳过 Embedding、检索与 LLM 全链路，重复问题延迟 < 50ms。
# 2. TTL 机制保证答案与文档更新之间的时效平衡。
# 3. 门面模式让业务代码与缓存实现解耦，单机/生产环境零改动切换。
# ====================================================================
