from functools import lru_cache

from .config import get_settings
from .core.fakes import FakeEmbedding, FakeRerank
from .core.providers.embedding.bge_m3_http import BgeM3Http
from .core.providers.llm.openai_compat import OpenAICompatLLM
from .core.providers.rerank.bge_rerank_http import BgeRerankHttp
from .store.milvus_store import MilvusStore
from .store.redis_store import RedisStore


@lru_cache
def get_llm() -> OpenAICompatLLM:
    return OpenAICompatLLM()


@lru_cache
def get_embedding():
    s = get_settings()
    if s.embedding_base_url:
        return BgeM3Http()
    return FakeEmbedding()


@lru_cache
def get_rerank():
    s = get_settings()
    if s.rerank_base_url:
        return BgeRerankHttp()
    return FakeRerank()


@lru_cache
def get_milvus() -> MilvusStore:
    return MilvusStore(get_embedding())


@lru_cache
def get_redis() -> RedisStore:
    return RedisStore()
