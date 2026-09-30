"""
Memory Module - 多轮对话与记忆系统
阶段5核心模块
"""

from .memory_manager import MemoryManager, MemoryContext
from .session_manager import SessionManager
from .role_manager import RoleManager, RoleConfig
from .redis_memory import RedisMemory
from .milvus_memory import MilvusMemory
from .extractors import InformationExtractor

__all__ = [
    "MemoryManager",
    "MemoryContext",
    "SessionManager",
    "RoleManager",
    "RoleConfig",
    "RedisMemory",
    "MilvusMemory",
    "InformationExtractor",
]
