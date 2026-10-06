"""短期记忆模块（Redis）"""
import json
from typing import List, Dict, Any, Optional
import redis
from redis.exceptions import RedisError

from src.config import config
from src.utils.logger import logger
from src.utils.tools import generate_id


class ShortTermMemory:
    """短期记忆 - 使用Redis"""
    
    def __init__(self):
        redis_config = config.get_redis_config()
        self.host = redis_config.get('host', 'localhost')
        self.port = redis_config.get('port', 6379)
        self.db = redis_config.get('db', 0)
        self.password = redis_config.get('password')
        self.session_expire = redis_config.get('session_expire', 3600)
        
        self.client = None
        self.connected = False
    
    def connect(self):
        """连接Redis"""
        if self.connected:
            return
        
        try:
            logger.info(f"连接Redis: {self.host}:{self.port}")
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # 测试连接
            self.client.ping()
            self.connected = True
            logger.info("Redis连接成功")
        except RedisError as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    def _get_session_key(self, session_id: str) -> str:
        """获取会话键"""
        return f"session:{session_id}"
    
    def _get_history_key(self, session_id: str) -> str:
        """获取历史记录键"""
        return f"history:{session_id}"
    
    def save_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """保存消息"""
        if not self.connected:
            self.connect()
        
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        # 保存到列表
        key = self._get_history_key(session_id)
        self.client.rpush(key, json.dumps(message, ensure_ascii=False))
        
        # 设置过期时间
        self.client.expire(key, self.session_expire)
        
        logger.debug(f"保存消息到会话 {session_id}")
    
    def get_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if not self.connected:
            self.connect()
        
        key = self._get_history_key(session_id)
        messages = self.client.lrange(key, -limit, -1)
        
        history = []
        for msg_str in messages:
            try:
                history.append(json.loads(msg_str))
            except json.JSONDecodeError:
                continue
        
        return history
    
    def clear_history(self, session_id: str):
        """清除对话历史"""
        if not self.connected:
            self.connect()
        
        key = self._get_history_key(session_id)
        self.client.delete(key)
        logger.info(f"清除会话 {session_id} 的历史")
    
    def save_context(
        self,
        session_id: str,
        key: str,
        value: Any
    ):
        """保存上下文数据"""
        if not self.connected:
            self.connect()
        
        context_key = f"context:{session_id}:{key}"
        self.client.set(context_key, json.dumps(value), ex=self.session_expire)
    
    def get_context(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """获取上下文数据"""
        if not self.connected:
            self.connect()
        
        context_key = f"context:{session_id}:{key}"
        value = self.client.get(context_key)
        
        if value is None:
            return default
        
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        if not self.connected:
            self.connect()
        
        keys = self.client.keys("history:*")
        return [k.replace("history:", "") for k in keys]


# 全局实例
short_term_memory = ShortTermMemory()
