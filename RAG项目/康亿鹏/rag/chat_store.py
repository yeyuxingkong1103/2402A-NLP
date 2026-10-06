"""对话历史存储：用 Redis 按用户+领域保存多轮对话，自动过期。"""  # 模块说明
import json  # JSON 序列化历史消息
from typing import List, Dict, Optional  # 类型注解

import redis  # Redis 客户端

import config  # 全局配置


def get_redis() -> redis.Redis:  # 工厂函数：创建 Redis 连接
    """返回 Redis 客户端（每次新建，连接开销小且避免线程问题）。"""
    return redis.Redis(  # 连接本地 Redis
        host=config.redis_host,  # 地址
        port=config.redis_port,  # 端口
        db=config.redis_db,  # 数据库编号
        decode_responses=True,  # 返回字符串而非字节
    )


def history_key(user_id, domain: Optional[str]) -> str:  # 按用户+领域生成 Redis 键名
    uid = user_id if user_id else "default"  # 无 user_id 时用 "default"（命令行模式）
    dom = domain if domain else "general"  # 无 domain 时用 "general"
    return f"rag:history:{uid}:{dom}"  # 例如 rag:history:1:medical


def get_history(user_id, domain: Optional[str]) -> List[Dict]:  # 读取某用户某领域的对话历史
    """返回历史消息列表（[{'role':'user','content':'...'}, ...]），无则空列表。"""
    data = get_redis().get(history_key(user_id, domain))  # 从 Redis 取原始字符串
    if not data:  # 没有历史
        return []
    return json.loads(data)  # 反序列化为消息列表


def add_message(user_id, domain: Optional[str], role: str, content: str) -> None:  # 追加一条消息到历史
    """把一条消息（user 或 assistant）追加到历史，截断到最近 N 轮并重设过期时间。"""
    history = get_history(user_id, domain)  # 先取出已有历史
    history.append({"role": role, "content": content})  # 追加新消息
    max_msgs = config.history_turns * 2  # N 轮 = N 问 + N 答 = 2N 条消息
    history = history[-max_msgs:]  # 只保留最近 2N 条
    get_redis().setex(  # 写回 Redis 并设置过期时间（TTL）
        history_key(user_id, domain),  # 键
        config.history_ttl,  # 过期秒数，到期自动删除
        json.dumps(history, ensure_ascii=False),  # 序列化为 JSON（保留中文）
    )


def format_history(user_id, domain: Optional[str]) -> str:  # 把历史消息格式化成提示词文本
    """把历史拼成 'user: xxx\nassistant: xxx' 形式的字符串，用于注入 prompt。"""
    history = get_history(user_id, domain)  # 取出历史
    if not history:  # 没有历史
        return ""
    lines = []  # 每行一条消息
    for msg in history:
        lines.append(f"{msg['role']}: {msg['content']}")  # 直接用英文角色名拼接
    return "\n".join(lines)  # 用换行连接所有消息


def clear_history(user_id, domain: Optional[str]) -> None:  # 清空某用户某领域的对话历史
    """删除指定用户和领域的历史。"""
    get_redis().delete(history_key(user_id, domain))  # 直接删键
