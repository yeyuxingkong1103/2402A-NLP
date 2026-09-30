import redis
from dotenv import load_dotenv
import os
from typing import List, Dict

load_dotenv()

class RedisChatHistory:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            db=int(os.getenv("REDIS_DB")),
            decode_responses=True
        )

    def add_msg(self, session_id: str, role: str, content: str):
        """role: user / assistant"""
        key = f"chat:{session_id}"
        msg = {"role": role, "content": content}
        self.redis_client.rpush(key, str(msg))
        # 设置会话过期 24小时，单位秒
        self.redis_client.expire(key, 86400)

    def get_history(self, session_id: str) -> List[Dict]:
        key = f"chat:{session_id}"
        raw_list = self.redis_client.lrange(key, 0, -1)
        history = []
        for item in raw_list:
            # 简易解析
            d = eval(item)
            history.append(d)
        return history

    def clear_history(self, session_id: str):
        key = f"chat:{session_id}"
        self.redis_client.delete(key)

# 测试入口
if __name__ == "__main__":
    chat_store = RedisChatHistory()
    sid = "session_001"
    chat_store.add_msg(sid, "user", "你是谁？")
    chat_store.add_msg(sid, "assistant", "我是自定义角色")
    his = chat_store.get_history(sid)
    print("✅ 获取对话历史：", his)
    # chat_store.clear_history(sid)
