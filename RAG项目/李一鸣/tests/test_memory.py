from app.core.config import Settings
from app.rag.memory import MemoryService


def test_local_memory_preserves_turn_order():
    memory = MemoryService(Settings(redis_enabled=False, max_memory_messages=4))
    memory.append_turn("u", "r", "c", "你好", "你好，我在。")
    memory.append_turn("u", "r", "c", "继续", "好的。")
    history = memory.get_history("u", "r", "c")
    assert [item["content"] for item in history] == ["你好", "你好，我在。", "继续", "好的。"]
