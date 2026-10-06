"""
memory.py — 记忆门面

短期记忆在 session_memory.py，长期记忆在 long_term.py。
本模块只保留同时涉及两者的聚合操作，避免把两份存储的细节混在一起。
"""

from __future__ import annotations

import long_term
import session_memory


def get_all_memory(user_id: str) -> dict:
    """汇总某个用户的记忆，供 GET /memory/{user_id} 接口使用。"""
    return {
        "user_id": user_id,
        "rounds": session_memory.get_rounds(user_id),
        "short_term": session_memory.get_short_term(user_id),
        "long_term": long_term.list_long_term(user_id),
    }


def clear_all(user_id: str) -> None:
    """清空某个用户的全部记忆（短期 + 长期 + 轮次计数）。"""
    session_memory.clear_short_term(user_id)
    session_memory.clear_rounds(user_id)
    long_term.clear_long_term(user_id)


if __name__ == "__main__":
    import config
    import vector_store

    config.LOCAL_MODE = True
    vector_store.reset_store()

    uid = "_selftest_mem"
    clear_all(uid)
    session_memory.save_short_term(uid, "问题", "回答")
    long_term.store_long_term(uid, "用户问题：问题\n回答要点：回答", domain="legal")

    snapshot = get_all_memory(uid)
    print(f"短期 {len(snapshot['short_term'])} 条，长期 {len(snapshot['long_term'])} 条")

    clear_all(uid)
    snapshot = get_all_memory(uid)
    assert not snapshot["short_term"] and not snapshot["long_term"]
    print("memory 自检通过。")
