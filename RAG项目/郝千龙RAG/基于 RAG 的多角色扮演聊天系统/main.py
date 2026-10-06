# -*- coding: utf-8 -*-
"""【命令行入口 · main.py】CLI 对话界面：加载知识库后在终端与默认角色多轮聊天。"""
from config import DEFAULT_ROLE_CODE
from database import SessionLocal, User, init_db
from rag import TranslationRetriever
from roles import get_role
from services import answer


def main() -> None:
    """CLI 入口：建库 → 加载知识库 → 用 guest 账号进入循环对话。"""
    init_db()
    role = get_role(DEFAULT_ROLE_CODE)
    print(f"正在加载知识库（角色：{role['name']}）…")
    retriever = TranslationRetriever.build()  # 加载/构建 BM25 索引
    with SessionLocal() as db:
        guest = db.query(User).filter_by(username="guest").one()  # 默认演示账号
        user_id = guest.id
    print(f"已加载 {len(retriever.pairs)} 条。输入 q 退出。当前用户 guest。\n")
    session_id = None  # None 表示首轮，之后复用返回的会话 id 实现多轮
    while True:
        query = input("你：").strip()
        if not query or query.lower() in {"q", "quit", "exit"}:
            print("再见。")
            break
        result = answer(user_id, DEFAULT_ROLE_CODE, query, session_id=session_id)
        session_id = result["session_id"]  # 记住会话，后续轮次带上下文
        print(f"{role['name']}：{result['answer']}\n")


if __name__ == "__main__":
    main()
