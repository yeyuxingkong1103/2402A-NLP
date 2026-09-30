from db import verify_user, list_characters
from rag_core import rag_chat, get_vs, clear_history
from pdf_parser import parse_pdf, split_text
from logger import get_logger
from config import REDIS_HOST, REDIS_PORT, REDIS_DB
import redis
import os

logger = get_logger()
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

def login():
    print("==== 用户登录 ====")
    username = input("用户名：").strip()
    password = input("密码：").strip()
    user = verify_user(username, password)
    if user:
        logger.info(f"用户登录成功：{username}")
        print(f"✅ 登录成功，欢迎 {username}")
        return username
    else:
        print("❌ 用户名或密码错误")
        return None

def select_character():
    print("\n==== 选择角色 ====")
    characters = list_characters()
    for c in characters:
        print(f"  {c[0]}. {c[1]} - {c[2]}")
    while True:
        name = input("请输入角色名称：").strip()
        char_names = [c[1] for c in characters]
        if name in char_names:
            logger.info(f"切换角色：{name}")
            print(f"✅ 已切换到角色：{name}")
            return name
        print("❌ 角色不存在，请重新输入")

def add_knowledge(character_name):
    print(f"\n==== 向 {character_name} 知识库添加文档 ====")
    print("输入文档内容（输入空行结束）：")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines)
    if text.strip():
        vs = get_vs(character_name)
        vs.insert_texts([text])
        print(f"✅ 已添加到 {character_name} 知识库")

def import_file(character_name):
    print(f"\n==== 导入文件到 {character_name} 知识库 ====")
    file_path = input("请输入文件路径（支持pdf/txt）：").strip()
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return
    if file_path.lower().endswith(".pdf"):
        blocks = parse_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            blocks = [f.read()]
    else:
        print("❌ 仅支持pdf和txt")
        return
    all_chunks = []
    for block in blocks:
        all_chunks.extend(split_text(block, method="sentence", chunk_size=300))
    vs = get_vs(character_name)
    vs.insert_texts(all_chunks)
    print(f"✅ 文件已导入到 {character_name} 知识库")

def main():
    logger.info("==== 基于RAG的角色扮演系统启动 ====")
    print("==== 基于RAG的角色扮演系统 ====")
    username = None
    while not username:
        username = login()
    character_name = select_character()
    get_vs(character_name)
    print(f"\n==== 当前角色：{character_name} ====")
    print("命令：exit退出 | clear清记忆 | switch切角色 | add加知识 | import导文件")
    while True:
        user_input = input(f"你（{username}）：").strip()
        if user_input.lower() == "exit":
            logger.info(f"用户{username}退出系统")
            print("再见！")
            break
        elif user_input.lower() == "clear":
            session_id = f"{username}_{character_name}"
            clear_history(session_id)
            print("✅ 对话历史已清空")
        elif user_input.lower() == "switch":
            character_name = select_character()
            get_vs(character_name)
            print(f"\n==== 当前角色：{character_name} ====")
        elif user_input.lower() == "add":
            add_knowledge(character_name)
        elif user_input.lower() == "import":
            import_file(character_name)
        elif user_input == "":
            continue
        else:
            logger.info(f"用户提问：{user_input}")
            character, answer, retrieved_docs, elapsed = rag_chat(username, character_name, user_input)
            if character:
                print(f"{character['name']}：{answer}")
            else:
                print(f"❌ {answer}")

if __name__ == "__main__":
    main()
