from vector_store_milvus import MilvusVectorStore
from db import get_character
from config import DEEPSEEK_API_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL, REDIS_HOST, REDIS_PORT, REDIS_DB, HISTORY_MAX_TURNS
import redis
import requests
import json
import re
import time

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
vs_cache = {}

PROMPT_TEMPLATE = """
{system_prompt}
请严格参考下面知识库内容和历史对话回答用户问题，不要编造不存在信息。
直接回答内容，不要重复角色名称前缀。

【历史对话】
{history}

【知识库】
{context}

用户问题：{question}
"""

def get_vs(character_name):
    if character_name in vs_cache:
        return vs_cache[character_name]
    char_info = get_character(character_name)
    if not char_info:
        return None
    collection_name = char_info[4]
    vs = MilvusVectorStore(collection_name)
    vs.create_collection()
    vs_cache[character_name] = vs
    return vs

def llm_call(messages):
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.7}
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "抱歉，大模型响应超时，请稍后再试。"
    except requests.exceptions.RequestException as e:
        return f"抱歉，大模型调用失败：{str(e)}"
    except (KeyError, IndexError, json.JSONDecodeError):
        return "抱歉，大模型返回格式异常，请稍后再试。"

def llm_stream(messages):
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.7, "stream": True}
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as e:
        yield f"抱歉，流式调用失败：{str(e)}"

def post_process(answer, character_name):
    answer = re.sub(r'^(' + re.escape(character_name) + r'[：:]\s*)+', '', answer)
    return answer.strip()

def query_rewrite(user_query):
    rewrite_map = {
        "多大": "年龄 多少岁", "叫什么": "名字 名称 叫做",
        "喜欢什么": "爱好 喜好 喜欢", "不喜欢": "讨厌 厌恶 不喜欢",
        "是谁": "身份 介绍",
    }
    for key, value in rewrite_map.items():
        if key in user_query:
            return user_query + " " + value
    return user_query

def get_history(session_id):
    try:
        key = f"chat_history:{session_id}"
        history = redis_client.lrange(key, 0, -1)[-(HISTORY_MAX_TURNS * 2):]
        return [json.loads(msg) for msg in history]
    except Exception:
        return []

def save_history(session_id, role, content):
    try:
        key = f"chat_history:{session_id}"
        redis_client.rpush(key, json.dumps({"role": role, "content": content}))
        redis_client.ltrim(key, -(HISTORY_MAX_TURNS * 2), -1)
    except Exception:
        pass

def clear_history(session_id):
    try:
        redis_client.delete(f"chat_history:{session_id}")
    except Exception:
        pass

def get_full_history(session_id):
    """获取完整对话历史，返回列表"""
    try:
        key = f"chat_history:{session_id}"
        history = redis_client.lrange(key, 0, -1)
        return [json.loads(msg) for msg in history]
    except Exception:
        return []

def export_history(session_id, character_name, format="markdown"):
    """导出对话历史为文本"""
    history = get_full_history(session_id)
    if not history:
        return "暂无对话记录"
    if format == "markdown":
        lines = [f"# 对话记录 - {character_name}", ""]
        for msg in history:
            role = "用户" if msg["role"] == "user" else character_name
            lines.append(f"**{role}**：{msg['content']}")
            lines.append("")
        return "\n".join(lines)
    else:
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else character_name
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

def build_prompt(character, user_query, session_id):
    vs = get_vs(character["name"])
    if not vs:
        return None, []
    expanded_query = query_rewrite(user_query)
    retrieved_docs = vs.search_with_rerank(expanded_query, recall_top_k=4, rerank_top_k=2)
    context = "\n".join(retrieved_docs)
    history_messages = get_history(session_id)
    history_str = ""
    for msg in history_messages:
        if msg["role"] == "user":
            history_str += f"用户：{msg['content']}\n"
        else:
            history_str += f"{character['name']}：{msg['content']}\n"
    prompt = PROMPT_TEMPLATE.format(
        system_prompt=character["system_prompt"],
        history=history_str, context=context, question=user_query
    )
    return prompt, retrieved_docs

def rag_chat(username, character_name, user_query):
    start_time = time.time()
    try:
        char_info = get_character(character_name)
        if not char_info:
            return None, "角色不存在", [], 0
        character = {"name": char_info[1], "system_prompt": char_info[3], "collection_name": char_info[4]}
        session_id = f"{username}_{character_name}"
        prompt, retrieved_docs = build_prompt(character, user_query, session_id)
        if prompt is None:
            return None, "向量库初始化失败", [], 0
        answer = llm_call([{"role": "user", "content": prompt}])
        answer = post_process(answer, character["name"])
        save_history(session_id, "user", user_query)
        save_history(session_id, "assistant", answer)
        elapsed = time.time() - start_time
        return character, answer, retrieved_docs, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return None, f"对话处理异常：{str(e)}", [], elapsed

def rag_chat_stream(username, character_name, user_query):
    try:
        char_info = get_character(character_name)
        if not char_info:
            yield "error", "角色不存在"
            return
        character = {"name": char_info[1], "system_prompt": char_info[3], "collection_name": char_info[4]}
        session_id = f"{username}_{character_name}"
        prompt, retrieved_docs = build_prompt(character, user_query, session_id)
        if prompt is None:
            yield "error", "向量库初始化失败"
            return
        yield "retrieved", json.dumps(retrieved_docs, ensure_ascii=False)
        full_answer = ""
        for chunk in llm_stream([{"role": "user", "content": prompt}]):
            full_answer += chunk
            yield "token", chunk
        full_answer = post_process(full_answer, character["name"])
        save_history(session_id, "user", user_query)
        save_history(session_id, "assistant", full_answer)
        yield "done", full_answer
    except Exception as e:
        yield "error", str(e)
