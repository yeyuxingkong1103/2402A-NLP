from vector_store_milvus import MilvusVectorStore
from rerank import BGERerank
import mysql.connector
import requests

# 初始化
vs = MilvusVectorStore()
rk = BGERerank()

# MySQL配置
mysql_cfg = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123456",
    "database": "rag_character_db"
}

# DeepSeek API
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-6e04da2625084194a695de2b19c13ad2"

def get_role_from_mysql(role_id=1):
    conn = mysql.connector.connect(**mysql_cfg)
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM role WHERE id=%s", (role_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def get_llm_reply(question, contexts, role_info):
    context_text = "\n".join([item["text"] for item in contexts])
    system_prompt = role_info["system_prompt"]
    messages = [
        {"role": "system", "content": f"{system_prompt}\n【知识库参考】{context_text}"},
        {"role": "user", "content": question}
    ]
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }
    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers)
    res_json = resp.json()
    return res_json["choices"][0]["message"]["content"]

if __name__ == "__main__":
    role = get_role_from_mysql(1)
    coll_name = role["milvus_collection"]

    test_set = [
        {
            "question": "Alice周末一般去哪里？",
            "ref_answer": "Alice周末会去美术馆看画展。"
        },
        {
            "question": "Alice喜欢什么食物？",
            "ref_answer": "Alice喜欢草莓蛋糕。"
        },
        {
            "question": "Alice有什么爱好？",
            "ref_answer": "Alice爱好画画，空闲喜欢阅读科幻小说。"
        }
    ]
    total = len(test_set)
    correct = 0

    print("==== 简易RAG评测开始 ====\n")
    for item in test_set:
        q = item["question"]
        ref = item["ref_answer"]
        retrieve_result = vs.search(coll_name, q, top_k=5)
        retrieve_result = rk.rerank(q, retrieve_result, top_k=2)
        ctx = [i["text"] for i in retrieve_result]
        ans = get_llm_reply(q, retrieve_result, role)

        print(f"【问题】{q}")
        print(f"【参考答案】{ref}")
        print(f"【模型输出】{ans}")
        print(f"【检索上下文】{ctx}\n")

        # 简单关键词判断
        if "美术馆" in ans or "草莓蛋糕" in ans or "画画" in ans:
            correct +=1
    print(f"==== 评测结束，正确数：{correct}/{total} ====")
