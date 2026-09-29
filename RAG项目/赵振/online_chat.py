import re
import redis
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from sentence_transformers import CrossEncoder
from offline_ingest import collection_name

MEMORY_NAME = "chat_memory"#长期聊天记忆所在的集合
RERANKER_PATH = r"C:\Users\ZhuanZ\.cache\modelscope\hub\models\AI-ModelScope\bge-reranker-v2-m3"
ROLES = {
    "1": "你是一名严谨的法律资料检索助手，回答详细，并指出相关条文。",
    "2": "你是一名法学老师，用通俗语言解释刑法概念，并举简单例子。",
}#给模型准备了两种身份回答问题

PROMPT = """{role}
只能根据下面的法律资料回答，不知道就说不知道，不要编造条文。
最近聊天：{recent}
长期记忆：{older}
法律资料：{context}
问题：{question}
回答："""#提示词模版

SERVICES = None

def get_services():
    global SERVICES#表示修改SERVICES
    if SERVICES:#判断是否有之前保存的服务
        return SERVICES
    cache = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    client = MilvusClient(uri="http://127.0.0.1:19530")
    if not client.has_collection(collection_name):
        raise RuntimeError("没有知识库")
    embedding = OllamaEmbeddings(model="bge-m3:567m", client_kwargs={"timeout": 180})
    if not client.has_collection(MEMORY_NAME):
        client.create_collection(collection_name=MEMORY_NAME,
                                 dimension=len(embedding.embed_query("测试")),
                                 auto_id=True, enable_dynamic_field=True)
    llm = ChatOllama(model="deepseek-r1:7b", temperature=0,client_kwargs={"timeout": 240})
    reranker = CrossEncoder(RERANKER_PATH)
    SERVICES = cache, client, embedding, llm, reranker
    return SERVICES

def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chat_events(question, user_id, role="1", session_id=""):
    if role not in ROLES or not question.strip():
        raise ValueError("问题或角色不正确")

    yield {"type": "stage", "text": "正在连接知识库并加载重排序模型"}
    cache, client, embedding, llm, reranker = get_services()
    key = f"chat:{user_id}:{role}:{session_id}"
    recent = cache.lrange(key, -6, -1)
    query = " ".join(recent[-2:] + [question])

    yield {"type": "stage", "text": "正在检索相关法律资料"}
    vector = embedding.embed_query(query)
    requests = [
        AnnSearchRequest(data=[vector], anns_field="vector", param={"metric_type": "COSINE"}, limit=5),
        AnnSearchRequest(data=[question], anns_field="sparse", param={"metric_type": "BM25"}, limit=5),
    ]
    hits = client.hybrid_search(collection_name=collection_name, reqs=requests,
                                ranker=RRFRanker(), limit=5,
                                output_fields=["text", "source", "location", "page"])[0]
    if not hits: raise RuntimeError("没有检索到法律资料")
    scores = reranker.predict([[question, hit["entity"]["text"]] for hit in hits])
    hits = [hit for score, hit in sorted(zip(scores, hits), reverse=True)[:3]]

    sources = [hit["entity"] for hit in hits]
    context = "\n\n".join(f"[{item.get('source', '')} · {item.get('location', '')}] {item['text']}" for item in sources)
    memories = client.search(collection_name=MEMORY_NAME, data=[vector],
                             anns_field="vector", limit=2,
                             filter=f'user_id == "{user_id}" and role == "{role}"',
                             output_fields=["text"])[0]
    older = "\n".join(hit["entity"]["text"] for hit in memories)

    yield {"type": "sources", "sources": sources}
    yield {"type": "stage", "text": "正在生成回答"}
    prompt = PROMPT.format(role=ROLES[role], recent="\n".join(recent),
                           older=older or "无", context=context, question=question)
    answer = clean(llm.invoke(prompt).content)
    if not answer: raise RuntimeError("模型没有返回回答")

    cache.rpush(key, f"用户：{question}", f"助手：{answer}")
    cache.ltrim(key, -6, -1)
    cache.expire(key, 86400)
    client.insert(collection_name=MEMORY_NAME, data=[{
        "user_id": user_id, "role": role,
        "text": f"用户：{question}\n助手：{answer}"[:1500], "vector": vector
    }])
    yield {"type": "answer", "answer": answer, "sources": sources}
