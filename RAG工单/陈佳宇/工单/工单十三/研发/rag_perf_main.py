# -*- coding: utf-8 -*-
# rag_perf_main.py 工单13 RAG全链路性能埋点主程序
import time
import uuid
import json
import os

# ========== 优化开关 可切换True/False对比优化前后 ==========
ENABLE_CACHE = False        # 查询结果缓存
ENABLE_RERANK = True        # 是否开启rerank重排
CONTEXT_MAX_TOKENS = 1200   # 上下文最大token限制
LLM_MAX_NEW_TOKENS = 512    # LLM最大输出token

LOG_DIR = "./perf_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 简易缓存字典
query_cache = {}

def get_perf_timestamp():
    return time.perf_counter()

def query_process(query:str):
    """阶段1 查询处理与增强，模拟query改写"""
    start = get_perf_timestamp()
    processed_query = query.strip()
    time.sleep(0.01) #模拟处理耗时
    cost_ms = (get_perf_timestamp() - start)*1000
    return processed_query, cost_ms

def embedding_query(text:str):
    """阶段2‑1 query向量化，模拟embedding推理"""
    start = get_perf_timestamp()
    time.sleep(0.15) #模拟CPU embedding耗时
    cost_ms = (get_perf_timestamp() - start)*1000
    vec = [0.1]*512
    return vec, cost_ms

def vector_search(vector):
    """阶段2‑2 向量检索，模拟向量库召回文档"""
    start = get_perf_timestamp()
    time.sleep(0.08)
    docs = [{"content":f"文档{i}内容"} for i in range(5)]
    cost_ms = (get_perf_timestamp() - start)*1000
    return docs, cost_ms

def rerank_docs(query, docs):
    """阶段2‑3 rerank重排序"""
    start = get_perf_timestamp()
    time.sleep(0.25) #模拟rerank耗时
    cost_ms = (get_perf_timestamp() - start)*1000
    return docs, cost_ms

def build_prompt(query, docs):
    """阶段3 上下文组装与提示工程"""
    start = get_perf_timestamp()
    context_text = "\n".join([d["content"] for d in docs])
    prompt = f"基于下面上下文回答问题：\n{context_text}\n问题:{query}"
    time.sleep(0.02)
    cost_ms = (get_perf_timestamp() - start)*1000
    return prompt, cost_ms

def llm_generate(prompt:str):
    """阶段4 LLM生成回答，模拟大模型调用"""
    start = get_perf_timestamp()
    time.sleep(1.2) #模拟LLM推理耗时，RAG主要瓶颈
    answer = "这是模拟的RAG回答结果"
    cost_ms = (get_perf_timestamp() - start)*1000
    return answer, cost_ms

def post_process(answer):
    """阶段5 后处理格式化"""
    start = get_perf_timestamp()
    final_ans = answer.strip()
    time.sleep(0.01)
    cost_ms = (get_perf_timestamp() - start)*1000
    return final_ans, cost_ms

def rag_full_pipeline(user_query:str):
    """完整RAG链路，全链路埋点计时"""
    req_id = str(uuid.uuid4())
    log_item = {"request_id":req_id,"query":user_query}
    total_start = get_perf_timestamp()

    # 缓存命中判断
    if ENABLE_CACHE and user_query in query_cache:
        ans = query_cache[user_query]
        total_cost = (get_perf_timestamp()-total_start)*1000
        log_item.update({"cache_hit":True,"total_ms":total_cost})
        with open(f"{LOG_DIR}/perf_log.jsonl","a",encoding="utf‑8") as f:
            f.write(json.dumps(log_item,ensure_ascii=False)+"\n")
        return ans, log_item

    # 阶段1 查询处理
    q_processed, t1 = query_process(user_query)
    log_item["stage1_query_process_ms"] = t1

    # 阶段2‑1 embedding
    vec, t2 = embedding_query(q_processed)
    log_item["stage2_embedding_ms"] = t2

    # 阶段2‑2向量检索
    docs, t3 = vector_search(vec)
    log_item["stage2_vector_search_ms"] = t3

    # 阶段2‑3 rerank
    t4 = 0
    if ENABLE_RERANK:
        docs, t4 = rerank_docs(q_processed, docs)
    log_item["stage2_rerank_ms"] = t4

    # 阶段3 prompt组装
    prompt, t5 = build_prompt(q_processed, docs)
    log_item["stage3_prompt_build_ms"] = t5

    # 阶段4 LLM生成
    ans_raw, t6 = llm_generate(prompt)
    log_item["stage4_llm_generate_ms"] = t6

    # 阶段5后处理
    final_ans, t7 = post_process(ans_raw)
    log_item["stage5_postprocess_ms"] = t7

    total_ms = (get_perf_timestamp()-total_start)*1000
    log_item["cache_hit"] = False
    log_item["total_ms"] = total_ms

    # 写入日志
    with open(f"{LOG_DIR}/perf_log.jsonl","a",encoding="utf‑8") as f:
        f.write(json.dumps(log_item,ensure_ascii=False)+"\n")

    # 写入缓存
    if ENABLE_CACHE:
        query_cache[user_query] = final_ans

    return final_ans, log_item

if __name__ == "__main__":
    test_query = "武汉力源本次募集资金投资哪些项目？"
    resp, log_info = rag_full_pipeline(test_query)
    print("===RAG返回答案===")
    print(resp)
    print("\n===全链路埋点日志===")
    print(json.dumps(log_info,indent=2,ensure_ascii=False))
