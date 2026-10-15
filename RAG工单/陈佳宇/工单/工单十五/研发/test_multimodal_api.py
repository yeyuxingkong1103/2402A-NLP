# -*- coding: utf-8 -*-
# test_multimodal_api.py 工单15批量测试脚本
import requests
import json
import time
from config import RAGFLOW_BASE_URL,RAGFLOW_API_KEY,KB_ID
from multimodal_retrieval import detect_figure_reference

HEADERS = {
    "Authorization":f"Bearer {RAGFLOW_API_KEY}",
    "Content‑Type":"application/json"
}

TEST_CASES = [
    {
        "id":"Q1",
        "question":"根据专利文本,本发明主要涉及哪种物料的分配装置?",
        "std_answer":"块状散料"
    },
    {
        "id":"Q2",
        "question":"根据专利文本,本发明的分散装置包含以下哪个组件?",
        "std_answer":"链条"
    },
    {
        "id":"Q3",
        "question":"在文件中第11 页图3 中,编号13 的部件相对于编号12 的部件的位置关系是?",
        "std_answer":"位于编号12 的部件之内"
    },
    {
        "id":"Q4",
        "question":"在文件中第11 页图3 中,编号14 的部件位于整个装置的哪个位置?",
        "std_answer":"顶部"
    },
    {
        "id":"Q5",
        "question":"根据文件中第11 页图3,散料从部件14 进入后,下一步会经过哪个部件?",
        "std_answer":"部件13"
    },
    {
        "id":"Q6",
        "question":"在文件中第11 页图3 的装置中,如果需要调整链条的位置,需要操作哪个部件?",
        "std_answer":"部件11"
    }
]

def ragflow_chat_api(question:str, kb_id:str):
    url = f"{RAGFLOW_BASE_URL}/chats/completions"
    payload = {
        "kb_ids":[kb_id],
        "question":question,
        "stream":False
    }
    resp = requests.post(url,headers=HEADERS,json=payload,timeout=120)
    resp.raise_for_status()
    return resp.json()

def batch_run():
    result_out = []
    for case in TEST_CASES:
        qid = case["id"]
        q = case["question"]
        std = case["std_answer"]
        print(f"\n====== {qid} ======")
        print(f"问题：{q}")
        has_fig,page,fig = detect_figure_reference(q)
        print(f"检测图纸引用: has={has_fig}, page={page}, fig={fig}")
        t0 = time.perf_counter()
        api_ret = ragflow_chat_api(q,KB_ID)
        cost_sec = round(time.perf_counter()-t0,2)
        ans = api_ret.get("data",{}).get("answer","")
        print(f"模型回答：{ans}")
        print(f"标准答案：{std}")
        print(f"耗时(秒):{cost_sec}")
        result_out.append({
            "id":qid,
            "question":q,
            "std_answer":std,
            "model_answer":ans,
            "detect_figure":{"has":has_fig,"page":page,"figure":fig},
            "cost_second":cost_sec
        })
    with open("./multimodal_test_result.json","w",encoding="utf‑8") as f:
        json.dump(result_out,f,ensure_ascii=False,indent=2)
    print("\n✅批量测试完成，输出multimodal_test_result.json")

if __name__ == "__main__":
    batch_run()
