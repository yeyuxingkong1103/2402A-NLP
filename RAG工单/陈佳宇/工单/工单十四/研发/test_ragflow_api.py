# -*- coding: utf-8 -*-
# test_ragflow_api.py 工单14 批量调用RAGFlow API执行6道测试问题
import requests
import json

# =========配置，请修改为你的RAGFlow信息=========
RAGFLOW_BASE_URL = "http://127.0.0.1:8000/api/v1"
API_KEY = "你的ragflow api key"
KNOWLEDGE_BASE_ID = "你的知识库ID"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content‑Type":"application/json"
}

test_case_list = [
    {
        "id":"Q1",
        "question":"根据文本信息，该静电除尘器的发明人是:",
        "std_answer":"A. P·吉特勒"
    },
    {
        "id":"Q2",
        "question":"根据文本信息，以下哪个描述符合该静电除尘器的特征?",
        "std_answer":"管状入口具有单个圆锥形部分,达到外壳直径的80 至95%,剩余部分采用台阶形式。"
    },
    {
        "id":"Q3",
        "question":"在文件中第7 页的图片中,部件4 相对于部件5 在图片中的位置关系是?",
        "std_answer":"部件4 位于部件5 的左侧"
    },
    {
        "id":"Q4",
        "question":"在文件中第7 页的图片中,尺寸X1,X2,X3 分别代表什么部件的间隔距离?",
        "std_answer":"配气带孔盘6,6',6''之间的间隔距离"
    },
    {
        "id":"Q5",
        "question":"根据文件中第7 页图示,气流方向(7)首先经过哪个部件?紧接着会经过哪个部件?",
        "std_answer":"先经过部件6'',再经过部件6'"
    },
    {
        "id":"Q6",
        "question":"根据文件中第7 页图示,如果已知外壳直径D, 那么h1 和h2 的尺寸可以用来计算什么?",
        "std_answer":"确定配气带孔盘6,6',6''的位置"
    }
]

def ragflow_chat(question:str, kb_id:str):
    url = f"{RAGFLOW_BASE_URL}/chats/completions"
    payload = {
        "kb_ids":[kb_id],
        "question":question,
        "stream":False
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def batch_test():
    out_result = []
    for case in test_case_list:
        print(f"\n====执行 {case['id']} ====")
        print(f"问题：{case['question']}")
        api_resp = ragflow_chat(case["question"], KNOWLEDGE_BASE_ID)
        answer = api_resp.get("data",{}).get("answer","")
        print(f"模型回答：{answer}")
        print(f"标准答案：{case['std_answer']}")
        out_result.append({
            "id":case["id"],
            "question":case["question"],
            "std_answer":case["std_answer"],
            "model_answer":answer
        })
    with open("./batch_test_result.json","w",encoding="utf‑8") as f:
        json.dump(out_result, f, ensure_ascii=False, indent=2)
    print("\n✅批量测试完成，结果保存至 batch_test_result.json")

if __name__ == "__main__":
    batch_test()
