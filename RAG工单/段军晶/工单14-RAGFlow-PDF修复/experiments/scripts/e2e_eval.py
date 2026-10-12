# -*- coding: utf-8 -*-
"""工单14 · Q1~Q6 端到端 RAG 问答评估（F 组配置）"""
import json, re, sys, time
import requests

BASE = "http://localhost:9380/api/v1"
TOKEN = "ragflow-Nl4U6CoqTT7HnaM8_kII_A1WJYuq708Z8ZKtpyVyqco"
DS = "2e7c1de0c2e411f19b360a36c60a8145"      # 工单14-CN100342976C专利
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# 题干与选项（来源：工单14 需求分析 附录A，及 pdf_groundtruth.md 的标准答案）
QUESTIONS = [
    dict(qid="Q1", group="纯文本", answer="A",
         question="根据文本信息，该静电除尘器的发明人是：",
         options=["A. P·吉特勒", "B. 未知", "C. 静电除尘器研究小组", "D. 该专利未提及发明人"]),
    dict(qid="Q2", group="纯文本", answer="C",
         question="根据文本信息，以下哪个描述符合该静电除尘器的特征？",
         options=["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极。",
                  "B. 具有管状入口和出口，它们分别由 3 种不同圆锥形部分所构成",
                  "C. 管状入口具有单个圆锥形部分，达到外壳直径的 80 至 95%，剩余部分采用台阶形式。",
                  "D. 主要用于液体的除尘"]),
    dict(qid="Q3", group="纯图像", answer="A",
         question="在文件中第 7 页的图片中，部件 4 相对于部件 5 在图片中的位置关系是？",
         options=["A. 部件 4 位于部件 5 的左侧", "B. 部件 4 位于部件 5 的右侧",
                  "C. 部件 4 位于部件 5 的上方", "D. 部件 4 位于部件 5 的下方"]),
    dict(qid="Q4", group="纯图像", answer="A",
         question="在文件中第 7 页的图片中，尺寸 X1，X2，X3 分别代表什么部件的间隔距离？",
         options=["A. 配气带孔盘 6，6'，6\" 之间的间隔距离",
                  "B. 外壳 2 与圆锥形部分 10 之间的间隔距离",
                  "C. 管状入口 1 与管状出口 9 之间的间隔距离",
                  "D. 外壳中心轴线 3 与外壳 2 之间的间隔距离"]),
    dict(qid="Q5", group="图文混合", answer="C",
         question="根据文件中第 7 页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？",
         options=["A. 先经过部件 4，再经过部件 5", "B. 先经过部件 10，再经过部件 6\"",
                  "C. 先经过部件 6\"，再经过部件 6'", "D. 先经过部件 6，再经过部件 4"]),
    dict(qid="Q6", group="图文混合", answer="B",
         question="根据文件中第 7 页图示，如果已知外壳直径 D，那么 h1 和 h2 的尺寸可以用来计算什么？",
         options=["A. 计算气流速度", "B. 确定配气带孔盘 6，6'，6\" 的位置",
                  "C. 计算除尘效率", "D. 确定外壳 2 的材料"]),
]

SUFFIX = "\n\n请严格依据检索到的资料作答。只回答一个选项，并在最后一行以「答案：X」的格式给出选项字母 X。"


def create_chat():
    body = {
        "name": f"工单14-F组6题验证-{int(time.time())}",
        "dataset_ids": [DS],
        "llm": {"model_name": "deepseek-chat@DeepSeek", "temperature": 0.1, "max_tokens": 1024},
        "prompt": {"similarity_threshold": 0.2, "vector_similarity_weight": 0.3,
                   "top_n": 8, "top_k": 1024},
    }
    r = requests.post(f"{BASE}/chats", headers=H, json=body, timeout=60)
    j = r.json()
    if j.get("code") != 0:
        print("create chat FAILED:", j)
        sys.exit(1)
    return j["data"]["id"]


def create_session(chat_id):
    r = requests.post(f"{BASE}/chats/{chat_id}/sessions", headers=H,
                      json={"name": "e2e-eval"}, timeout=60)
    j = r.json()
    if j.get("code") != 0:
        print("create session FAILED:", j)
        sys.exit(1)
    return j["data"]["id"]


def extract_letter(ans):
    m = re.findall(r"答案\s*[:：]\s*\**\s*([ABCD])", ans, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", ans)
    return m[-1].upper() if m else None


def main():
    chat_id = create_chat()
    sess_id = create_session(chat_id)
    print(f"chat_id={chat_id}\nsession_id={sess_id}\n")

    results = []
    for item in QUESTIONS:
        prompt = item["question"] + "\n" + "\n".join(item["options"]) + SUFFIX
        r = requests.post(f"{BASE}/chats/{chat_id}/completions",
                          headers=H,
                          json={"question": prompt, "session_id": sess_id, "stream": False},
                          timeout=180)
        j = r.json()
        ans = (j.get("data") or {}).get("answer", "")
        refs = (j.get("data") or {}).get("reference") or {}
        chunks = refs.get("chunks") or []
        got = extract_letter(ans)
        ok = (got == item["answer"])
        results.append({**item, "raw_answer": ans, "got": got, "ok": ok,
                        "n_chunks": len(chunks),
                        "chunk_snippets": [c.get("content", "")[:260] for c in chunks[:3]]})
        print("=" * 90)
        print(f"{item['qid']} [{item['group']}] 期望={item['answer']}  模型给出={got}  "
              f"{'✔ 正确' if ok else '✘ 错误'}")
        print(f"题: {item['question']}")
        print(f"答(原文): {ans.strip()[:600]}")
        print(f"检索命中 chunk 数: {len(chunks)}")

    n_ok = sum(1 for x in results if x["ok"])
    print("\n" + "=" * 90)
    print(f"总准确率: {n_ok}/{len(results)} = {n_ok/len(results)*100:.1f}%")
    for x in results:
        print(f"  {x['qid']} 期望{x['answer']} 实得{x['got']} {'✔' if x['ok'] else '✘'}")

    json.dump({"chat_id": chat_id, "session_id": sess_id, "results": results},
              open("e2e_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved -> e2e_results.json")


if __name__ == "__main__":
    main()
