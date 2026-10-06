# -*- coding: utf-8 -*-
# ChatHaruhi-54K -> ShareGPT 格式转换脚本（AutoDL 无卡模式可跑，streaming 省内存）
from datasets import load_dataset
import json

OUT_PATH = "/root/autodl-tmp/LlamaFactory/data/my_data.json"
MAX_SAMPLES = 10000

ds = load_dataset(
    "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
    split="train",
    streaming=True,
)

results = []
for i, row in enumerate(ds):
    if i >= MAX_SAMPLES:
        break

    agent_role = (row.get("agent_role") or "").strip()
    if not agent_role:
        continue  # 没有角色名的跳过

    conversations = []

    # system 人设
    system_prompt = (
        f"你现在是{agent_role}。请完全代入这个角色，"
        f"用{agent_role}的性格、语气和说话方式与用户对话，不要跳出角色。"
    )
    conversations.append({"from": "system", "value": system_prompt})

    # more_dialogues：历史多轮对话（防御性处理：只接受 dict 列表）
    more = row.get("more_dialogues")
    if isinstance(more, list):
        for turn in more:
            if not isinstance(turn, dict):
                continue
            q = turn.get("question") or turn.get("user") or turn.get("human") or ""
            a = turn.get("answer") or turn.get("assistant") or turn.get("gpt") or ""
            if q:
                conversations.append({"from": "human", "value": str(q)})
            if a:
                conversations.append({"from": "gpt", "value": str(a)})

    # 当前主对话
    if row.get("user_question"):
        conversations.append({"from": "human", "value": str(row["user_question"])})
    if row.get("agent_response"):
        conversations.append({"from": "gpt", "value": str(row["agent_response"])})

    # 至少 system + 一轮完整问答
    if len(conversations) >= 3:
        results.append({"conversations": conversations})

    if (i + 1) % 1000 == 0:
        print(f"已处理 {i + 1} 行，保留 {len(results)} 条")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"保存完成：{OUT_PATH}，共 {len(results)} 条")
