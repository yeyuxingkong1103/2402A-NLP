# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
"""
数据集生成模块（二）：调用 DeepSeek API 为每个段落生成问答对。

输入：dataset/passages.jsonl
输出：dataset/qa_pairs.jsonl —— {question, answer, passage, doc_id, page}
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")

# DeepSeek 客户端（环境变量 deepseek_api_key1 / deepseek_base_url）
client = OpenAI(
    api_key=os.getenv("deepseek_api_key1"),
    base_url=os.getenv("deepseek_base_url") or os.getenv("deepseek_base_url1"),
    timeout=60,
)

PROMPT_TMPL = """你是金融领域训练数据生成专家。请根据下面的年报段落，生成 1 个高质量问答对，用于训练文本向量模型。

要求：
1. 问题必须具体、明确，答案能完全从段落中找到（优先事实型、数字型、定义型问题）；
2. 问题表述要像真实用户提问，可以使用代词替换或省略部分实体，但不可出现段落中没有的信息；
3. 答案直接摘自段落，简短准确（数字、名称、定义优先），不超过 50 字；
4. 只输出 JSON，不要输出任何解释。

输出格式：
{{"question": "...", "answer": "..."}}

年报段落：
{passage}
"""


def gen_one(item: dict, max_retry: int = 3) -> dict:
    """为单个段落生成 QA。"""
    text = item["text"]
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": PROMPT_TMPL.format(passage=text)}],
                temperature=0.3,
                max_tokens=256,
            )
            content = resp.choices[0].message.content.strip()
            # 兼容 ```json 包裹
            if content.startswith("```"):
                content = content.strip("`")
                content = content[content.find("{"): content.rfind("}") + 1]
            qa = json.loads(content)
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if question and answer:
                return {
                    "question": question,
                    "answer": answer,
                    "passage": text,
                    "doc_id": item.get("doc_id", ""),
                    "page": item.get("page", 0),
                }
        except Exception as e:
            if attempt == max_retry - 1:
                return {"_error": str(e)[:100]}
            time.sleep(2)
    return {"_error": "empty"}


def main():
    passages = []
    with open(os.path.join(DATASET, "passages.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))

    print(f"共 {len(passages)} 个段落，开始并发生成 QA（20 线程）...")

    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(gen_one, p): p for p in passages}
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if "_error" not in r:
                results.append(r)
            if done % 30 == 0:
                elapsed = time.time() - t0
                print(f"  进度 {done}/{len(passages)} | 成功 {len(results)} | 耗时 {elapsed:.0f}s")

    out_path = os.path.join(DATASET, "qa_pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n生成完成：成功 {len(results)}/{len(passages)} -> {out_path}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
