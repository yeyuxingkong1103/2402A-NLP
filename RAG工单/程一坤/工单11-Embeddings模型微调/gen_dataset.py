# -*- coding: utf-8 -*-
"""
工单11 步骤1：微调数据集生成（问答对生成）
工单编号：人工智能NLP-RAG 项目-Embedding模型微调任务
功能：从《招股说明书1.pdf》抽样分块，用 LLM 为每个分块生成问答对，
     输出 train.jsonl / eval.jsonl（正例对格式：query + 相关文档片段）。
运行：python gen_dataset.py
"""
import sys
import os
import json
import random

# 当前脚本目录与公共模块路径，保证能 import 00-公共模块
HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "00-公共模块")
sys.path.insert(0, COMMON)

from config import PDF_ZGS1  # 招股说明书1.pdf 路径配置
from pdf_parser import parse_pdf_text  # PDF 文本解析（按页返回）
from chunker import chunk_pages  # 按页信息分块
from ollama_client import client  # Ollama 本地模型客户端

# 问答对生成提示词：每块生成3个问题（答案须能在块内找到，只输出问题不输出答案）
GEN_PROMPT = """你是训练数据构造器。请根据下面的文档片段生成 3 个问答对：
1. 问题必须是用户会自然提出的（如针对片段中的数字、人名、事实提问）；
2. 答案必须能在片段中找到（答案不输出，只输出问题）；
3. 只输出 JSON 数组：["问题1", "问题2", "问题3"]

【文档片段】
{chunk}

【JSON数组】"""


def main():
    # 固定随机种子保证抽样/打乱可复现（重跑结果一致，便于验收）
    random.seed(42)
    out_dir = os.path.join(HERE, "dataset")
    os.makedirs(out_dir, exist_ok=True)  # 数据集输出目录，不存在则创建
    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")
    # 幂等保护：数据集已生成过则跳过（LLM批量生成耗时长且token有成本）
    if os.path.exists(train_path):
        print("数据集已存在，跳过生成")
        return

    # 1. 分块并抽样
    pages = parse_pdf_text(PDF_ZGS1)  # 解析PDF为逐页文本
    chunks = chunk_pages(pages)  # 按页滑动窗口分块
    # 优先抽含数字/事实的块，微调目标正是"问题→事实块"检索
    # 只保留含金额/注册信息/股东等关键词的块，纯叙述块的问答对检索价值低
    fact_chunks = [c["text"] for c in chunks
                   if any(x in c["text"] for x in ("万元", "亿元", "注册资本", "法定代表人", "持股"))]
    random.shuffle(fact_chunks)  # 打乱保证抽样均匀覆盖全文
    sampled = fact_chunks[:45]  # 抽45块，每块3问约130+对，规模够微调用
    print(f"抽样 {len(sampled)} 个事实分块（共{len(chunks)}块）")

    # 2. LLM 生成问答对
    pairs = []  # 累积正例对 {"query": 问题, "pos": 相关文档片段}
    for i, chunk in enumerate(sampled, 1):
        try:
            # 块截断800字防超模型窗口；temperature=0.3 略给多样性但保持事实性
            raw = client.generate(GEN_PROMPT.format(chunk=chunk[:800]),
                                  temperature=0.3, num_predict=200)
            import re
            # 正则抠出模型输出中的 JSON 数组（模型可能带说明文字）
            m = re.search(r"\[.*\]", raw, re.S)
            qs = json.loads(m.group(0)) if m else []  # 解析失败则视为0个问题
            for q in qs[:3]:  # 每块最多取3个问题
                q = str(q).strip()
                # 长度过滤：太短（<8字）问题无信息量，太长（>80字）不像真实用户提问
                if 8 <= len(q) <= 80:
                    pairs.append({"query": q, "pos": chunk})  # 正例对：问题 + 其来源块
            # 打印进度与累计数量
            print(f"  [{i}/{len(sampled)}] +{len(qs)}对 (累计{len(pairs)})")
        except Exception as e:
            # 单块生成失败（JSON解析错/模型超时）只跳过该块，不中断整体
            print(f"  [{i}] 生成失败: {e}")

    # 3. 切分训练/评估集
    random.shuffle(pairs)  # 打乱后按序切分，保证两集分布一致
    # 评估集取 max(20, 总数/6)：至少20对保证评估统计有意义
    n_eval = max(20, len(pairs) // 6)
    # 前 n_eval 条写评估集（jsonl 每行一个JSON对象，ensure_ascii=False 保留中文）
    with open(eval_path, "w", encoding="utf-8") as f:
        for p in pairs[:n_eval]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # 其余写训练集
    with open(train_path, "w", encoding="utf-8") as f:
        for p in pairs[n_eval:]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"完成: 训练 {len(pairs)-n_eval} 对 → {train_path}")
    print(f"      评估 {n_eval} 对 → {eval_path}")


if __name__ == "__main__":
    main()
