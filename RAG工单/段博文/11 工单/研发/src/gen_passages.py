# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
"""
数据集生成模块（一）：从 9 份金融年报中采样高质量段落。

输入：ccf_competition/txt/*.txt（JSONL：page / type / inside）
输出：passages.jsonl —— 每份报告均匀采样的候选段落
"""

import json
import os
import random
import re

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(BASE, "data", "ccf_competition", "txt")
OUT = os.path.join(BASE, "dataset")
os.makedirs(OUT, exist_ok=True)

# 每份报告采样段落数
PER_DOC = 40


def clean_text(text: str) -> str:
    """清洗段落文本。"""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_doc_passages(path: str):
    """解析一份年报 JSONL，提取文本段落。"""
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # type 为文本块（非表格/图）
            btype = obj.get("type", "")
            inside = obj.get("inside", "")
            page = obj.get("page", 0)

            if not inside:
                continue
            text = clean_text(inside)

            # 长度过滤：过短无信息，过长不适合单段 QA
            if 80 <= len(text) <= 500:
                # 剔除乱码比例过高的段落
                zh_ratio = len(re.findall(r"[\u4e00-\u9fa5]", text)) / max(len(text), 1)
                if zh_ratio < 0.3:
                    continue
                passages.append({"text": text, "page": page, "block_type": btype})

    return passages


def main():
    files = sorted(f for f in os.listdir(TXT_DIR) if f.endswith(".txt"))
    print(f"共发现 {len(files)} 份年报")

    all_passages = []
    for fname in files:
        path = os.path.join(TXT_DIR, fname)
        passages = load_doc_passages(path)
        print(f"  {fname[:40]}... 合格段落 {len(passages)} 个")

        # 均匀采样：按页间隔抽样，保证全文覆盖
        if len(passages) > PER_DOC:
            step = len(passages) / PER_DOC
            sampled = [passages[int(i * step)] for i in range(PER_DOC)]
        else:
            sampled = passages

        # 简短文档名（公司简称）
        doc_id = fname
        for p in sampled:
            p["doc_id"] = doc_id
        all_passages.extend(sampled)

    random.shuffle(all_passages)

    out_path = os.path.join(OUT, "passages.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n采样完成：共 {len(all_passages)} 个段落 -> {out_path}")


if __name__ == "__main__":
    main()
