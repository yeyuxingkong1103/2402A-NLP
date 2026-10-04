# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单
数据转换脚本：将 IMDR 的 QA 对 + 图文块转换为 VLM 微调格式
JSONL，每行含 image / question / answer 字段。
"""
import json
import os
import zipfile

import config


def load_questions(limit=None):
    """从 zip 直接读取 questions.jsonl（无需整体解压）。"""
    z = zipfile.ZipFile(config.DATA_ZIP)
    rows = []
    with z.open(config.QUESTIONS_IN_ZIP) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line.decode("utf-8")))
    z.close()
    return rows


def resolve_answer(entry):
    """把答案字母（A/B/C/D）解析为完整答案文本。"""
    ans = entry.get("answer", "")
    if ans in "ABCD" and entry.get("options"):
        idx = "ABCD".index(ans)
        if idx < len(entry["options"]):
            return entry["options"][idx]
    return ans


def build_instruction(entry):
    """构造 question：拼接选项，形成指令式问句。"""
    q = entry["question"]
    opts = entry.get("options") or []
    if opts:
        q += "\n" + "\n".join(opts)
    return q


def convert(limit=None):
    """IMDR QA 对 -> VLM 微调 JSONL。"""
    rows = load_questions(limit)
    os.makedirs(config.OUT_DIR, exist_ok=True)
    n = 0
    with open(config.TRAIN_JSONL, "w", encoding="utf-8") as f:
        for entry in rows:
            if "image" in entry and entry["image"]:
                image = entry["image"]
            else:
                # 图片型 PDF：image 字段指向文档对应页截图
                image = f"{entry['document']}#page1"
            sample = {
                "image": image,
                "question": build_instruction(entry),
                "answer": resolve_answer(entry),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1
    print(f"[data_convert] 生成 {n} 条训练样本 -> {config.TRAIN_JSONL}")
    return n


if __name__ == "__main__":
    convert()
