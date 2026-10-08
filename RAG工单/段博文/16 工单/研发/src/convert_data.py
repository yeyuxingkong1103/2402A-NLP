# -*- coding: utf-8 -*-
# 工单16：微调数据格式转换脚本
"""
将 IMDR 的 questions.jsonl 转换为 VLM 微调所需的 JSONL 格式。
每条数据包含：image（页面图像路径）、conversations（question+answer对话格式）。
基于 LLaMA-Factory 的视觉微调数据格式。
"""
import json
import re
from pathlib import Path

SRC = Path(r"d:\作业\6-专高NLP 作业\RAG 工单\14-17附件\original_problems")
QUESTIONS = SRC / "questions.jsonl"
DOCS = SRC / "documents"
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

# 工业术语词典（用于评估）
INDUSTRY_TERMS = [
    "淬火", "回火", "正火", "退火", "渗碳", "渗氮", "公差", "配合", "轴承",
    "法兰", "齿轮", "链条", "螺栓", "焊接", "铸造", "锻造", "轧制",
    "电极", "除尘器", "熔化", "气化器", "还原", "铁矿石", "海绵铁",
    "冷却剂", "辊套", "密封", "衬板", "耐磨", "高温", "合金钢",
    "落料架", "分散装置", "配气带孔盘", "圆锥形", "圆柱形",
    "内冷导坯辊", "法兰定位器", "静电除尘器",
]


def parse_visual_ref(question: str) -> dict:
    """提取视觉引用信息。"""
    ref = {"has_image": False, "page": None, "figure": None, "component_ids": []}
    m = re.search(r"第\s*(\d+)\s*页.*?(图|图片|图示)\s*(\d+)", question)
    if m:
        ref["has_image"] = True
        ref["page"] = int(m.group(1))
        ref["figure"] = int(m.group(3))
    for cid in re.findall(r"编号\s*(\d+)|部件\s*(\d+)", question):
        ref["component_ids"].append(int(cid[0] or cid[1]))
    return ref


def answer_text(options, answer_letter):
    """将选项字母转换为答案文本。"""
    idx = ord(answer_letter) - ord("A")
    if 0 <= idx < len(options):
        return options[idx]
    return answer_letter


def convert():
    """转换主函数。"""
    records = []
    stats = {"total": 0, "text_only": 0, "image_related": 0, "has_industry_terms": 0}

    with open(QUESTIONS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item["question"]
            ans = answer_text(item["options"], item["answer"])
            doc = item["document"]
            vref = parse_visual_ref(q)

            # 图像路径（如果问题涉及图片）
            image_path = ""
            if vref["has_image"]:
                page = vref["page"] or 1
                image_path = f"documents/{doc}"

            # 检测工业术语
            terms_hit = [t for t in INDUSTRY_TERMS if t in q or t in ans]

            record = {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": ans},
                ],
                "metadata": {
                    "document": doc,
                    "group": item.get("group", 1),
                    "answer_letter": item["answer"],
                    "has_image": vref["has_image"],
                    "page": vref["page"],
                    "figure": vref["figure"],
                    "component_ids": vref["component_ids"],
                    "industry_terms": terms_hit,
                },
            }
            records.append(record)
            stats["total"] += 1
            if vref["has_image"]:
                stats["image_related"] += 1
            else:
                stats["text_only"] += 1
            if terms_hit:
                stats["has_industry_terms"] += 1

    # 划分训练集/验证集（9:1）
    split = int(len(records) * 0.9)
    train = records[:split]
    val = records[split:]

    # 输出
    train_file = OUT / "train.jsonl"
    val_file = OUT / "val.jsonl"
    for fpath, data in [(train_file, train), (val_file, val)]:
        with open(fpath, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["train"] = len(train)
    stats["val"] = len(val)
    stats_file = OUT / "data_stats.json"
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"转换完成:")
    print(f"  总数据: {stats['total']} 条")
    print(f"  训练集: {stats['train']} 条")
    print(f"  验证集: {stats['val']} 条")
    print(f"  纯文本题: {stats['text_only']} 条")
    print(f"  图像相关题: {stats['image_related']} 条")
    print(f"  含工业术语: {stats['has_industry_terms']} 条")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    return stats


if __name__ == "__main__":
    convert()
