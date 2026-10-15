# -*- coding: utf-8 -*-
# data_convert.py 工单16 IMDR原始数据集转LLaMA‑Factory多模态格式
import json
import os

RAW_DATA_PATH = "./dataset/raw/questions.jsonl"
OUT_TRAIN_JSON = "./dataset/llama_factory_train.json"

def convert_imdr_to_llamafactory():
    out_list = []
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"原始数据集不存在 {RAW_DATA_PATH}")

    with open(RAW_DATA_PATH,"r",encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            """
            IMDR原始字段示例:
            {"image":"xxx.jpg","question":"xxx","answer":"xxx"}
            """
            img_path = item.get("image","")
            q_text = item.get("question","")
            a_text = item.get("answer","")

            sample = {
                "image": img_path,
                "conversations":[
                    {"from":"human","value":f"<image>\n{q_text}"},
                    {"from":"gpt","value":a_text}
                ]
            }
            out_list.append(sample)

    with open(OUT_TRAIN_JSON,"w",encoding="utf‑8") as fw:
        json.dump(out_list,fw,ensure_ascii=False,indent=2)
    print(f"转换完成，样本总数:{len(out_list)}，输出：{OUT_TRAIN_JSON}")

if __name__ == "__main__":
    convert_imdr_to_llamafactory()
