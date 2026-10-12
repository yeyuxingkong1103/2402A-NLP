# -*- coding: utf-8 -*-
"""
工单十六：微调数据转换脚本
将 IMDR 的 QA 对（questions.jsonl）+ 文档页面图片，转换成视觉语言模型微调格式 JSONL。
输出字段：{"image": <图片相对路径>, "question": <问题>, "answer": <完整答案>, "document": ...}
同时处理纯文本问题（image 置空，用于文本侧领域适应）。
"""
import os, re, json, random, argparse

QA_PATH_DEFAULT = "/mnt/c/Users/Lenovo/Desktop/zg6工单/14-17附件/original_problems/questions.jsonl"
IMG_ROOT_DEFAULT = "/home/dabaie/code/工单/工单十六/data/images"
OUT_DIR_DEFAULT = "/home/dabaie/code/工单/工单十六/data"

# 图像类问题模式
IMG_PAT = re.compile(r"第\s*(\d+)\s*页.*图|图片中|图中|图示|附图|部件\s*\d|位置关系|气流方向|尺寸\s*[xXhH]\d")
PAGE_PAT = re.compile(r"第\s*(\d+)\s*页")


def pick_image(doc: str, question: str, img_root: str):
    """根据问题定位页面图片；无页码则取文档首页或含图页。"""
    doc_dir = os.path.join(img_root, doc)
    if not os.path.isdir(doc_dir):
        return None
    pages = sorted(f for f in os.listdir(doc_dir) if f.endswith(".png"))
    if not pages:
        return None
    m = PAGE_PAT.search(question)
    if m:
        pno = int(m.group(1))
        cand = f"page_{pno:02d}.png"
        if cand in pages:
            return os.path.join(doc, cand)
    # 默认首页
    return os.path.join(doc, pages[0])


def build_answer_text(rec: dict) -> str:
    """把选择题 answer(选项字母) 转成完整答案文本，供模型学习输出。"""
    ans = rec.get("answer", "")
    options = rec.get("options", [])
    if options and isinstance(ans, str) and len(ans) == 1 and ans.upper() in "ABCDEFGH":
        idx = ord(ans.upper()) - ord("A")
        if 0 <= idx < len(options):
            # 去掉 "A. " 前缀，返回纯内容
            txt = re.sub(r"^[A-H]\.\s*", "", options[idx])
            return txt
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=QA_PATH_DEFAULT)
    ap.add_argument("--img-root", default=IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--eval-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    available_docs = set(d for d in os.listdir(args.img_root)
                         if os.path.isdir(os.path.join(args.img_root, d)))

    records = []
    with open(args.qa, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc = rec["document"].replace(".pdf", "")
            if doc not in available_docs:
                continue  # 仅保留有图片的文档，保证图文对齐
            is_img = bool(IMG_PAT.search(rec["question"]))
            item = {
                "question": rec["question"],
                "answer": build_answer_text(rec),
                "document": doc,
                "group": rec.get("group"),
                "is_image": is_img,
            }
            if is_img:
                img = pick_image(doc, rec["question"], args.img_root)
                if img:
                    item["image"] = img
                else:
                    item["image"] = ""
            else:
                item["image"] = ""
            records.append(item)

    # 打乱并划分 train/eval
    random.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_ratio))
    eval_set = records[:n_eval]
    train_set = records[n_eval:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    eval_path = os.path.join(args.out_dir, "eval.jsonl")
    for path, data in [(train_path, train_set), (eval_path, eval_set)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_img = sum(1 for r in records if r["image"])
    print(f"总样本: {len(records)} | 含图样本: {n_img} | 纯文本: {len(records)-n_img}")
    print(f"训练集: {len(train_set)} -> {train_path}")
    print(f"评估集: {len(eval_set)} -> {eval_path}")
    # 样例
    print("=== 样例（含图）===")
    for r in [x for x in records if x["image"]][:2]:
        print(json.dumps(r, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
