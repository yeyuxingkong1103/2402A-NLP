# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 模块：tools/vision_build —— 图像语义解析（CLIP 多模态模型）→ 生成可检索的图像块
# 说明：按工单 04 要求，图像语义解析使用**多模态模型 CLIP**（openai/clip-vit-base-patch32，本地缓存）：
#   1) 对整页渲染图 + 抽取出的图表做 CLIP 图像编码；
#   2) 用一组“图像类型”文本提示做零样本分类（CLIP 图文语义匹配），得到该图的语义标签；
#   3) 与页面文字层/抽取图的文字层拼接，写成 type=image 的检索块。
# 运行环境：需 torch+transformers 的环境（本机 D:\an\envs\nlp2），并设 HF_HUB_OFFLINE=1 走本地缓存。
# 输出：data/vision/image_blocks.jsonl
import os
import sys
import json

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "data", "figures")
OUT = os.path.join(ROOT, "data", "vision")

# 图像类型提示（英文给 CLIP，中文作为可检索语义标签落到块文本里）
LABELS = [
    ("organization chart", "组织结构图"),
    ("ownership structure chart", "股权结构图"),
    ("sales network map", "销售网络/销售处分布图"),
    ("market share pie chart", "市场份额饼图"),
    ("bar chart of industry growth", "行业增长柱状图"),
    ("line chart of revenue trend", "收入趋势折线图"),
    ("business flow chart", "业务流程图"),
    ("financial data table", "财务数据表格"),
    ("product photo of electronic components", "电子元器件产品图"),
    ("company logo or seal", "公司标志/印章"),
]


def main():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    doc = sys.argv[1] if len(sys.argv) > 1 else "招股说明书2"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    prompts = [l[0] for l in LABELS]
    gloss = {l[0]: l[1] for l in LABELS}
    with torch.no_grad():
        tfeat = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    # 待解析的图片：整页渲染 + 抽取图表
    items = []
    pages_dir = os.path.join(FIG, doc, "pages")
    if os.path.isdir(pages_dir):
        for f in sorted(os.listdir(pages_dir)):
            if f.endswith(".png"):
                items.append({"kind": "page", "page": int(f[1:4]), "path": os.path.join(pages_dir, f),
                              "text_layer": ""})
    fig_json = os.path.join(FIG, "figures.json")
    if os.path.exists(fig_json):
        for x in json.load(open(fig_json, encoding="utf-8")):
            if x["doc"] != doc:
                continue
            items.append({"kind": "figure", "page": x["page"],
                          "path": os.path.join(ROOT, x["file"]), "text_layer": x.get("text_layer", "")})

    blocks = []
    for it in items:
        try:
            img = Image.open(it["path"]).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        with torch.no_grad():
            ifeat = model.get_image_features(**proc(images=img, return_tensors="pt"))
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            sims = (ifeat @ tfeat.T)[0]
            probs = sims.softmax(dim=-1)
        k = int(probs.argmax())
        tags = [(prompts[i], gloss[prompts[i]], round(float(probs[i]), 3))
                for i in probs.argsort(descending=True)[:3]]
        zh_tags = "、".join("%s(%s)" % (z, p) for p, z, _ in tags)
        if it["kind"] == "page":
            text = "【图片·第%d页整页】图像语义标签：%s" % (it["page"], zh_tags)
        else:
            text = "【图片·第%d页图表】图像语义标签：%s" % (it["page"], zh_tags)
        if it["text_layer"].strip():
            text += "；图内文字：" + it["text_layer"].strip()[:400]
        blocks.append({"doc": doc, "page": it["page"], "kind": it["kind"], "type": "image",
                       "text": text, "img_path": os.path.relpath(it["path"], ROOT).replace("\\", "/"),
                       "clip_tags": [{"en": p, "zh": z, "p": pr} for p, z, pr in tags],
                       "title": ""})

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "image_blocks.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print("image blocks:", len(blocks), "->", p)
    for b in blocks:
        if b["page"] in (38, 39, 72, 310) or "组织结构" in b["text"] or "市场" in b["text"]:
            print("  p%-4d %s | %s" % (b["page"], b["kind"], b["text"][:110]))


if __name__ == "__main__":
    main()
