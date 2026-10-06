# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的图像内容解析及检索优化
步骤4：图像内容入库 —— 每个图表页生成"图像内容块"（页内文字 + OCR文字 + BLIP描述译文）
运行：python build_index_img.py
"""
import json
import re
from pathlib import Path

import chromadb
import fitz
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION = "zhaogu2_img_v4"
WATERMARK = re.compile(r"八维教育|人工智能刘毅|rengongzhinengliumin|八维教")
CJK = r"\u4e00-\u9fff"


def compact_vertical_text(text: str) -> str:
    """图表页的竖排文字被抽成"珠 海 销 售 处"形式，合并单字间空格便于检索与阅读"""
    return re.sub(rf"([{CJK}]) (?=[{CJK}])", r"\1", text)


def analyze_org_chart(page):
    """组织结构图几何解析：把竖排文字按坐标聚类成'方框'，识别销售处框及其上级框。
    返回结构化描述（未能解析时返回空串）。"""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                t = s["text"].strip()
                if t:
                    spans.append((s["bbox"], t))
    # 两步聚类：①按 x 中心归列（竖排框一列=一个框）；②列内按 y 排序，y 间隔>20 切分为不同框
    spans.sort(key=lambda s: (s[0][0] + s[0][2]) / 2)
    columns = []
    for (x0, y0, x1, y1), t in spans:
        cx = (x0 + x1) / 2
        if columns and abs(columns[-1]["cx"] / columns[-1]["n"] - cx) < 12:
            c = columns[-1]
            c["spans"].append((y0, y1, x0, x1, t))
            c["cx"] += cx
            c["n"] += 1
        else:
            columns.append({"cx": cx, "n": 1, "spans": [(y0, y1, x0, x1, t)]})
    boxes = []
    for c in columns:
        c["cx"] /= c["n"]
        c["spans"].sort()
        cur = None
        for y0, y1, x0, x1, t in c["spans"]:
            if cur and y0 - cur["y1"] <= 20:
                cur["y1"] = y1
                cur["text"] += t
            else:
                if cur:
                    boxes.append(cur)
                cur = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": t}
        if cur:
            boxes.append(cur)
    boxes = [b for b in boxes if len(b["text"]) >= 4 and b["y0"] > 60]  # 去掉单字与页眉
    offices = [b for b in boxes if b["text"].endswith("销售处")]
    if not offices:
        return ""
    row_y = min(b["y0"] for b in offices)
    row = [b for b in offices if abs(b["y0"] - row_y) < 30]  # 同一行
    names = "".join(b["text"] for b in sorted(row, key=lambda b: b["x0"]))
    parents = [b for b in boxes if b["y1"] <= row_y + 5 and b["text"].endswith("销售部")]
    parent = min(parents, key=lambda b: (row_y - b["y1"]) + abs((b["x0"] + b["x1"]) / 2 - (min(x["x0"] for x in row) + max(x["x1"] for x in row)) / 2) * 0.3) if parents else None
    desc = f"图结构分析：本页组织结构图中，{parent['text']} 下设 {len(row)} 个销售处：{names}。" if parent \
        else f"图结构分析：本页共有 {len(row)} 个销售处标签框：{names}。"
    return desc


def page_text(pdf_path: str, pno: int) -> str:
    doc = fitz.open(pdf_path)
    t = doc[pno - 1].get_text()
    doc.close()
    return t


def translate_captions(caps: dict) -> dict:
    """BLIP 英文描述 -> 中文（qwen2.5 本地翻译，一次调用）"""
    src = "\n".join(f"{k}: {v}" for k, v in caps.items())
    prompt = (f"把下面的英文图像描述逐条翻译成中文，保持编号，只输出译文：\n{src}")
    resp = ollama.generate(model="qwen2.5:7b", prompt=prompt, stream=False,
                           options={"temperature": 0.1})
    out = {}
    for line in resp["response"].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = re.sub(r"^[\d\.\s\-\*、]+", "", k.strip().replace("：", ":").split(":")[0])
            if k in caps:
                out[k] = v.strip()
    return out


def main():
    pdf_path = Path("pdf_path2.txt").read_text(encoding="utf-8").strip()
    meta = json.loads(Path("images/chart_meta.json").read_text(encoding="utf-8"))
    ocr = json.loads(Path("images/ocr_results.json").read_text(encoding="utf-8"))
    caps_en = json.loads(Path("images/blip_captions.json").read_text(encoding="utf-8"))
    caps_zh = translate_captions({k: v for k, v in caps_en.items() if not v.startswith("a sample of a sample")})

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() > 0:
        print(f"集合已有 {col.count()} 条，跳过")
        return

    ids, docs, metas = [], [], []
    doc_fit = fitz.open(pdf_path)
    for pno_str, info in meta.items():
        pno = int(pno_str)
        ocr_key = f"page{pno_str}"  # ocr_results.json 的键为 png 文件名（pageXX）
        ocr_text = " | ".join(x["text"] for x in ocr.get(ocr_key, [])
                              if x["conf"] > 0.3 and not WATERMARK.search(x["text"]))
        ocr_text = ocr_text[:1500]
        vec_text = compact_vertical_text(page_text(pdf_path, pno)[:900].replace("\n", " "))
        struct = analyze_org_chart(doc_fit[pno - 1]) if "销售处" in vec_text else ""
        cap = caps_zh.get(pno_str, caps_en.get(pno_str, ""))
        chunk = (f"【图像内容】招股说明书2 第{pno}页（图表页，页内含图/结构图/增长图）。\n"
                 f"{struct}"
                 f"图像语义描述：{cap}\n"
                 f"图中识别文字（OCR）：{ocr_text}\n"
                 f"页面相关文字：{vec_text}")
        ids.append(f"img_p{pno}")
        docs.append(chunk)
        metas.append({"doc": "zhaogu2", "page": pno, "kind": "image"})
        print(f"页{pno} 图像块 {len(chunk)} 字符（OCR {len(ocr_text)}）")

    col.add(ids=ids, embeddings=ollama.embed(model="bge-m3", input=docs)["embeddings"],
            documents=docs, metadatas=metas)
    doc_fit.close()
    print(f"图像内容索引完成：{col.count()} 条 -> {COLLECTION}")


if __name__ == "__main__":
    main()
