# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的图像内容解析及检索优化
步骤2：图像 OCR —— EasyOCR（中文+英文）识别图表页文字
  运行环境：rag_env（D:\anaconda3\envs\rag_env\python.exe，含 easyocr 与已缓存的中文识别模型）
  运行：D:\anaconda3\envs\rag_env\python.exe ocr_charts.py
"""
import json
from pathlib import Path

import easyocr

OUT = Path("images") / "ocr_results.json"


def main():
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True, verbose=False)
    results = {}
    for png in sorted(Path("images").glob("page*.png")):
        res = reader.readtext(str(png), detail=1, paragraph=False)
        # detail=1: [box四角, text, conf]；按阅读顺序（先y后x）排序，保留图表空间结构
        items = [(b[0], b[2], " ".join(b[1].split())) for b in res if b[2] > 0.25]
        items.sort(key=lambda it: (min(p[1] for p in it[0]) // 40, min(p[0] for p in it[0])))
        results[png.stem] = [{"text": t, "conf": round(c, 3)} for _, c, t in items]
        print(f"{png.stem}: {len(items)} 个文本块")

    OUT.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"OCR 完成 -> {OUT}")


if __name__ == "__main__":
    main()
