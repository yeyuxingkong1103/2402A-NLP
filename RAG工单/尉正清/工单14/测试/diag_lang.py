# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""诊断：复刻 RAGFlow 的 is_english 判定，看这份中文专利会不会被误判

对应源码 deepdoc/parser/pdf_parser.py:1098-1105。判成英文的后果是
pdf_parser.py:1130 直接丢弃原生文字层，全量依赖 OCR。
"""
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pdfplumber

PDF = r"D:\BW\RAG 工单\14-17附件\original_problems\documents\CN100342976C.pdf"
# 与 pdf_parser.py:1099 完全一致的正则
PAT = r"[ a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}"


def main():
    random.seed(0)
    with pdfplumber.open(PDF) as pdf:
        print(f"页数: {len(pdf.pages)}")
        votes = 0
        for i, page in enumerate(pdf.pages):
            chars = [c["text"] for c in page.chars]
            txt = "".join(chars)
            cjk = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
            ratio = cjk / max(len(txt), 1)
            sample = "".join(random.choices(chars, k=min(100, len(chars)))) if chars else ""
            hit = bool(re.search(PAT, sample))
            votes += hit
            print(f"  第{i+1}页 字符数={len(chars):5d} 中文占比={ratio:.0%} 判英文={hit}")
        print(f"\n英文票数 {votes}/{len(pdf.pages)}"
              f" -> 整体 is_english={votes > len(pdf.pages) / 2}")

        print("\n=== 第7页原生文字层前 500 字 ===")
        print(pdf.pages[6].extract_text()[:500])


if __name__ == "__main__":
    main()
