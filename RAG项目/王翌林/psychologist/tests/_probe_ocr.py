import sys, os
sys.path.insert(0, "/home/dabaie/code/psychologist")
from src.rag.ocr_loader import mineru_extract_pdf, paddle_ocr_pdf

print("=== MinerU ===")
text1 = mineru_extract_pdf("/home/dabaie/code/psychologist/心理医生/陈认知医生（CBT 认知行为治疗型）/30天认知训练营.pdf")
print(f"  len={len(text1) if text1 else 0}")
print(f"  preview={text1[:200] if text1 else 'EMPTY'}")

print("\n=== PaddleOCR ===")
text2 = paddle_ocr_pdf("/home/dabaie/code/psychologist/心理医生/陈认知医生（CBT 认知行为治疗型）/思维改变生活：积极而实用的认知行为疗法.扫描版.pdf", pages="1-3")
print(f"  len={len(text2) if text2 else 0}")
print(f"  preview={text2[:200] if text2 else 'EMPTY'}")
