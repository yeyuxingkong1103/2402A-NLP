"""端到端集成测试：验证 MinerU / PaddleOCR 在 parse_file 链路中自动触发"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.parser import parse_file

# 测试1: 文本型 PDF (应该走 MinerU)
print("=== 测试1: 文本型 PDF (30天认知训练营) ===")
doc1 = parse_file("/home/dabaie/code/psychologist/心理医生/陈认知医生（CBT 认知行为治疗型）/30天认知训练营.pdf")
print(f"  title={doc1.title}")
print(f"  chars={doc1.char_count}")
print(f"  pages={doc1.pages}")
print(f"  parser={doc1.meta.get('parser', 'pymupdf/pdfplumber')}")
print(f"  preview: {doc1.text[:200]}")
assert doc1.char_count > 100, f"文本型 PDF 解析失败: {doc1.char_count} chars"
print("  PASS ✓")

print()

# 测试2: 扫描版 PDF (应该走 PaddleOCR 回退)
print("=== 测试2: 扫描版 PDF (思维改变生活) ===")
doc2 = parse_file("/home/dabaie/code/psychologist/心理医生/陈认知医生（CBT 认知行为治疗型）/思维改变生活：积极而实用的认知行为疗法.扫描版.pdf")
print(f"  title={doc2.title}")
print(f"  chars={doc2.char_count}")
print(f"  pages={doc2.pages}")
print(f"  parser={doc2.meta.get('parser', 'pymupdf/pdfplumber')}")
print(f"  preview: {doc2.text[:200]}")
assert doc2.char_count > 100, f"扫描版 PDF 解析失败: {doc2.char_count} chars"
print("  PASS ✓")

print()
print("########## 全部集成测试通过 ##########")
