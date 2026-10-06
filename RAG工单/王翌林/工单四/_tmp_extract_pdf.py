# 临时脚本：提取工单四 PDF 文本（工单编号：人工智能NLP-RAG-图像内容解析及检索优化）
import fitz

PDF = "/mnt/c/Users/Lenovo/Desktop/zg6工单/RAG 工单/人工智能NLP-RAG项目-04-PDF文档的图像内容解析及检索优化任务工单V1.1-20250206.pdf"

doc = fitz.open(PDF)
for i, page in enumerate(doc):
    print(f"===== 第{i+1}页 =====")
    print(page.get_text())
