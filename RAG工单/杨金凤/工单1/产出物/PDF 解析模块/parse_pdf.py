from langchain_community.document_loaders import PDFPlumberLoader
from pathlib import Path

pdf_path = "/root/autodl-tmp/projects/RAG/0_raw_data/招股说明书1.pdf"
output_path = "/root/autodl-tmp/projects/RAG/1_trans_data/招股说明书1.txt"

# 加载 PDF
loader = PDFPlumberLoader(pdf_path)
docs = loader.load()

print(f"共加载 {len(docs)} 页")

# 将每页内容拼接并保存为纯文本，便于后续知识库管理
full_text = "\n\n".join([doc.page_content for doc in docs])
Path(output_path).write_text(full_text, encoding="utf-8")

print(f"解析完成，文本已保存至 {output_path}")
print("前 500 字符预览：")
print(full_text[:500])