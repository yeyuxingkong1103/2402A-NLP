import pdfplumber
from pathlib import Path

class PdfParser:
    def __init__(self):
        pass

    def extract_text_and_table(self, pdf_path: str):
        """
        提取PDF全部文本 + 表格
        返回：full_text, table_list
        """
        full_text = ""
        table_list = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 提取页面文字
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                # 提取表格
                table = page.extract_table()
                if table:
                    table_list.append(table)
        return full_text, table_list

    def split_chunk(self, text: str, chunk_size=512, overlap=100):
        """简单文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

# 测试
if __name__ == "__main__":
    parser = PdfParser()
    # 你把一个pdf放到项目目录，改下面文件名
    test_pdf = "test.pdf"
    if Path(test_pdf).exists():
        txt, tables = parser.extract_text_and_table(test_pdf)
        chunks = parser.split_chunk(txt)
        print(f"✅ 解析完成，分块数量：{len(chunks)}")
        print("第一个chunk样例：")
        print(chunks[0])
    else:
        print(f"⚠️  请把 {test_pdf} 放到项目目录再测试")
