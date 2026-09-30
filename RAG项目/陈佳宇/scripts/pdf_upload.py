import pdfplumber
from vector_store_milvus import MilvusVectorStore

vs = MilvusVectorStore()

def extract_pdf_text(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # 简单分块，按200字符切割
                for i in range(0, len(text), 200):
                    chunk = text[i:i+200]
                    chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    # 把你的pdf放到项目目录，修改这里文件名
    pdf_file = "test.pdf"
    print(f"开始解析PDF：{pdf_file}")
    text_chunks = extract_pdf_text(pdf_file)
    print(f"一共切分 {len(text_chunks)} 个文本块")
    # 插入milvus集合（和角色绑定的collection）
    collection_name = "rag_character_collection"
    vs.insert_texts(collection_name, text_chunks)
    print("✅ PDF文本块已经全部存入Milvus向量库！")
