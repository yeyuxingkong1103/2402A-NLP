import os, subprocess
from pathlib import Path
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
import pymupdf
from docx import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pymilvus import MilvusClient, DataType, Function, FunctionType

data_path = Path(r"C:\Users\ZhuanZ\Desktop\a\data\刑法资料合编.pdf")
collection_name = "criminal_law_chunks"#一张存放刑法资料片段的表
mineru = r"D:\an\envs\langchain\Scripts\mineru-kit.exe"#mineru 程序的位置


def make_record(file, text, place="全文", page=0):
    #file：资料文件的路径 text：从资料里取得的文字 place：记录文字在资料中的来源；没填写就用 "全文" page：页码；没填写就用 0
    return {"text": str(text).strip(), "source": file.name, "location": place, "page": page}

def parse_pdf(file):
    output_dir = Path(r"C:\Users\ZhuanZ\Desktop\a\mineru")
    subprocess.run([mineru, "parse", str(file), "-o", str(output_dir), "--tier", "flash", "--pages", "all"], check=True)
    #启动外部程序 mineru check=True：如果 MinerU执行失败，就报错并停在这里
    text = (output_dir / f"{file.stem}.md").read_text(encoding="utf-8")#把里面的文字读出来，存进text
    return [make_record(file, text, "MinerU解析")]

def parse_word(file):
    results = []
    for i, p in enumerate(Document(file).paragraphs, 1):#逐个取出段落，同时从 1 开始给它们编号
        if p.text.strip():#去掉首尾空白
            results.append(make_record(file, p.text, f"第{i}段"))
    return results

def parse_json(file):
    with pymupdf.open(stream=file.read_bytes(), filetype="txt") as document:#读取内容并以文本文件打开
        text = "\n".join(page.get_text() for page in document)
    return [make_record(file, text, "fitz读取")]

def parse_other(file):#支持CSV、TXT、MD、HTML、XML
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url=os.getenv("DEEPSEEK_BASE_URL"))
    text = file.read_text(encoding="utf-8-sig")
    rows = []
    for i, start in enumerate(range(0, len(text), 5000), 1):#带索引遍历
        content = text[start:start + 5000]
        result = client.chat.completions.create(model="deepseek-flash",stream=True,
        messages=[{"role": "user","content": "把下面数据完整转换成Markdown，不要总结：\n" + content}])
        answer = ""
        for chunk in result:#每次取出一小块内容
            if chunk.choices:
                answer += chunk.choices[0].delta.content or ""
        if answer:
            rows.append(make_record(file, answer, f"第{i}段"))
    return rows

def load_file(file):
    suffix = file.suffix.lower()#取文件后缀
    if suffix == ".pdf": return parse_pdf(file)
    if suffix == ".docx": return parse_word(file)
    if suffix in {".json", ".jsonl"}: return parse_json(file)
    return parse_other(file)

def main():

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for source in load_file(data_path):
        for text in splitter.split_text(source["text"]):#取出text对应的值
            chunks.append({**source, "text": text})#覆盖前text的值

    embedding = OllamaEmbeddings(model="bge-m3:567m", client_kwargs={"timeout": 180})
    vectors = []
    for start in range(0, len(chunks), 32):
        texts = [row["text"] for row in chunks[start:start + 32]]#从chunks取最多32条记录并组成列表
        vectors.extend(embedding.embed_documents(texts))#计算向量

    print("向量化完成，准备写入 Milvus", flush=True)
    client = MilvusClient(uri="http://127.0.0.1:19530")#创建客户端
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)#删除旧集合
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=4096,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=len(vectors[0]))
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                 input_field_names=["text"], output_field_names=["sparse"]))
    indexes = client.prepare_index_params()
    indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")#为向量检索建立索引
    indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")#为BM25关键词检索建立索引
    client.create_collection(collection_name=collection_name, schema=schema, index_params=indexes)
    client.insert(collection_name=collection_name, data=[{**row, "vector": vector} for row, vector in zip(chunks, vectors)])
    client.close()
    print("写入完成：", len(chunks), "条")
if __name__ == "__main__":
    main()



