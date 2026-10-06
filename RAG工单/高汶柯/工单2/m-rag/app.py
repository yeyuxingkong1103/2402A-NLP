# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

import os
import gradio as gr
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from sentence_transformers import CrossEncoder

os.environ["DASHSCOPE_API_KEY"] = "sk-af131926ed6d4d599031fe231cbe9fa5"

# ============ 1. PDF 解析（pdfplumber 保留表格） ============
def load_pdf_with_tables(pdf_path):
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 正文
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i + 1, "type": "text"}))
            # 表格单独提取
            for table in page.extract_tables():
                table_md = "\n".join([" | ".join([c or "" for c in row]) for row in table if row])
                if table_md.strip():
                    docs.append(Document(page_content=table_md, metadata={"page": i + 1, "type": "table"}))
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    result = []
    for d in docs:
        if d.metadata.get("type") == "table":
            # 表格整个作为一个 chunk，不切断数字
            result.append(d)
        else:
            result.extend(splitter.split_documents([d]))
    return result


# ============ 2. 混合检索 + 重排序 ============
def build_retriever(chunks):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20

    # 混合检索：向量 0.5 + BM25 0.5
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble, chunks


# 轻量重排序模型
reranker = CrossEncoder("BAAI/bge-reranker-base")


def retrieve_and_rerank(retriever, chunks, query, top_k=3):
    candidates = retriever.invoke(query)
    if not candidates:
        return []
    pairs = [[query, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


# ============ 3. 问答链 ============
def build_qa_chain(chunks):
    retriever, _ = build_retriever(chunks)
    llm = ChatTongyi(model="qwen-turbo", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"])

    prompt = PromptTemplate(
        template="""你是一个专业的招股说明书问答助手。请严格根据以下参考信息回答问题。
如果参考信息中没有相关内容，请直接回答"文档中未找到相关信息"，不要编造。
支持中文和英文回答。

参考信息：
{context}

问题：{question}

回答：""",
        input_variables=["context", "question"]
    )

    class CustomQA:
        def __init__(self, retriever, chunks, llm, prompt):
            self.retriever = retriever
            self.chunks = chunks
            self.llm = llm
            self.prompt = prompt

        def invoke(self, inputs):
            query = inputs["query"]
            docs = retrieve_and_rerank(self.retriever, self.chunks, query, top_k=3)
            context = "\n\n".join([d.page_content for d in docs])
            final_prompt = self.prompt.format(context=context, question=query)
            answer = self.llm.invoke(final_prompt).content
            return {"result": answer, "source_documents": docs}

    return CustomQA(retriever, chunks, llm, prompt)


# ============ 4. Gradio 界面 ============
qa_chain = None

def upload_pdf(file):
    global qa_chain
    if file is None:
        return "请先上传 PDF 文件"
    try:
        docs = load_pdf_with_tables(file.name)
        chunks = split_docs(docs)
        qa_chain = build_qa_chain(chunks)
        return f"✅ 解析完成，共 {len(chunks)} 个文本块（含表格），可以开始提问了"
    except Exception as e:
        return f"❌ 解析失败：{str(e)}"


def answer_question(question):
    global qa_chain
    if qa_chain is None:
        return "请先上传 PDF 文件", ""
    if not question.strip():
        return "请输入问题", ""
    try:
        result = qa_chain.invoke({"query": question})
        sources = "\n\n".join([
            f"[来源 {i+1}] {doc.page_content[:200]}..."
            for i, doc in enumerate(result["source_documents"])
        ])
        return result["result"], sources
    except Exception as e:
        return f"出错了：{str(e)}", ""


with gr.Blocks(title="招股说明书 RAG 问答系统（优化版）") as demo:
    gr.Markdown("# 📄 招股说明书 RAG 问答系统（优化版）")
    gr.Markdown("工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化")

    with gr.Row():
        pdf_input = gr.File(label="上传 PDF 文档", file_types=[".pdf"])
        upload_btn = gr.Button("解析文档", variant="primary")
    status = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        question = gr.Textbox(label="输入你的问题", placeholder="例如：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？")
        ask_btn = gr.Button("提问", variant="primary")

    answer = gr.Textbox(label="答案", lines=6)
    sources = gr.Textbox(label="参考来源", lines=6)

    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=status)
    ask_btn.click(answer_question, inputs=question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()