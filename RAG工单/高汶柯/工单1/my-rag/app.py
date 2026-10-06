# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# ============ 1. 配置 API Key ============
# 去 https://bailian.console.aliyun.com/ 注册，创建 API Key 后填在这里
os.environ["DASHSCOPE_API_KEY"] = "sk-af131926ed6d4d599031fe231cbe9fa5"

# ============ 2. PDF 解析 ============
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    return splitter.split_documents(docs)

# ============ 3. 构建向量库 ============
def build_vectorstore(chunks):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    return Chroma.from_documents(chunks, embeddings)

# ============ 4. 构建问答链 ============
def build_qa_chain(vectorstore):
    llm = ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    prompt = PromptTemplate(
        template="""你是一个专业的招股说明书问答助手。请严格根据以下参考信息回答问题。
如果参考信息中没有相关内容，请直接回答“文档中未找到相关信息”，不要编造。

参考信息：
{context}

问题：{question}

回答：""",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ============ 5. Gradio 界面 ============
vectorstore = None
qa_chain = None

def upload_pdf(file):
    global vectorstore, qa_chain
    if file is None:
        return "请先上传 PDF 文件"
    try:
        chunks = load_and_split_pdf(file.name)
        vectorstore = build_vectorstore(chunks)
        qa_chain = build_qa_chain(vectorstore)
        return f"✅ 解析完成，共 {len(chunks)} 个文本块，可以开始提问了"
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
        answer = result["result"]
        sources = "\n\n".join([
            f"[来源 {i+1}] {doc.page_content[:200]}..."
            for i, doc in enumerate(result["source_documents"])
        ])
        return answer, sources
    except Exception as e:
        return f"出错了：{str(e)}", ""

with gr.Blocks(title="招股说明书 RAG 问答系统") as demo:
    gr.Markdown("# 📄 招股说明书 RAG 问答系统")
    gr.Markdown("工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统")

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