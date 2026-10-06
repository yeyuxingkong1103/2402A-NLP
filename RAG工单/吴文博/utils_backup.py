import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def load_pdf(file_path):
    """加载 PDF 并解析为文档对象"""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents):
    """将文档切片"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)


def get_embeddings():
    """本地 Embedding 模型（首次运行会自动下载，约 100MB）"""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


def create_vectorstore(chunks):
    """向量化并存入 ChromaDB"""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory="./chroma_db",
    )


def get_llm():
    """初始化大语言模型"""
    return ChatOpenAI(
        temperature=0.1,
        model="glm-4-flash",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )


def build_rag_chain(vectorstore, llm):
    """构建 RAG 问答链（langchain 1.x LCEL 写法）"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """基于以下已知信息，简洁且专业地回答用户的问题。
如果无法从已知信息中得到答案，请直接说"根据目前的信息无法回答该问题"，不要编造答案。

已知信息：
{context}

问题：{question}

答案："""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def init_rag(pdf_path):
    """总入口：加载 PDF → 切片 → 向量化 → 构建链"""
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    llm = get_llm()
    chain, retriever = build_rag_chain(vectorstore, llm)
    return chain, retriever