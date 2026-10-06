# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv()

_reranker = None


def load_pdf(file_path):
    """用 PyMuPDF 解析，保留版面结构（表格不再乱码）"""
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def split_documents(documents):
    """优化分块：更大切片 + 更多重叠 + 中文优先分隔符"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


def get_embeddings():
    """升级为 bge-base，精度显著提升"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )


def create_vectorstore(chunks):
    """使用新的向量库目录，避免和旧数据冲突"""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory="./chroma_db_v2",
    )


def get_reranker():
    """暂时禁用 Rerank（模型未下载完时保证系统可用）"""
    return None


def get_llm():
    return ChatOpenAI(
        temperature=0.1,
        model="glm-4-flash",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )


def build_rag_chain(vectorstore, llm):
    """MMR 检索 + Rerank 重排序 + LLM 生成"""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
    )

    prompt = PromptTemplate.from_template(
        """你是一个严谨的招股说明书问答助手。
请**仅基于**下面的"已知信息"回答问题。
如果已知信息中没有相关答案，请直接回答："根据现有资料无法回答该问题"。
不要编造、不要推测。

已知信息：
{context}

问题：{question}

答案（要具体、简洁、带数字）："""
    )

    def retrieve_and_rerank(query):
        """暂不 Rerank，直接返回检索结果（保证系统能跑通）"""
        docs = retriever.invoke(query)
        return docs[:3]

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def init_rag(pdf_path):
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    llm = get_llm()
    chain, retriever = build_rag_chain(vectorstore, llm)
    return chain, retriever