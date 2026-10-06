# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

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
from sentence_transformers import CrossEncoder
from dashscope import MultiModalConversation

os.environ["DASHSCOPE_API_KEY"] = "sk-af131926ed6d4d599031fe231cbe9fa5"


# ============ 多模态模型：解析图片 ============
def describe_image(image_path):
    """用 qwen-vl 识别图片内容，返回文字描述"""
    try:
        messages = [{
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": "请详细描述这张图的内容，包括所有文字、数据、表格、流程、结构。"
                         "如果是流程图或组织架构图，请完整列出图中所有方框的文字和层级关系。"
                         "如果是图表，请列出图表标题、坐标轴、各数据点和数值。"}
            ]
        }]
        response = MultiModalConversation.call(
            model="qwen-vl-max",
            messages=messages
        )
        return response.output.choices[0].message.content[0]["text"]
    except Exception as e:
        return f"[图片解析失败：{e}]"


# ============ PDF 解析：正文 + 表格 + 图片 ============
def load_pdf_with_all(pdf_paths):
    docs = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1

                # 1. 正文
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num, "type": "text"}
                    ))

                # 2. 表格
                for table in page.extract_tables():
                    table_md = "\n".join([
                        " | ".join([c or "" for c in row]) for row in table if row
                    ])
                    if table_md.strip():
                        docs.append(Document(
                            page_content=table_md,
                            metadata={"source": pdf_path, "page": page_num, "type": "table"}
                        ))

                # 3. 图片（坐标夹紧，避免超界报错）
                for j, img in enumerate(page.images):
                    img_path = f"temp_{page_num}_{j}.png"
                    try:
                        x0 = max(0, img["x0"])
                        top = max(0, img["top"])
                        x1 = min(page.width, img["x1"])
                        bottom = min(page.height, img["bottom"])
                        if x1 <= x0 or bottom <= top:
                            continue
                        img_obj = page.crop((x0, top, x1, bottom))
                        img_obj.to_image(resolution=150).save(img_path)
                        desc = describe_image(img_path)
                        docs.append(Document(
                            page_content=f"[图片内容] {desc}",
                            metadata={"source": pdf_path, "page": page_num, "type": "image"}
                        ))
                        os.remove(img_path)
                    except Exception as e:
                        print(f"图片 {j} 解析失败：{e}")
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    result = []
    for d in docs:
        if d.metadata.get("type") in ("table", "image"):
            result.append(d)
        else:
            result.extend(splitter.split_documents([d]))
    return result


# ============ 混合检索 + 重排序 ============
def build_retriever(chunks):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble, chunks


reranker = CrossEncoder("BAAI/bge-reranker-base")


def retrieve_and_rerank(retriever, chunks, query, top_k=3):
    candidates = retriever.invoke(query)
    if not candidates:
        return []
    pairs = [[query, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


# ============ 问答链 ============
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


# ============ Gradio 界面 ============
qa_chain = None

def upload_pdfs(files):
    global qa_chain
    if not files:
        return "请先上传 PDF 文件"
    try:
        pdf_paths = [f.name for f in files]
        docs = load_pdf_with_all(pdf_paths)
        chunks = split_docs(docs)
        qa_chain = build_qa_chain(chunks)
        return f"✅ 解析完成，共 {len(chunks)} 个文本块（含表格和图片），可以开始提问了"
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


with gr.Blocks(title="招股说明书 RAG 问答系统（工单04优化版）") as demo:
    gr.Markdown("# 📄 招股说明书 RAG 问答系统（工单04优化版）")
    gr.Markdown("工单编号：人工智能NLP-RAG-图像内容解析及检索优化")

    with gr.Row():
        pdf_input = gr.File(
            label="上传 PDF 文档（支持多个）",
            file_types=[".pdf"],
            file_count="multiple"
        )
        upload_btn = gr.Button("解析文档", variant="primary")
    status = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        question = gr.Textbox(label="输入你的问题", placeholder="例如：武汉力源信息技术股份有限公司本次发行股数是多少？")
        ask_btn = gr.Button("提问", variant="primary")

    answer = gr.Textbox(label="答案", lines=6)
    sources = gr.Textbox(label="参考来源", lines=6)

    upload_btn.click(upload_pdfs, inputs=pdf_input, outputs=status)
    ask_btn.click(answer_question, inputs=question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()