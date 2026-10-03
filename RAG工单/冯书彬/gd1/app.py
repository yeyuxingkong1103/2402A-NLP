import json
import os
import tempfile
import time
from pathlib import Path

import requests
import streamlit as st
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBED_MODEL = "text-embedding-v4"
CHAT_MODEL = "qwen-plus"
MINERU_BASE_URL = "https://mineru.net/api/v1/agent"
DEFAULT_PDF_PATH = r"C:\Users\bin\Desktop\作业\招股说明书1-无水印.pdf"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严格基于PDF资料回答问题的RAG助手。只能使用参考资料回答；"
            "如果资料不足，请说明“PDF中未找到明确依据”。回答要简洁、准确。",
        ),
        ("human", "参考资料：\n{context}\n\n问题：{question}"),
    ]
)


def get_secret(name: str, label: str) -> str:
    value = os.getenv(name) or st.secrets.get(name, "")
    if not value:
        st.error(f"请先配置 {label}。")
        st.stop()
    return value


def save_upload(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file.getvalue())
        return temp.name


def mineru_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def split_page_ranges(page_range: str, batch_size: int = 20) -> list[str]:
    ranges = []
    for part in page_range.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x.strip()) for x in part.split("-", 1)]
        else:
            start = end = int(part)
        if start <= 0 or end < start:
            raise ValueError(f"页码范围不合法：{part}")
        while start <= end:
            batch_end = min(start + batch_size - 1, end)
            ranges.append(f"{start}-{batch_end}" if start != batch_end else str(start))
            start = batch_end + 1
    return ranges or ["1-20"]


def parse_pdf_with_mineru(pdf_path: str, api_key: str, page_range: str, is_ocr: bool = True) -> str:
    file_name = Path(pdf_path).name
    payload = {
        "file_name": file_name,
        "language": "ch",
        "enable_table": True,
        "enable_formula": False,
        "is_ocr": is_ocr,
    }
    if page_range.strip():
        payload["page_range"] = page_range.strip()
    headers = mineru_headers(api_key)
    resp = requests.post(f"{MINERU_BASE_URL}/parse/file", json=payload, headers=headers, timeout=30)
    data = resp.json()
    if resp.status_code != 200 or data.get("code") != 0:
        raise RuntimeError(f"MinerU创建任务失败：{data}")

    task_id = data["data"]["task_id"]
    upload_url = data["data"]["file_url"]
    with open(pdf_path, "rb") as file:
        upload_resp = requests.put(upload_url, data=file, timeout=120)
    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(f"MinerU上传文件失败：HTTP {upload_resp.status_code}")

    started = time.time()
    while time.time() - started < 900:
        result_resp = requests.get(f"{MINERU_BASE_URL}/parse/{task_id}", headers=headers, timeout=30)
        result = result_resp.json()
        state = result.get("data", {}).get("state")
        if state == "done":
            md_url = result["data"].get("markdown_url")
            if not md_url:
                raise RuntimeError("MinerU解析完成，但没有返回 Markdown 下载地址。")
            return requests.get(md_url, timeout=120).text
        if state == "failed":
            raise RuntimeError(f"MinerU解析失败：{result.get('data', {}).get('err_msg', result)}")
        time.sleep(3)
    raise TimeoutError(f"MinerU解析超时，请稍后重试。task_id={task_id}")


def make_embeddings(api_key: str):
    return DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=api_key)


def make_llm(api_key: str):
    return ChatTongyi(model=CHAT_MODEL, dashscope_api_key=api_key)


def build_vectorstore(pdf_path: str, qwen_key: str, mineru_key: str, page_range: str, chunk_size: int, overlap: int, progress=None):
    markdown_parts = []
    ranges = split_page_ranges(page_range)
    for index, current_range in enumerate(ranges, start=1):
        if progress:
            progress.info(f"正在解析第 {index}/{len(ranges)} 批：{current_range}")
        markdown_parts.append(parse_pdf_with_mineru(pdf_path, mineru_key, current_range, is_ocr=True))
    markdown = "\n\n".join(markdown_parts)
    if not markdown.strip():
        raise ValueError("MinerU没有解析出文本，请检查 PDF 内容。")
    docs = [Document(page_content=markdown, metadata={"source": pdf_path, "page_range": page_range})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, make_embeddings(qwen_key))
    return vectorstore, chunks, markdown


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str, vectorstore, llm, top_k: int):
    hits = vectorstore.similarity_search_with_score(question, k=top_k)
    docs = [doc for doc, _ in hits]
    chain = PROMPT | llm
    answer = chain.invoke({"context": format_docs(docs), "question": question}).content
    return answer, hits


def load_questions(file) -> list[dict]:
    if not file:
        return []
    data = json.loads(file.getvalue().decode("utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    return [{"id": item.get("id"), "question": item.get("question", "")} for item in data]


def main():
    st.set_page_config(page_title="PDF RAG 问答系统", page_icon="📄", layout="wide")
    st.title("📄 基于 PDF 的 RAG 问答系统")
    st.caption("MinerU 解析 + LangChain + FAISS + 通义 DashScope + Streamlit")

    with st.sidebar:
        st.header("配置")
        top_k = st.slider("检索片段数", 1, 8, 4)
        chunk_size = st.slider("切分长度", 300, 1500, 900, 100)
        overlap = st.slider("重叠长度", 50, 300, 120, 10)
        st.write("当前模型：", CHAT_MODEL, "/", EMBED_MODEL)

    local_path = st.text_input("本地 PDF 路径", value=DEFAULT_PDF_PATH)
    page_range = st.text_input("MinerU解析页码范围（会自动按20页分批）", value="1-333")
    st.caption("例如：1-20、75-94、1-333。超过20页会自动拆分并合并知识库。")
    pdf_file = st.file_uploader("或上传 PDF 文件", type=["pdf"])
    question_file = st.file_uploader("可选：上传 questions.json", type=["json"])

    if "rag" not in st.session_state:
        st.session_state.rag = None

    can_build = pdf_file is not None or Path(local_path).exists()
    if st.button("构建 / 重建知识库", type="primary", disabled=not can_build):
        qwen_key = get_secret("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY")
        mineru_key = get_secret("MINERU_API_KEY", "MINERU_API_KEY")
        pdf_path = save_upload(pdf_file) if pdf_file else local_path
        progress_box = st.empty()
        with st.spinner(f"正在用 MinerU 分批解析 PDF 页码范围 {page_range}，请稍候..."):
            vectorstore, chunks, markdown = build_vectorstore(pdf_path, qwen_key, mineru_key, page_range, chunk_size, overlap, progress_box)
            st.session_state.rag = {"vectorstore": vectorstore, "chunks": chunks, "markdown": markdown}
        progress_box.empty()
        st.success(f"知识库构建完成，共 {len(chunks)} 个文本片段。")

    rag = st.session_state.rag
    if not rag:
        st.info("请先选择 PDF 并点击“构建 / 重建知识库”。")
        return

    llm = make_llm(get_secret("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY"))
    tab_single, tab_batch, tab_doc = st.tabs(["单问题问答", "批量问题", "解析文本"])

    with tab_single:
        question = st.text_area("输入问题", height=90)
        if st.button("开始回答") and question.strip():
            with st.spinner("正在检索并生成答案..."):
                answer, hits = answer_question(question, rag["vectorstore"], llm, top_k)
            st.subheader("答案")
            st.write(answer)
            with st.expander("查看检索片段"):
                for idx, (doc, score) in enumerate(hits, start=1):
                    st.markdown(f"**片段 {idx}｜距离 {score:.3f}**")
                    st.write(doc.page_content)

    with tab_batch:
        questions = load_questions(question_file)
        st.write(f"已读取问题数：{len(questions)}")
        if st.button("批量生成答案", disabled=not questions):
            results = []
            progress = st.progress(0)
            for i, item in enumerate(questions, start=1):
                q = item["question"]
                answer, hits = answer_question(q, rag["vectorstore"], llm, top_k)
                results.append({"id": item["id"], "question": q, "answer": answer, "contexts": [doc.page_content for doc, _ in hits]})
                progress.progress(i / len(questions))
            output = json.dumps(results, ensure_ascii=False, indent=2)
            st.success("批量回答完成。")
            st.json(results)
            st.download_button("下载 answers.json", output, "answers.json", "application/json")

    with tab_doc:
        st.download_button("下载 MinerU Markdown", rag["markdown"], "mineru_output.md", "text/markdown")
        st.text_area("Markdown 预览", rag["markdown"][:10000], height=500)


if __name__ == "__main__":
    main()
