import json
import os
import re
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
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


EMBED_MODEL = "text-embedding-v4"
CHAT_MODEL = "qwen-plus"
MINERU_BASE_URL = "https://mineru.net/api/v1/agent"
DEFAULT_PDF_PATH = r"C:\Users\bin\Desktop\作业\招股说明书2.pdf"
CACHE_DIR = Path("cache")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是PDF招股说明书多模态RAG助手。请严格依据参考资料回答。"
            "问题涉及图、图片、流程、结构、IC、芯片、设备或示意图时，优先使用图像依据；"
            "涉及表格、金额、比例、年份时，优先使用表格依据。保留原文单位和关键术语，不得编造。",
        ),
        ("human", "参考资料：\n{context}\n\n问题：{question}"),
    ]
)


def get_secret(name: str) -> str:
    value = os.getenv(name) or st.secrets.get(name, "")
    if not value:
        st.error(f"请配置 {name}。")
        st.stop()
    return value


def save_upload(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file.getvalue())
        return temp.name


def split_page_ranges(page_range: str, batch_size: int = 20) -> list[str]:
    ranges = []
    for part in page_range.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        start, end = [int(x) for x in part.split("-", 1)] if "-" in part else (int(part), int(part))
        if start <= 0 or end < start:
            raise ValueError(f"页码范围不合法：{part}")
        while start <= end:
            batch_end = min(start + batch_size - 1, end)
            ranges.append(f"{start}-{batch_end}" if start != batch_end else str(start))
            start = batch_end + 1
    return ranges or ["1-20"]


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z一-龥_-]+", "_", text)[:80]


def parse_one_range(pdf_path: str, api_key: str, page_range: str) -> str:
    payload = {
        "file_name": Path(pdf_path).name,
        "language": "ch",
        "enable_table": True,
        "enable_formula": False,
        "is_ocr": True,
        "page_range": page_range,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(f"{MINERU_BASE_URL}/parse/file", json=payload, headers=headers, timeout=30)
    data = resp.json()
    if resp.status_code != 200 or data.get("code") != 0:
        raise RuntimeError(f"MinerU创建任务失败：{data}")

    task_id = data["data"]["task_id"]
    with open(pdf_path, "rb") as file:
        upload_resp = requests.put(data["data"]["file_url"], data=file, timeout=120)
    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(f"MinerU上传文件失败：HTTP {upload_resp.status_code}")

    started = time.time()
    while time.time() - started < 900:
        result_resp = requests.get(f"{MINERU_BASE_URL}/parse/{task_id}", headers=headers, timeout=30)
        result = result_resp.json()
        state = result.get("data", {}).get("state")
        if state == "done":
            return requests.get(result["data"].get("markdown_url"), timeout=120).text
        if state == "failed":
            raise RuntimeError(f"MinerU解析失败：{result.get('data', {}).get('err_msg', result)}")
        time.sleep(3)
    raise TimeoutError(f"MinerU解析超时：{task_id}")


def load_or_parse_markdown(pdf_path: str, mineru_key: str, page_range: str, progress=None) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{safe_name(Path(pdf_path).stem)}_{safe_name(page_range)}.md"
    if cache_file.exists():
        if progress:
            progress.info(f"已命中缓存：{cache_file}")
        return cache_file.read_text(encoding="utf-8")

    parts = []
    ranges = split_page_ranges(page_range)
    for i, current_range in enumerate(ranges, start=1):
        if progress:
            progress.info(f"MinerU解析第 {i}/{len(ranges)} 批：{current_range}")
        parts.append(f"\n\n<!-- page_range: {current_range} -->\n\n" + parse_one_range(pdf_path, mineru_key, current_range))
    markdown = "\n\n".join(parts)
    cache_file.write_text(markdown, encoding="utf-8")
    return markdown


def extract_tables(markdown: str, pdf_path: str) -> list[Document]:
    tables, current = [], []
    for line in markdown.splitlines():
        if "|" in line and line.count("|") >= 2:
            current.append(line)
        elif current:
            if len(current) >= 2:
                tables.append("\n".join(current))
            current = []
    if len(current) >= 2:
        tables.append("\n".join(current))
    return [Document(page_content=f"表格{i}\n{table}", metadata={"source": pdf_path, "type": "table", "table_id": i}) for i, table in enumerate(tables, 1)]


def extract_images(markdown: str, pdf_path: str, window: int = 3) -> list[Document]:
    lines = markdown.splitlines()
    image_docs = []
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    for i, line in enumerate(lines):
        match = image_pattern.search(line)
        if not match:
            continue
        alt_text, image_path = match.groups()
        before = lines[max(0, i - window):i]
        after = lines[i + 1:i + 1 + window]
        caption = "\n".join(x for x in before + [line] + after if x.strip())
        image_docs.append(Document(
            page_content=f"图片{len(image_docs) + 1}\n图片说明：{alt_text or '无'}\n图片路径：{image_path}\n相邻上下文：\n{caption}",
            metadata={"source": pdf_path, "type": "image", "image_id": len(image_docs) + 1, "image_path": image_path},
        ))
    return image_docs


def split_text_docs(markdown: str, pdf_path: str, chunk_size: int, overlap: int) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
    docs = header_splitter.split_text(markdown)
    for doc in docs:
        doc.metadata["source"] = pdf_path
        doc.metadata["type"] = "text"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def make_embeddings(api_key: str):
    return DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=api_key)


def make_llm(api_key: str):
    return ChatTongyi(model=CHAT_MODEL, dashscope_api_key=api_key, temperature=0)


def build_indexes(pdf_path: str, qwen_key: str, mineru_key: str, page_range: str, chunk_size: int, overlap: int, progress=None):
    markdown = load_or_parse_markdown(pdf_path, mineru_key, page_range, progress)
    if not markdown.strip():
        raise ValueError("MinerU没有解析出文本。")

    text_docs = split_text_docs(markdown, pdf_path, chunk_size, overlap)
    table_docs = extract_tables(markdown, pdf_path)
    image_docs = extract_images(markdown, pdf_path)
    embeddings = make_embeddings(qwen_key)
    stores = {"text": FAISS.from_documents(text_docs, embeddings)}
    stores["table"] = FAISS.from_documents(table_docs, embeddings) if table_docs else None
    stores["image"] = FAISS.from_documents(image_docs, embeddings) if image_docs else None
    return stores, text_docs, table_docs, image_docs, markdown


def is_table_question(question: str) -> bool:
    return any(w in question for w in ["表", "金额", "收入", "利润", "成本", "费用", "资产", "负债", "比例", "报告期", "万元", "亿元"])


def is_image_question(question: str) -> bool:
    return any(w in question for w in ["图", "图片", "示意", "流程", "结构", "架构", "IC", "芯片", "2008", "设备", "外观"])


def add_unique(result: list[Document], docs: list[Document], limit: int):
    seen = {doc.page_content[:200] for doc in result}
    for doc in docs:
        key = doc.page_content[:200]
        if key not in seen:
            result.append(doc)
            seen.add(key)
        if len(result) >= limit:
            break


def retrieve(question: str, stores, top_k: int, fetch_k: int):
    result = []
    if stores.get("image") and is_image_question(question):
        add_unique(result, stores["image"].max_marginal_relevance_search(question, k=4, fetch_k=max(8, fetch_k)), top_k + 4)
    if stores.get("table") and is_table_question(question):
        add_unique(result, stores["table"].max_marginal_relevance_search(question, k=4, fetch_k=max(8, fetch_k)), top_k + 4)
    add_unique(result, stores["text"].max_marginal_relevance_search(question, k=top_k, fetch_k=max(fetch_k, top_k)), top_k + 4)
    return result


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        doc_type = {"text": "文本", "table": "表格", "image": "图像"}.get(doc.metadata.get("type"), "文本")
        title = " / ".join(str(v) for k, v in doc.metadata.items() if k.startswith("h"))
        parts.append(f"[依据{i}｜{doc_type}{'｜' + title if title else ''}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, rag, llm, top_k: int, fetch_k: int):
    docs = retrieve(question, rag["stores"], top_k, fetch_k)
    answer = (PROMPT | llm).invoke({"context": format_docs(docs), "question": question}).content
    return answer, docs


def load_questions(file) -> list[dict]:
    if not file:
        return []
    data = json.loads(file.getvalue().decode("utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    return [{"id": item.get("id"), "question": item.get("question", "")} for item in data]


def main():
    st.set_page_config(page_title="PDF图像RAG优化版", page_icon="🖼️", layout="wide")
    st.title("🖼️ PDF 文档图像内容解析及检索优化")
    st.caption("MinerU图文解析 + 文本/表格/图像三索引 + FAISS MMR + Qwen")

    with st.sidebar:
        st.header("检索配置")
        top_k = st.slider("文本检索片段数", 2, 12, 6)
        fetch_k = st.slider("候选召回片段数", 8, 50, 24)
        chunk_size = st.slider("文本切分长度", 500, 1800, 1000, 100)
        overlap = st.slider("重叠长度", 80, 400, 180, 20)
        st.write("当前模型：", CHAT_MODEL, "/", EMBED_MODEL)

    local_path = st.text_input("本地 PDF 路径", value=DEFAULT_PDF_PATH)
    page_range = st.text_input("解析页码范围（自动按20页分批并缓存）", value="1-333")
    pdf_file = st.file_uploader("或上传 2.pdf", type=["pdf"])
    question_file = st.file_uploader("上传 questions.json", type=["json"])

    if "rag" not in st.session_state:
        st.session_state.rag = None

    if st.button("构建 / 重建图像优化知识库", type="primary", disabled=not (pdf_file or Path(local_path).exists())):
        pdf_path = save_upload(pdf_file) if pdf_file else local_path
        progress_box = st.empty()
        with st.spinner("正在解析PDF图像、表格和文本并构建三索引..."):
            stores, text_docs, table_docs, image_docs, markdown = build_indexes(
                pdf_path, get_secret("DASHSCOPE_API_KEY"), get_secret("MINERU_API_KEY"), page_range, chunk_size, overlap, progress_box
            )
            st.session_state.rag = {"stores": stores, "text_docs": text_docs, "table_docs": table_docs, "image_docs": image_docs, "markdown": markdown}
        progress_box.empty()
        st.success(f"构建完成：文本 {len(text_docs)} 个，表格 {len(table_docs)} 个，图像 {len(image_docs)} 个。")

    rag = st.session_state.rag
    if not rag:
        st.info("请先构建知识库。图像类问题会优先检索图像索引。")
        return

    llm = make_llm(get_secret("DASHSCOPE_API_KEY"))
    tab_single, tab_batch, tab_images, tab_doc = st.tabs(["单问题问答", "批量评测", "图像片段", "解析文本"])

    with tab_single:
        question = st.text_area("输入问题", height=90)
        if st.button("开始回答") and question.strip():
            with st.spinner("正在检索文本、表格、图像并生成答案..."):
                answer, docs = answer_question(question, rag, llm, top_k, fetch_k)
            st.subheader("答案")
            st.write(answer)
            with st.expander("查看召回依据"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**依据 {i}｜{doc.metadata.get('type')}**")
                    st.write(doc.page_content)

    with tab_batch:
        questions = load_questions(question_file)
        st.write(f"已读取问题数：{len(questions)}")
        if st.button("批量生成图像优化答案", disabled=not questions):
            results = []
            progress = st.progress(0)
            for i, item in enumerate(questions, 1):
                answer, docs = answer_question(item["question"], rag, llm, top_k, fetch_k)
                results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "answer": answer,
                    "contexts": [{"type": doc.metadata.get("type"), "content": doc.page_content} for doc in docs],
                })
                progress.progress(i / len(questions))
            output = json.dumps(results, ensure_ascii=False, indent=2)
            st.json(results)
            st.download_button("下载 image_optimized_answers.json", output, "image_optimized_answers.json", "application/json")

    with tab_images:
        st.write(f"图像片段数：{len(rag['image_docs'])}")
        for doc in rag["image_docs"][:40]:
            with st.expander(f"图片 {doc.metadata.get('image_id')}"):
                st.write(doc.page_content)

    with tab_doc:
        st.download_button("下载 MinerU Markdown", rag["markdown"], "mineru_output.md", "text/markdown")
        st.text_area("Markdown 预览", rag["markdown"][:12000], height=500)


if __name__ == "__main__":
    main()
