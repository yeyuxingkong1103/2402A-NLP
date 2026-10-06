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

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是PDF招股说明书RAG助手。请严格依据参考资料回答，不得编造。表格问题保留数值单位，图像问题优先依据图像上下文。"),
        ("human", "原始问题：{question}\nQuery理解：{analysis}\n参考资料：\n{context}\n\n请给出答案："),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你负责优化RAG检索Query。输出JSON，不要解释。字段：type、rewrite、sub_queries。type只能是text/table/image/mixed。sub_queries给3到5个中文检索问题。"),
        ("human", "用户问题：{question}"),
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
        while start <= end:
            batch_end = min(start + batch_size - 1, end)
            ranges.append(f"{start}-{batch_end}" if start != batch_end else str(start))
            start = batch_end + 1
    return ranges or ["1-20"]


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z一-龥_-]+", "_", text)[:80]


def parse_one_range(pdf_path: str, api_key: str, page_range: str) -> str:
    payload = {"file_name": Path(pdf_path).name, "language": "ch", "enable_table": True, "enable_formula": False, "is_ocr": True, "page_range": page_range}
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
        result = requests.get(f"{MINERU_BASE_URL}/parse/{task_id}", headers=headers, timeout=30).json()
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
    for i, current_range in enumerate(split_page_ranges(page_range), 1):
        if progress:
            progress.info(f"MinerU解析第 {i} 批：{current_range}")
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
    return [Document(page_content=f"表格{i}\n{table}", metadata={"type": "table", "source": pdf_path, "table_id": i}) for i, table in enumerate(tables, 1)]


def extract_images(markdown: str, pdf_path: str, window: int = 3) -> list[Document]:
    lines = markdown.splitlines()
    docs = []
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        alt, path = match.groups()
        context = "\n".join(x for x in lines[max(0, i - window): i + window + 1] if x.strip())
        docs.append(Document(page_content=f"图片{len(docs)+1}\n图片说明：{alt or '无'}\n图片路径：{path}\n相邻上下文：\n{context}", metadata={"type": "image", "source": pdf_path, "image_id": len(docs)+1}))
    return docs


def split_text_docs(markdown: str, pdf_path: str, chunk_size: int, overlap: int) -> list[Document]:
    docs = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]).split_text(markdown)
    for doc in docs:
        doc.metadata.update({"type": "text", "source": pdf_path})
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""])
    return splitter.split_documents(docs)


def make_embeddings(api_key: str):
    return DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=api_key)


def make_llm(api_key: str):
    return ChatTongyi(model=CHAT_MODEL, dashscope_api_key=api_key, temperature=0)


def build_indexes(pdf_path: str, qwen_key: str, mineru_key: str, page_range: str, chunk_size: int, overlap: int, progress=None):
    markdown = load_or_parse_markdown(pdf_path, mineru_key, page_range, progress)
    text_docs = split_text_docs(markdown, pdf_path, chunk_size, overlap)
    table_docs = extract_tables(markdown, pdf_path)
    image_docs = extract_images(markdown, pdf_path)
    embeddings = make_embeddings(qwen_key)
    stores = {"text": FAISS.from_documents(text_docs, embeddings)}
    stores["table"] = FAISS.from_documents(table_docs, embeddings) if table_docs else None
    stores["image"] = FAISS.from_documents(image_docs, embeddings) if image_docs else None
    return stores, text_docs, table_docs, image_docs, markdown


def keyword_type(question: str) -> str:
    table_words = ["表", "金额", "收入", "利润", "成本", "费用", "资产", "负债", "比例", "报告期", "万元", "亿元"]
    image_words = ["图", "图片", "示意", "流程", "结构", "架构", "IC", "芯片", "设备", "外观", "2008"]
    has_table = any(w in question for w in table_words)
    has_image = any(w in question for w in image_words)
    if has_table and has_image:
        return "mixed"
    if has_table:
        return "table"
    if has_image:
        return "image"
    return "text"


def parse_json_from_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("没有JSON")
    return json.loads(match.group())


def analyze_query(question: str, llm) -> dict:
    fallback = {
        "type": keyword_type(question),
        "rewrite": question,
        "sub_queries": [question],
    }
    try:
        content = (REWRITE_PROMPT | llm).invoke({"question": question}).content
        data = parse_json_from_text(content)
        qtype = data.get("type") if data.get("type") in {"text", "table", "image", "mixed"} else fallback["type"]
        rewrite = data.get("rewrite") or question
        subs = [x for x in data.get("sub_queries", []) if isinstance(x, str) and x.strip()]
        return {"type": qtype, "rewrite": rewrite, "sub_queries": subs[:5] or [rewrite]}
    except Exception:
        return fallback


def add_unique(result: list[Document], docs: list[Document], limit: int):
    seen = {doc.page_content[:200] for doc in result}
    for doc in docs:
        key = doc.page_content[:200]
        if key not in seen:
            result.append(doc)
            seen.add(key)
        if len(result) >= limit:
            break


def search_store(store, query: str, k: int, fetch_k: int) -> list[Document]:
    if not store:
        return []
    return store.max_marginal_relevance_search(query, k=k, fetch_k=max(fetch_k, k))


def retrieve_with_analysis(analysis: dict, stores, top_k: int, fetch_k: int) -> list[Document]:
    queries = [analysis["rewrite"], *analysis["sub_queries"]]
    result = []
    for query in queries:
        if analysis["type"] in {"image", "mixed"}:
            add_unique(result, search_store(stores.get("image"), query, 3, fetch_k), top_k + 6)
        if analysis["type"] in {"table", "mixed"}:
            add_unique(result, search_store(stores.get("table"), query, 3, fetch_k), top_k + 6)
        add_unique(result, search_store(stores["text"], query, top_k, fetch_k), top_k + 6)
        if len(result) >= top_k + 6:
            break
    return result


def format_docs(docs: list[Document]) -> str:
    names = {"text": "文本", "table": "表格", "image": "图像"}
    return "\n\n---\n\n".join(f"[依据{i}｜{names.get(doc.metadata.get('type'), '文本')}]\n{doc.page_content}" for i, doc in enumerate(docs, 1))


def answer_question(question: str, rag, llm, top_k: int, fetch_k: int):
    analysis = analyze_query(question, llm)
    docs = retrieve_with_analysis(analysis, rag["stores"], top_k, fetch_k)
    answer = (ANSWER_PROMPT | llm).invoke({"question": question, "analysis": json.dumps(analysis, ensure_ascii=False), "context": format_docs(docs)}).content
    return answer, docs, analysis


def load_questions(file) -> list[dict]:
    if not file:
        return []
    data = json.loads(file.getvalue().decode("utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    return [{"id": item.get("id"), "question": item.get("question", "")} for item in data]


def main():
    st.set_page_config(page_title="Query理解优化RAG", page_icon="🧠", layout="wide")
    st.title("🧠 Query 理解优化 RAG 系统")
    st.caption("Query分类/改写/扩展 + 文本/表格/图像多路检索融合 + Qwen")

    with st.sidebar:
        st.header("检索配置")
        top_k = st.slider("基础检索片段数", 2, 12, 6)
        fetch_k = st.slider("候选召回片段数", 8, 50, 24)
        chunk_size = st.slider("文本切分长度", 500, 1800, 1000, 100)
        overlap = st.slider("重叠长度", 80, 400, 180, 20)

    local_path = st.text_input("本地 PDF 路径", value=DEFAULT_PDF_PATH)
    page_range = st.text_input("解析页码范围（自动按20页分批并缓存）", value="1-333")
    pdf_file = st.file_uploader("或上传 PDF", type=["pdf"])
    question_file = st.file_uploader("上传 questions.json", type=["json"])

    if "rag" not in st.session_state:
        st.session_state.rag = None

    if st.button("构建 / 重建 Query 优化知识库", type="primary", disabled=not (pdf_file or Path(local_path).exists())):
        pdf_path = save_upload(pdf_file) if pdf_file else local_path
        progress_box = st.empty()
        with st.spinner("正在构建文本/表格/图像索引..."):
            stores, text_docs, table_docs, image_docs, markdown = build_indexes(pdf_path, get_secret("DASHSCOPE_API_KEY"), get_secret("MINERU_API_KEY"), page_range, chunk_size, overlap, progress_box)
            st.session_state.rag = {"stores": stores, "text_docs": text_docs, "table_docs": table_docs, "image_docs": image_docs, "markdown": markdown}
        progress_box.empty()
        st.success(f"构建完成：文本 {len(text_docs)}，表格 {len(table_docs)}，图像 {len(image_docs)}。")

    rag = st.session_state.rag
    if not rag:
        st.info("请先构建知识库。")
        return

    llm = make_llm(get_secret("DASHSCOPE_API_KEY"))
    tab_single, tab_batch, tab_query, tab_doc = st.tabs(["单问题问答", "批量评测", "Query理解", "解析文本"])

    with tab_single:
        question = st.text_area("输入问题", height=90)
        if st.button("开始回答") and question.strip():
            with st.spinner("正在理解Query、多路检索并生成答案..."):
                answer, docs, analysis = answer_question(question, rag, llm, top_k, fetch_k)
                st.session_state.last_query = analysis
            st.subheader("答案")
            st.write(answer)
            st.subheader("Query理解")
            st.json(analysis)
            with st.expander("查看召回依据"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**依据 {i}｜{doc.metadata.get('type')}**")
                    st.write(doc.page_content)

    with tab_batch:
        questions = load_questions(question_file)
        st.write(f"已读取问题数：{len(questions)}")
        if st.button("批量生成Query优化答案", disabled=not questions):
            results = []
            progress = st.progress(0)
            for i, item in enumerate(questions, 1):
                answer, docs, analysis = answer_question(item["question"], rag, llm, top_k, fetch_k)
                results.append({"id": item["id"], "question": item["question"], "query_analysis": analysis, "answer": answer, "contexts": [{"type": d.metadata.get("type"), "content": d.page_content} for d in docs]})
                progress.progress(i / len(questions))
            output = json.dumps(results, ensure_ascii=False, indent=2)
            st.json(results)
            st.download_button("下载 query_optimized_answers.json", output, "query_optimized_answers.json", "application/json")

    with tab_query:
        demo = st.text_input("输入一个问题查看Query理解", value="这个公司赚钱吗？")
        if st.button("分析Query"):
            st.json(analyze_query(demo, llm))

    with tab_doc:
        st.download_button("下载 MinerU Markdown", rag["markdown"], "mineru_output.md", "text/markdown")
        st.text_area("Markdown 预览", rag["markdown"][:12000], height=500)


if __name__ == "__main__":
    main()
