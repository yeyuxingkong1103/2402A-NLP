"""工单编号：人工智能NLP-RAG-混合检索任务。"""

import json
import os
import time
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from rag import INDEX_DIR, ROOT, answer_verified, build_index, index_key, load_index, model
from conversation import resolve_query
from hybrid_retrieval import search


UPLOAD_DIR = ROOT / "data" / "uploads"
TASK_PDF = Path(r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书2.pdf")
LEGACY_PDF = Path(r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书1.pdf")
FEEDBACK = ROOT / "data" / "feedback.jsonl"

st.set_page_config(page_title="PDF 文档问答", page_icon="📄", layout="wide")
st.title("PDF 文档问答")
st.caption("上传或选择 PDF，建库后提问。答案显示原文证据与 PDF 页码。")


@st.cache_resource
def get_model(name):
    return model(name)


@st.cache_resource
def get_index(folder):
    return load_index(folder)


with st.sidebar:
    st.header("知识库")
    uploaded_files = st.file_uploader("上传 PDF（可同时上传两份招股书）", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            pdf_path = UPLOAD_DIR / Path(uploaded.name).name
            if not pdf_path.exists() or pdf_path.stat().st_size != uploaded.size:
                pdf_path.write_bytes(uploaded.getbuffer())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        pdf_files = [path for path in [TASK_PDF, LEGACY_PDF] if path.exists()]
    pdf_path = None
    if pdf_files:
        default_choice = next((i for i, path in enumerate(pdf_files) if "招股说明书2" in path.name), 0)
        pdf_path = st.selectbox("选择当前 PDF", pdf_files, index=default_choice, format_func=lambda path: path.name)
    else:
        st.info("请上传《招股说明书2.pdf》或其他 PDF。")
        pdf_path = UPLOAD_DIR / "招股说明书2.pdf"
    st.write(f"当前文档：{pdf_path.name}")
    st.header("嵌入模型")
    embedding_name = st.selectbox("嵌入模型", ["moka-ai/m3e-small", "BAAI/bge-small-zh-v1.5"])
    if st.button("解析并建立索引", disabled=not pdf_path.exists()):
        bar = st.progress(0)
        status = st.empty()
        try:
            def update(done, total):
                bar.progress(done / total)
                status.write(f"正在向量化：{done}/{total}")

            folder = build_index(pdf_path, get_model(embedding_name), update, embedding_name)
            get_index.clear()
            st.success(f"建库完成：{folder.name}")
        except Exception as exc:
            st.error(f"建库失败：{exc}")

    folders = sorted(INDEX_DIR.glob("*/source.json")) if INDEX_DIR.exists() else []
    sources = [(json.loads(p.read_text(encoding="utf-8"))["name"], p.parent) for p in folders]
    if sources:
        labels = [f"{name} · {folder.name}" for name, folder in sources]
        current = index_key(pdf_path, embedding_name) if pdf_path.exists() else ""
        default = next((i for i, (_, folder) in enumerate(sources) if folder.name == current), 0)
        choice = st.selectbox("已建知识库", range(len(sources)), index=default, format_func=lambda i: labels[i])
        selected_folder = sources[choice][1]
        source = json.loads((selected_folder / "source.json").read_text(encoding="utf-8"))
        if source.get("embedding_model", "moka-ai/m3e-small") != embedding_name:
            st.warning("所选知识库与嵌入模型不同，请按当前模型重新建库。")
            selected_folder = None
        else:
            st.caption(f"{source['pages']} 页 · {source['chunks']} 个片段 · {source.get('images', 0)} 张图表")
    else:
        selected_folder = None
        st.info("请先建立索引。")

    st.header("生成模型")
    model_name = st.text_input("模型名称", value=os.getenv("OPENAI_MODEL", "deepseek-r1:1.5b"))
    base_url = st.text_input("兼容 API 地址", value="http://127.0.0.1:11434/v1")
    api_key = st.text_input("API Key（远程模型才需要）", type="password")

    st.header("检索策略")
    strategy = st.selectbox("检索方式", ["混合检索", "向量检索", "全文检索"])
    vector_weight = st.slider("向量权重", 0.0, 1.0, 0.6, 0.05)
    fusion = st.selectbox("融合算法", ["加权平均", "RRF投票", "排序投票"])
    reranker = st.selectbox("重排算法", ["TF-IDF", "LLM重排", "用户反馈"])
    search_field = st.selectbox("全文字段", ["全部字段", "title", "summary", "body"])
    fuzzy = st.checkbox("全文模糊匹配", value=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if st.session_state["chat_history"]:
    st.subheader("本轮对话")
    for item in st.session_state["chat_history"]:
        st.markdown(f"**你：** {item['question']}")
        st.markdown(f"**回答：** {item['answer']}")
    if st.button("清空对话"):
        st.session_state["chat_history"] = []
        st.rerun()

if "voice_question" in st.session_state:
    st.session_state["question_input"] = st.session_state.pop("voice_question")
st.text_area("输入问题（中文或英文）", placeholder="例如：报告期内，公司来自军用领域的收入分别是多少？", height=90, key="question_input")
question = st.session_state["question_input"]
resolved_question = resolve_query(question, st.session_state["chat_history"])

if hasattr(st, "audio_input"):
    audio = st.audio_input("语音提问（可选）")
    if audio and st.button("识别语音"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key="ollama" if "127.0.0.1:11434" in base_url else api_key, base_url=base_url)
            transcript = client.audio.transcriptions.create(model=os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"), file=("question.wav", audio.getvalue(), "audio/wav"))
            st.session_state["voice_question"] = transcript.text
            st.rerun()
        except Exception as exc:
            st.error(f"语音识别失败：{exc}")
else:
    st.caption("语音输入需要 Streamlit 1.44 或更新版本。")

if st.button("提问", type="primary", disabled=selected_folder is None):
    st.session_state.pop("last_answer", None)
    if not question.strip():
        st.warning("请输入问题。")
    else:
        try:
            index = get_index(str(selected_folder))
            start = time.perf_counter()
            info, results, search_info = search(
                resolved_question, index, get_model(embedding_name), strategy, vector_weight,
                fusion, reranker, search_field, fuzzy, model_name, base_url, api_key,
            )
            if "error" in info:
                st.warning(info["error"])
            else:
                st.caption(f"{strategy} · {fusion} · {reranker} · 检索用时：{time.perf_counter() - start:.2f} 秒")
                st.caption(f"向量召回 {search_info['vector_candidates']} 条 · 全文召回 {search_info['fulltext_candidates']} 条")
                if search_info["warning"]:
                    st.warning(search_info["warning"])
                if len(info["parts"]) > 1:
                    st.caption(f"已拆分为 {len(info['parts'])} 个子问题")
                    answers = []
                    raw_answers = []
                    evidence = {}
                    generate_seconds = 0
                    for part in info["parts"]:
                        _, part_results, _ = search(part, index, get_model(embedding_name), strategy,
                                                    vector_weight, fusion, reranker, search_field, fuzzy,
                                                    model_name, base_url, api_key)
                        if not part_results or part_results[0]["score"] <= 0:
                            answers.append(f"{part}：文档中没有找到足够证据。")
                            continue
                        part_answer, seconds, part_raw = answer_verified(part, part_results, model_name, base_url, api_key, index)
                        answers.append(f"{part}：{part_answer}")
                        raw_answers.append(part_raw)
                        generate_seconds += seconds
                        for item in part_results:
                            evidence[item["page"]] = item
                    results = list(evidence.values())
                    response = "\n\n".join(answers)
                    raw = "\n\n".join(raw_answers)
                elif results and results[0]["score"] > 0:
                    response, generate_seconds, raw = answer_verified(question, results, model_name, base_url, api_key, index)
                if not results or results[0]["score"] <= 0:
                    st.warning("未找到足够相关的文档片段，请换一种问法。")
                else:
                    total_seconds = time.perf_counter() - start
                    st.subheader("回答")
                    st.write(response)
                    if response != raw:
                        st.caption("模型表述与原文可能不一致，已按 PDF 原文校正。")
                    st.caption(f"总用时：{total_seconds:.2f} 秒（生成 {generate_seconds:.2f} 秒）")
                    st.caption(f"检索改写：{resolved_question}")
                    st.session_state["last_answer"] = {"question": question, "resolved_question": resolved_question,
                                                        "answer": response, "pages": [r["page"] for r in results]}
                    st.session_state["chat_history"].append({"question": question, "resolved_question": resolved_question,
                                                             "answer": response})
                    st.subheader("原文证据")
                    for item in results:
                        with st.expander(f"PDF 第 {item['page']} 页 · 相关度 {item['score']:.2f}"):
                            st.text(item["text"])
        except Exception as exc:
            st.error(f"问答失败：{exc}")

if "last_answer" in st.session_state:
    st.subheader("反馈")
    rating = st.radio("这次回答是否有帮助？", ["有帮助", "无帮助"], horizontal=True)
    note = st.text_input("补充意见（可选）")
    if st.button("提交反馈"):
        FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        record = {**st.session_state["last_answer"], "rating": rating, "note": note}
        with FEEDBACK.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success("反馈已保存。")


