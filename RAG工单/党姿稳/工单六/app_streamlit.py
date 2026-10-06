# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
交互界面（Streamlit）：可切换 向量检索 / 全文检索 / 混合检索 三种策略，
可配置嵌入模型、重排算法（LLM / TF-IDF / 用户反馈）、融合算法与权重，并支持 👍/👎 用户反馈。
"""
import os
import socket
import streamlit as st

from kb import load_all_chunks
from hybrid_retriever import HybridRetriever
from rag_chain import call_llm, build_rag_prompt, RAG_SYSTEM_PROMPT
from config import EVAL_QUESTIONS, EMBED_MODELS

st.set_page_config(page_title="RAG混合检索问答系统", page_icon="🔀", layout="wide")


@st.cache_resource(show_spinner="正在加载知识库与向量索引...")
def get_retriever():
    allc, _, _ = load_all_chunks()
    return HybridRetriever(allc)


def find_free_port(start=8501, end=8600):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("🔀 基于PDF文档的问答系统（工单六·混合检索）")
st.caption("工单编号：人工智能NLP-RAG-混合检索任务")

with st.sidebar:
    st.header("⚙️ 检索策略配置")
    strategy = st.radio("检索策略", ["混合检索", "向量检索（召回+重排）", "全文检索"],
                        help="混合检索 = 向量检索 + 全文检索 同时执行后融合")
    embed_model = st.selectbox("嵌入模型", ["m3e"] + [k for k in EMBED_MODELS if k != "m3e"])
    reranker = st.selectbox("重排算法", ["llm", "tfidf", "feedback", "none"],
                            format_func=lambda x: {"llm": "LLM重排器", "tfidf": "TF-IDF重排器",
                                                   "feedback": "用户反馈自适应重排器",
                                                   "none": "不重排"}[x])
    ft_mode = st.selectbox("全文检索模式", ["auto", "boolean", "phrase", "fuzzy"],
                           format_func=lambda x: {"auto": "自动（布尔/短语）", "boolean": "布尔查询",
                                                  "phrase": "短语匹配", "fuzzy": "模糊匹配"}[x])
    fusion = st.radio("融合算法", ["weighted", "vote"],
                      format_func=lambda x: "加权平均" if x == "weighted" else "投票(RRF)")
    vector_weight = st.slider("向量检索权重", 0.0, 1.0, 0.6, 0.05)
    top_k = st.slider("返回片段数 Top-K", 1, 10, 5)
    st.caption("全文检索支持：`A AND B`、`A OR B`、`NOT A`、`\"短语\"`、`词~`（模糊）")

retriever = get_retriever()
retriever.reranker_name = reranker
from rerankers import build_reranker
retriever.reranker = build_reranker(reranker)
st.success(f"知识库就绪：{len(retriever.chunks)} 个文档块；当前策略：{strategy}")

quick = st.selectbox("⚡ 验收问题（可直接选用）", [""] + [q["question"] for q in EVAL_QUESTIONS])
question = st.text_input("请输入问题（中英文均可）", value=quick)

if st.button("🚀 检索并回答", type="primary") and question:
    mode = {"混合检索": "hybrid", "向量检索（召回+重排）": "vector",
            "全文检索": "fulltext"}[strategy]
    kwargs = {} if mode == "vector" else {"ft_mode": ft_mode}
    if mode == "hybrid":
        kwargs.update(fusion=fusion, vector_weight=vector_weight,
                      fulltext_weight=round(1 - vector_weight, 2))
    results = retriever.search(question, strategy=mode, top_k=top_k, **kwargs)

    import time
    t0 = time.time()
    answer = call_llm(RAG_SYSTEM_PROMPT, build_rag_prompt(question, results))
    st.info(answer)
    st.caption(f"检索 + 生成耗时：{time.time()-t0:.2f}s")

    st.subheader("📚 检索片段（可反馈，用于自适应重排）")
    from rerankers import FeedbackReranker
    store = retriever.reranker if isinstance(retriever.reranker, FeedbackReranker) else FeedbackReranker()
    for i, (c, s) in enumerate(results, 1):
        tag = "🖼️ 图像块" if c.get("type") == "image" else ("📊 表格块" if c.get("type") == "table" else "📄 文本块")
        with st.expander(f"片段{i}｜{tag}｜{c['doc']}·第{c['page']}页｜分数{s:.4f}"):
            st.markdown(c["text"][:800])
            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button("👍 有用", key=f"up{i}"):
                store.record(c, True)
                st.success("已记录有用反馈")
            if c2.button("👎 无用", key=f"down{i}"):
                store.record(c, False)
                st.warning("已记录无用反馈")
    st.caption(f"累计反馈：{store.stats()}")

with st.expander("🔎 检索技术说明"):
    st.markdown("""
- **向量检索（召回+重排）**：m3e/bge 向量嵌入 → 余弦相似度召回 Top-N → 重排器（LLM / TF-IDF / 用户反馈自适应）优化排序。
- **全文检索**：多字段（标题/摘要/正文）倒排索引 + BM25 打分，支持布尔（AND/OR/NOT）、短语（"..."）、模糊（词~）匹配。
- **混合检索**：同时执行向量与全文检索，按加权平均或投票(RRF)融合；权重可实时调整。
- **用户反馈**：👍/👎 结果会被持久化，反馈自适应重排器据此调整后续排序（越用越准）。
""")

if __name__ == "__main__" and not st.runtime.exists():
    # `streamlit run` 与 `python app_streamlit.py` 都会把脚本以 __name__="__main__" 执行；
    # 用"Runtime 是否已存在"判断，可避免启动器递归调用导致的 Runtime instance already exists
    import streamlit.runtime
    from streamlit.web import cli as stcli
    import sys
    port = find_free_port()
    print(f"启动地址: http://localhost:{port}")
    sys.argv = ["streamlit", "run", __file__, "--server.port", str(port)]
    sys.exit(stcli.main())
