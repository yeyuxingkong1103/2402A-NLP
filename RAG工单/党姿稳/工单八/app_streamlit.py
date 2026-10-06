# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
交互界面（Streamlit）：
  - 文字/语音提问（语音经 ASR 转写为文字）
  - Graph RAG 检索：展示答案所在的文本块（图谱定位 + 向量/全文混合）
  - 知识图谱结构可视化：命中子图（交互 HTML）与全局图谱（静态图/交互 HTML）
"""
import os
import sys
import socket
import streamlit as st
import streamlit.components.v1 as components

from kb import load_chunks
from graph_builder import load_graph
from graph_retriever import GraphRetriever
from hybrid_retriever import HybridRetriever
from graph_viz import render_html, render_png
from rag_chain import call_llm, build_rag_prompt, RAG_SYSTEM_PROMPT
from config import GRAPH_FILE, GRAPH_HTML, GRAPH_IMG, EVAL_QUESTIONS
from asr import transcribe

st.set_page_config(page_title="Graph RAG 金融问答", page_icon="🕸️", layout="wide")


@st.cache_resource(show_spinner="正在加载语料、召回索引与知识图谱...")
def get_resources():
    chunks = load_chunks()
    gdata = load_graph()
    hybrid = HybridRetriever(chunks, reranker="none")
    return chunks, gdata, GraphRetriever(chunks, gdata, hybrid)


def find_free_port(start=8501, end=8600):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("🕸️ 金融年报 Graph RAG 问答系统（工单八）")
st.caption("工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答　|　语料：ccf_competition 金融年报")

chunks, gdata, gret = get_resources()
n_doc = len({c["doc"] for c in chunks})
st.success(f"就绪：{len(chunks)} 个文本块 / {n_doc} 份年报；图谱节点 {len(gdata['nodes'])}、关系 {len(gdata['edges'])}")

with st.sidebar:
    st.header("⚙️ 设置")
    top_k = st.slider("返回文本块 Top-K", 1, 10, 5)
    hop = st.slider("图谱扩展跳数", 0, 2, 1)
    gw = st.slider("图谱权重", 0.0, 1.0, 0.5, 0.1)
    st.markdown("---")
    st.subheader("验收问题（eval_question.md）")
    for q in EVAL_QUESTIONS:
        st.caption(f"{q['id']}. {q['question'][:40]}…")

tab1, tab2 = st.tabs(["💬 问答（文字/语音）", "🕸️ 知识图谱"])

with tab1:
    quick = st.selectbox("⚡ 选择验收问题", [""] + [q["question"] for q in EVAL_QUESTIONS])
    mode = st.radio("输入方式", ["文字", "语音"], horizontal=True)
    question = ""
    if mode == "文字":
        question = st.text_input("请输入问题（中英文均可）", value=quick)
    else:
        try:
            audio = st.audio_input("🎤 点击录音，说出你的问题")
            if audio is not None:
                with st.spinner("语音识别中..."):
                    question = transcribe(audio.read()) or ""
                st.write(f"识别结果：{question or '（识别失败，请改用文字输入）'}")
        except Exception:
            st.warning("当前 Streamlit 版本不支持 audio_input，请改用文字输入。")

    if st.button("🚀 检索并回答", type="primary") and question:
        gret.hop, gret.graph_weight = hop, gw
        results, gstruct = gret.search(question, top_k=top_k)
        contexts = [(c, s) for c, s, _src, _g in results]
        st.info(call_llm(RAG_SYSTEM_PROMPT, build_rag_prompt(question, contexts)))

        st.subheader("📚 答案所在的文本块")
        for i, (c, s, src, gs) in enumerate(results, 1):
            with st.expander(f"块{i}｜{c['doc']}·第{c['page']}页｜分数{s:.3f}｜来源：{src}"
                             + (f"（图谱分数 {gs:.2f}）" if gs else "")):
                st.markdown(c["text"][:900])

        st.subheader("🔗 解析出的知识图谱结构（命中子图）")
        st.caption(f"命中实体：{'、'.join(gstruct['seeds']) or '（无直接命中）'}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**实体**")
            st.dataframe([{"实体": n["id"], "类型": n["type"]} for n in gstruct["nodes"]], height=260)
        with c2:
            st.markdown("**关系**")
            st.dataframe([{"头实体": e["source"], "关系": e["relation"], "尾实体": e["target"]}
                          for e in gstruct["edges"]], height=260)
        if gstruct["nodes"]:
            sub_html = os.path.join(os.path.dirname(GRAPH_FILE), "subgraph.html")
            render_html(gstruct, sub_html, max_nodes=80)
            components.html(open(sub_html, encoding="utf-8").read(), height=460)

with tab2:
    st.subheader("全局知识图谱（实体-关系）")
    st.caption("节点颜色代表实体类型：红=公司，蓝=指标，绿=数值，紫=人物，橙=职位，青=策略/领域")
    if st.button("🔄 重新渲染图谱"):
        render_png(gdata, GRAPH_IMG)
        render_html(gdata, GRAPH_HTML)
    if os.path.exists(GRAPH_IMG):
        st.image(GRAPH_IMG, caption="静态图谱（networkx + matplotlib）")
    if os.path.exists(GRAPH_HTML):
        components.html(open(GRAPH_HTML, encoding="utf-8").read(), height=620)
    else:
        st.info("请先运行 graph_builder.py 与 graph_viz.py 生成图谱。")

if __name__ == "__main__" and not st.runtime.exists():
    # `streamlit run` 与 `python app_streamlit.py` 都会把脚本以 __name__="__main__" 执行；
    # 用"Runtime 是否已存在"判断，可避免启动器递归调用导致的 Runtime instance already exists
    import streamlit.runtime
    from streamlit.web import cli as stcli
    port = find_free_port()
    print(f"启动地址: http://localhost:{port}")
    sys.argv = ["streamlit", "run", __file__, "--server.port", str(port)]
    sys.exit(stcli.main())
