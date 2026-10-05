"""工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答。"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "工单7"))
from run import load, rank, read_json, save_json, evaluate, build, DATA

# 同义词只帮助识别实体，不存测试题的答案。
TOPICS = {
    "盈利增长": ["盈利", "营收", "利润", "profit", "revenue"],
    "业务结构": ["零售", "对公", "业务结构", "retail"],
    "资产质量": ["资产质量", "不良贷款率", "asset quality", "npl"],
    "拨备覆盖率": ["拨备", "provision coverage"],
    "资本充足率": ["资本充足率", "capital adequacy"],
    "金融科技": ["科技", "数字化", "fintech"],
    "绿色金融": ["绿色", "新能源", "环保", "green finance"],
    "创新商业模式": ["创新商业模式", "逾越者联盟", "咖啡零售", "business model"],
    "偿付能力": ["偿付能力", "solvency"],
    "投资收益": ["投资收益", "investment return"],
    "风险管理": ["风险管理", "抵御风险", "risk management"],
    "ESG": ["ESG", "可持续", "sustainability"],
}
ALIASES = {"平安银行": "ping an bank", "招商银行": "china merchants bank",
           "邮储银行": "postal savings bank", "中国平安": "ping an insurance",
           "中国人寿": "china life", "中国太保": "china pacific insurance",
           "中信证券": "citic securities", "招商证券": "china merchants securities",
           "国泰君安": "guotai junan"}


def make_graph(chunks):
    nodes, edges = {}, {}
    for chunk in chunks:
        company, year = chunk["company"], chunk["year"]
        report = company + "·" + year
        nodes[company] = {"id": company, "type": "机构"}
        nodes[report] = {"id": report, "type": "年报"}
        edges.setdefault((company, report, "发布"), []).append(chunk["id"])
        for topic, words in TOPICS.items():
            if any(word.lower() in chunk["text"].lower() for word in words):
                nodes[topic] = {"id": topic, "type": "主题"}
                edges.setdefault((report, topic, "涉及"), []).append(chunk["id"])
    relations = [{"source": a, "target": b, "relation": r, "chunk_ids": ids}
                 for (a, b, r), ids in edges.items()]
    graph = {"nodes": list(nodes.values()), "edges": relations,
             "note": "发布关系来自文件名；涉及关系表示文本提及，不表示因果或投资关系。"}
    save_json(ROOT / "data/graph.json", graph)
    return graph


def translate(question):
    result = question.lower()
    for company, english in ALIASES.items():
        result = result.replace(english, company)
    for topic, words in TOPICS.items():
        for word in words:
            if word.isascii():
                result = result.replace(word.lower(), topic)
    return result


def graph_search(question, index, top_k=5):
    query = translate(question)
    graph = index["graph"]
    companies = [name for name in ALIASES if name in query]
    topics = [topic for topic, words in TOPICS.items()
              if topic.lower() in query or any(w.lower() in query for w in words)]
    # 从命中主题反向走到年报，再走到对应文本块；保留来源证据。
    candidates = set()
    for edge in graph["edges"]:
        if edge["relation"] != "涉及" or edge["target"] not in topics:
            continue
        if companies and not any(edge["source"].startswith(c + "·") for c in companies):
            continue
        candidates.update(edge["chunk_ids"])
    vector = index["model"].encode([query], normalize_embeddings=True)[0]
    dense = index["vectors"] @ vector
    lexical = (index["matrix"] @ index["tfidf"].transform([query]).T).toarray().ravel()
    scores = 0.65 * dense + 0.35 * lexical
    if candidates:
        scores += np.array([0.15 if c["id"] in candidates else 0 for c in index["chunks"]])
    return rank(scores, index["chunks"], top_k)


def graph_dot(graph, company):
    edges = [e for e in graph["edges"] if e["source"] == company or e["source"].startswith(company + "·")]
    lines = ['digraph G { rankdir=LR; node [style=filled, fillcolor="#dbeafe", fontname="Microsoft YaHei"];']
    for edge in edges:
        a, b = json.dumps(edge["source"], ensure_ascii=False), json.dumps(edge["target"], ensure_ascii=False)
        lines.append(f'{a} -> {b} [label="{edge["relation"]} ({len(edge["chunk_ids"])}块)"];')
    return "\n".join(lines) + "\n}"


def answer(question, hits, model_name):
    context = "\n\n".join(f'[{i}] {h["company"]} {h["year"]} PDF第{h["page"]}页\n{h["text"]}'
                            for i, h in enumerate(hits, 1))
    prompt = ("仅根据证据回答，使用问题的语言，引用[编号]。缺少证据就明确说明，勿猜测数字。\n"
              + context + "\n问题：" + question)
    if not re.search(r"[\u4e00-\u9fff]", question):
        prompt += "\nYou must answer in English and cite the evidence numbers."
    body = json.dumps({"model": model_name, "stream": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0, "num_predict": 1200, "num_ctx": 8192}}).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read())["message"]["content"]


def page():
    import streamlit as st
    import streamlit.components.v1 as components
    st.set_page_config(page_title="金融图谱问答", layout="wide")
    st.title("金融图谱问答 · 工单8")
    st.caption("金融年报 · 实体关系图谱 · 证据页码 · 与工单7对比")
    @st.cache_resource
    def prepare():
        index = load()
        index["graph"] = make_graph(index["chunks"])
        return index
    index = prepare()
    graph = index["graph"]
    company = st.sidebar.selectbox("查看机构", [n["id"] for n in graph["nodes"] if n["type"] == "机构"])
    st.sidebar.metric("图谱实体", len(graph["nodes"]))
    st.sidebar.metric("图谱关系", len(graph["edges"]))
    model_name = st.sidebar.text_input("Ollama模型", "deepseek-r1:7b")
    uploads = st.sidebar.file_uploader("添加PDF后重建索引", type="pdf", accept_multiple_files=True)
    if uploads and st.sidebar.button("保存并重建索引"):
        folder = ROOT / "uploads"
        folder.mkdir(exist_ok=True)
        for file in uploads:
            (folder / Path(file.name).name).write_bytes(file.getvalue())
        paths = [Path(s["path"]) if Path(s["path"]).is_absolute() else DATA / s["path"]
                 for s in read_json(DATA / "sources.json")]
        with st.spinner("正在解析并重建索引，请稍等"):
            build(list(set(paths + list(folder.glob("*.pdf")))))
        st.cache_resource.clear()
        st.rerun()
    tab1, tab2, tab3 = st.tabs(["问答", "知识图谱", "评估对比"])
    with tab1:
        components.html('''<button onclick="start()">语音转文字</button>
          <select id="lang"><option value="zh-CN">中文</option><option value="en-US">English</option></select>
          <p>识别后复制文字到下方提问框。</p><textarea id="voice" style="width:95%"></textarea>
          <script>function start(){let R=window.SpeechRecognition||window.webkitSpeechRecognition;
          if(!R){document.getElementById('voice').value='浏览器不支持语音识别，请输入文字';return;}
          let r=new R();r.lang=document.getElementById('lang').value;
          r.onresult=e=>document.getElementById('voice').value=e.results[0][0].transcript;
          r.onerror=e=>document.getElementById('voice').value='语音识别失败：'+e.error;r.start();}</script>''', height=150)
        question = st.text_area("输入中文或英文问题", "招商银行2019年探索了哪些创新商业模式？")
        generate = st.checkbox("调用模型生成答案（需启动Ollama）")
        if st.button("检索并回答"):
            hits = graph_search(question, index)
            if generate:
                try:
                    st.write(answer(question, hits, model_name))
                except Exception as exc:
                    st.warning(f"模型调用失败：{type(exc).__name__}。下方保留真实检索证据。")
            else:
                st.info("当前显示原文证据。勾选模型选项可生成答案。")
            for i, hit in enumerate(hits, 1):
                with st.expander(f'[{i}] {hit["company"]} {hit["year"]} · PDF第{hit["page"]}页', expanded=i == 1):
                    st.write(hit["text"])
            st.download_button("下载证据", json.dumps(hits, ensure_ascii=False, indent=2), "evidence.json")
    with tab2:
        st.graphviz_chart(graph_dot(graph, company))
        st.caption(graph["note"])
        st.download_button("导出图谱", json.dumps(graph, ensure_ascii=False, indent=2), "graph.json")
    with tab3:
        baseline = read_json(ROOT.parent / "工单7/评估结果.json")
        current = read_json(ROOT / "评估结果.json")
        st.table({"普通RAG": baseline["summary"], "图谱RAG": current["summary"]})
        st.caption("固定10题的已保存评估结果；指标只评估检索。更新索引后需重新运行两个工单的评估。")


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        index = load()
        index["graph"] = make_graph(index["chunks"])
        evaluate(index, graph_search, ROOT / "评估结果.json")
    else:
        page()
