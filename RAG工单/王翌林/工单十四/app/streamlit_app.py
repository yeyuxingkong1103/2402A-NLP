# -*- coding: utf-8 -*-
"""
工单十四：Streamlit 多模态问答界面
"""
import os
import sys
import json
import time
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import WORK_ORDER_NO, APP_PORT

API_BASE = f"http://localhost:{APP_PORT}"

st.set_page_config(page_title=f"{WORK_ORDER_NO} - 工业PDF多模态RAG", layout="wide")
st.title(f"🏭 {WORK_ORDER_NO}：低质量工业PDF解析与多模态问答")
st.caption("针对图片型PDF（无文本层）的 OCR + 多模态VQA + RAG 问答系统")

# 测试问题
QUESTIONS = [
    {"id": 1, "q": "根据文本信息，该静电除尘器的发明人是？",
     "options": ["A. P·吉特勒", "B. 未知", "C. 静电除尘器研究小组", "D. 该专利未提及发明人"], "ans": "A"},
    {"id": 2, "q": "根据文本信息，以下哪个描述符合该静电除尘器的特征？",
     "options": ["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极",
                 "B. 具有管状入口和出口，它们分别由3种不同圆锥形部分所构成",
                 "C. 管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式",
                 "D. 主要用于液体的除尘"], "ans": "C"},
    {"id": 3, "q": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？",
     "options": ["A. 部件4位于部件5的左侧", "B. 部件4位于部件5的右侧",
                 "C. 部件4位于部件5的上方", "D. 部件4位于部件5的下方"], "ans": "A"},
    {"id": 4, "q": "在文件中第7页的图片中，尺寸X1，X2，X3分别代表什么部件的间隔距离？",
     "options": ["A. 配气带孔盘6，6'，6\"之间的间隔距离",
                 "B. 外壳2与圆锥形部分10之间的间隔距离",
                 "C. 管状入口1与管状出口9之间的间隔距离",
                 "D. 外壳中心轴线3与外壳2之间的间隔距离"], "ans": "A"},
    {"id": 5, "q": "根据文件中第7页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？",
     "options": ["A. 先经过部件4，再经过部件5", "B. 先经过部件10，再经过部件6\"",
                 "C. 先经过部件6\"，再经过部件6'", "D. 先经过部件6，再经过部件4"], "ans": "C"},
    {"id": 6, "q": "根据文件中第7页图示，如果已知外壳直径D，那么h1和h2的尺寸可以用来计算什么？",
     "options": ["A. 计算气流速度", "B. 确定配气带孔盘6，6'，6\"的位置",
                 "C. 计算除尘效率", "D. 确定外壳2的材料"], "ans": "B"},
]

st.sidebar.header("📋 测试问题")
selected_q = st.sidebar.selectbox("选择问题", [f"Q{q['id']}. {q['q'][:40]}..." for q in QUESTIONS])
qid = int(selected_q.split(".")[0].replace("Q", ""))
q = QUESTIONS[qid - 1]

st.sidebar.markdown("---")
st.sidebar.markdown("#### 标准答案")
st.sidebar.markdown(f"**{q['ans']}**")

# 主区域
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"问题 {q['id']}")
    st.markdown(f"**{q['q']}**")
    st.markdown("**选项：**")
    for opt in q["options"]:
        st.markdown(f"- {opt}")

    if st.button("🔍 提问", type="primary", use_container_width=True):
        with st.spinner("正在检索与生成答案..."):
            try:
                t0 = time.time()
                resp = requests.post(f"{API_BASE}/api/ask", json={"query": q["q"]}, timeout=120)
                elapsed = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ 答案（耗时 {data.get('latency_ms', 0):.0f}ms）")
                    st.markdown(f"**{data.get('answer', '')}**")
                    st.caption(f"路由模式: {data.get('mode', '')} | 路由类型: {data.get('route', {}).get('query_type', '')}")
                    if data.get("image_path"):
                        st.image(data["image_path"], caption="检索到的图纸", use_container_width=True)
                    if data.get("references"):
                        with st.expander("📚 引用来源"):
                            for ref in data["references"]:
                                st.markdown(f"- 第{ref.get('page')}页 | score={ref.get('score')} | {ref.get('preview', '')[:80]}")
                else:
                    st.error(f"请求失败: {resp.status_code}")
            except Exception as e:
                st.error(f"错误: {e}")

with col2:
    st.subheader("📄 专利文档")
    # 显示第7页图纸（图像问题的关键页面）
    page7_path = "./data/pages/rendered/page_07.png"
    if os.path.exists(page7_path):
        st.image(page7_path, caption="第7页 技术图纸（图1）", use_container_width=True)
    else:
        st.info("图纸未渲染，请先运行知识库构建")

st.markdown("---")
st.markdown("### 📊 6题批量测试")
if st.button("运行全部6题测试", use_container_width=True):
    results = []
    progress = st.progress(0)
    for i, q in enumerate(QUESTIONS):
        try:
            resp = requests.post(f"{API_BASE}/api/ask", json={"query": q["q"]}, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                # 简单判断答案是否包含正确选项字母
                correct_letter = q["ans"]
                is_correct = correct_letter in answer or (
                    correct_letter == "A" and any(k in answer for k in ["吉特勒", "左侧", "配气带孔盘6", "6\"", "6'", "位置"])
                )
                results.append({"id": q["id"], "question": q["q"][:30], "answer": answer[:100],
                                "correct_ans": q["ans"], "correct": is_correct,
                                "latency_ms": data.get("latency_ms", 0)})
        except Exception as e:
            results.append({"id": q["id"], "question": q["q"][:30], "answer": f"ERROR: {e}",
                            "correct_ans": q["ans"], "correct": False, "latency_ms": 0})
        progress.progress((i + 1) / 6)

    st.dataframe(results)
    correct_count = sum(1 for r in results if r["correct"])
    st.metric("准确率", f"{correct_count}/6 = {correct_count/6*100:.0f}%")
    st.metric("平均响应时间", f"{sum(r['latency_ms'] for r in results)/len(results):.0f}ms")
