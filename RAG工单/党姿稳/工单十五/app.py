# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：Streamlit 演示界面。展示查询理解（视觉引用识别）、图像增强查询、
#           多路召回规模与跨模态重排结果，并支持"优化前/优化后"检索流程对比。

import os                                                        # 路径
import json                                                      # 读取评估结果
import streamlit as st                                           # 界面框架
import config                                                    # 配置
from indexer import build_index                                  # 索引（带缓存）
from cross_modal_retriever import pick_top_chunks, assemble_context  # 跨模态检索
from generator import answer_question, call_llm                  # 生成

st.set_page_config(page_title="工单十五 · 跨模态检索优化", layout="wide")  # 页面配置

TYPE_CN = {"text": "正文", "biblio": "著录项", "figure": "图纸"}    # 块类型中文名
TYPE_COLOR = {"text": "#e8f4ff", "biblio": "#fff4e0", "figure": "#e6ffe9"}  # 类型底色


@st.cache_resource(show_spinner="正在加载索引（首次运行需解析 PDF 并生成图纸描述）...")
def get_index():
    """索引全局缓存，避免每次交互重建。"""
    return build_index()


def render_meta(idx):
    """顶部展示文档解析分流结果。"""
    meta = idx["meta"]                                           # 文档级信息
    c1, c2, c3, c4 = st.columns(4)                               # 四列指标
    c1.metric("文档", meta["file"])                              # 文件名
    c2.metric("页面类型", meta["kind"])                          # 文字型/扫描型/混合型
    c3.metric("总页数", meta["page_count"])                      # 页数
    c4.metric("扫描页占比", f"{meta['scan_ratio'] * 100:.0f}%")   # 扫描页占比
    with st.expander("解析与图纸块详情", expanded=False):
        st.dataframe([{"页码": p["page_no"],                       # 页码
                       "文本来源": "OCR（扫描页）" if p["source"] == "ocr" else "直接提取",
                       "字符数": p["chars"],                       # 字符数
                       "图纸": "、".join(p.get("labels") or []) or "—"}
                      for p in idx["pages"]], use_container_width=True)


def render_diag(diag):
    """展示查询理解与多路召回诊断信息。"""
    refs = diag.get("refs", {})                                  # 视觉引用
    c1, c2, c3 = st.columns(3)                                   # 三列指标
    c1.metric("视觉引用", "是" if refs.get("has_visual") else "否",
              help="问题中是否出现 图N / 第N页 / 图纸 等视觉引用")
    c2.metric("识别到的图号", "、".join(refs.get("labels") or []) or "—")  # 图号
    c3.metric("图像增强查询", "已启用" if diag.get("enhanced_query_used") else "未启用",
              help=f"追加图纸描述 {diag.get('figure_desc_chars', 0)} 字")
    st.write("**各路召回规模**：", diag.get("route_sizes", {}))    # 召回条数


def render_chunks(picked):
    """渲染检索片段，按块类型着色并标出图号。"""
    for c in picked:                                             # 逐块
        color = TYPE_COLOR.get(c["type"], "#f5f5f5")             # 底色
        tag = TYPE_CN.get(c["type"], c["type"])                  # 中文类型
        labels = "、".join(c.get("labels") or []) or "—"          # 图号
        st.markdown(
            f"<div style='background:{color};padding:8px;border-radius:6px;margin-bottom:6px'>"
            f"<b>[{tag}] {c['chunk_id']}</b>　第 {c['page_no']} 页　图号 {labels}　"
            f"综合分 {c['score']}<br><span style='font-size:13px'>{c['text'][:800]}</span></div>",
            unsafe_allow_html=True)


def show_figure_images():
    """展示导出到 data/figures 的图纸图片。"""
    fig_dir = os.path.join(config.DATA_DIR, "figures")           # 图纸目录
    if not os.path.isdir(fig_dir):                               # 不存在则跳过
        return
    imgs = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png"))  # PNG 列表
    if not imgs:                                                 # 无图片
        return
    st.subheader("识别出的图纸页")
    cols = st.columns(min(len(imgs), 3))                         # 最多三列
    for i, name in enumerate(imgs):                              # 逐图展示
        cols[i % len(cols)].image(os.path.join(fig_dir, name), caption=name)


def render_report():
    """展示优化前后的评估结果。"""
    report = os.path.join(config.EVAL_DIR, "results.json")       # 评估结果
    if not os.path.exists(report):                               # 不存在则跳过
        return
    data = json.load(open(report, encoding="utf-8"))             # 读取
    with st.expander("优化前后评估结果", expanded=False):         # 折叠区
        c1, c2 = st.columns(2)                                   # 双列
        c1.metric("优化前准确率", f"{data['before']['summary']['accuracy'] * 100:.1f}%",
                  help="文本向量 + BM25 等权融合")
        c2.metric("优化后准确率", f"{data['after']['summary']['accuracy'] * 100:.1f}%",
                  delta=f"{(data['after']['summary']['accuracy'] - data['before']['summary']['accuracy']) * 100:+.1f}%")
        st.dataframe([{"题号": c["id"], "题型": c["group"],       # 优化后逐题
                       "标准": c["gold"], "预测": c["pred"],
                       "图纸命中名次": c.get("figure_hit_rank") or "—",
                       "是否正确": "✓" if c["correct"] else "✗",
                       "耗时(s)": c["seconds"]}
                      for c in data["after"]["cases"]], use_container_width=True)


def main():
    st.title("工单十五 · 优化技术图纸与文本的跨模态检索流程")      # 标题
    st.caption("查询理解（视觉引用识别）→ 图像增强查询 → 多路召回 → 跨模态重排 的完整链路")
    idx = get_index()                                            # 加载索引
    render_meta(idx)                                             # 解析概况

    st.sidebar.header("检索设置")                                 # 侧边栏
    optimized = st.sidebar.radio("检索流程", ["优化后（跨模态多路检索）", "优化前（文本向量+BM25）"],
                                 index=0) == "优化后（跨模态多路检索）"
    preset = st.sidebar.selectbox("选择验收问题",                 # 预置 6 题
                                  ["（自定义）"] + [f"{q['id']}. {q['question']}"
                                                    for q in config.TEST_QUESTIONS])
    default_q = ""                                               # 自定义默认值
    if preset != "（自定义）":                                    # 选中预置题
        qid = int(preset.split(".")[0])                          # 题号
        default_q = next(q["question"] for q in config.TEST_QUESTIONS if q["id"] == qid)
    question = st.text_area("问题", value=default_q, height=80)  # 输入框
    options, gold = None, ""                                     # 选项与标准答案
    matched = [q for q in config.TEST_QUESTIONS if q["question"] == question]
    if matched:                                                  # 预置题带选项
        options, gold = matched[0]["options"], matched[0]["answer"]
        st.info(f"本题为选择题，标准答案为：{gold}")
    compare = st.sidebar.checkbox("同时展示纯 LLM 回答（不检索）", value=False)  # 对比开关
    go = st.sidebar.button("提交提问")                            # 提交按钮

    if go and question.strip():                                  # 触发问答
        picked, diag = pick_top_chunks(question, idx, optimized=optimized)  # 跨模态检索
        st.subheader("查询理解与检索诊断")                         # 诊断标题
        render_diag(diag)                                        # 诊断信息
        letter, raw, context = answer_question(question, options, picked, assemble_context)
        col1, col2 = st.columns(2) if compare else (st.container(), None)  # 双列对比
        with col1:                                               # RAG 回答
            st.subheader("RAG 回答" + (f"（选项 {letter}）" if letter else ""))
            st.write(raw)
        if compare:                                              # 纯 LLM 回答
            with col2:
                st.subheader("纯 LLM 回答")
                st.write(call_llm(f"【问题】{question}\n请直接回答。",
                                  system="你是通用助手，请仅凭自身知识回答。"))
        st.subheader("检索到的知识片段")                          # 片段
        render_chunks(picked)                                    # 渲染
        with st.expander("送入 LLM 的完整上下文"):                 # 上下文
            st.text(context)                                     # 原文

    show_figure_images()                                         # 图纸预览
    render_report()                                              # 评估报告


if __name__ == "__main__":                                        # 直接运行入口
    main()
