# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：Streamlit 演示界面。展示解析修复结果（文档分流、OCR 补文、图纸描述），
#           并对比"RAG 回答"与"纯 LLM 回答"，高亮检索到的正文/著录项/图形块。

import os                                                        # 路径
import json                                                      # 读取评估结果
import streamlit as st                                           # 界面框架
import config                                                    # 配置
from indexer import build_index                                  # 索引（带缓存）
from retriever import pick_top_chunks, assemble_context          # 检索
from generator import answer_question, call_llm                  # 生成

st.set_page_config(page_title="工单十四 · 工业PDF解析修复与问答", layout="wide")  # 页面配置

TYPE_CN = {"text": "正文", "biblio": "著录项", "figure": "图纸"}  # 块类型中文名
TYPE_COLOR = {"text": "#e8f4ff", "biblio": "#fff4e0", "figure": "#e6ffe9"}  # 类型底色


@st.cache_resource(show_spinner="正在加载索引（首次运行需解析 PDF 并 OCR）...")
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
    with st.expander("解析修复详情（每页来源与图纸识别）", expanded=False):
        rows = []                                                # 明细表
        for p in idx["pages"]:
            rows.append({"页码": p["page_no"],                     # 页码
                         "文本来源": "OCR（扫描页）" if p["source"] == "ocr" else "直接提取",
                         "字符数": p["chars"],                      # 字符数
                         "置信度": p.get("confidence", ""),          # OCR 置信度
                         "图纸": "、".join(p.get("labels") or []) or "—"})  # 图号
        st.dataframe(rows, use_container_width=True)             # 表格展示


def render_chunks(picked):
    """渲染检索片段，按块类型着色。"""
    for c in picked:                                             # 逐块
        color = TYPE_COLOR.get(c["type"], "#f5f5f5")             # 底色
        tag = TYPE_CN.get(c["type"], c["type"])                  # 中文类型
        st.markdown(
            f"<div style='background:{color};padding:8px;border-radius:6px;margin-bottom:6px'>"
            f"<b>[{tag}] {c['chunk_id']}</b>　第 {c['page_no']} 页　"
            f"融合分 {c['score']}<br><span style='font-size:13px'>{c['text'][:800]}</span></div>",
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


def main():
    st.title("工单十四 · 修复低质量工业 PDF 的解析与信息丢失")     # 标题
    st.caption("RAGFlow 式解析链路的本地复现：扫描页 OCR 补文 + 专利著录项整体保留 + 图纸页跨模态描述")
    idx = get_index()                                            # 加载索引
    render_meta(idx)                                             # 解析概况
    show_figure_images()                                         # 图纸预览

    st.sidebar.header("提问")                                    # 侧边栏
    preset = st.sidebar.selectbox("选择验收问题",                 # 预置 6 题
                                  ["（自定义）"] + [f"{q['id']}. {q['question']}"
                                                    for q in config.TEST_QUESTIONS])
    default_q = ""                                               # 自定义默认值
    if preset != "（自定义）":                                    # 选中预置题
        qid = int(preset.split(".")[0])                          # 题号
        default_q = next(q["question"] for q in config.TEST_QUESTIONS if q["id"] == qid)
    question = st.text_area("问题", value=default_q, height=80)  # 输入框
    options = None                                               # 选项
    matched = [q for q in config.TEST_QUESTIONS if q["question"] == question]
    if matched:                                                  # 预置题带选项
        options = matched[0]["options"]
        st.info("本题为选择题，系统将输出选项字母，标准答案为：" + matched[0]["answer"])
    compare = st.sidebar.checkbox("同时展示纯 LLM 回答（不检索）", value=True)  # 对比开关
    go = st.sidebar.button("提交提问")                            # 提交

    if go and question.strip():                                  # 触发问答
        picked = pick_top_chunks(question, idx)                  # 检索
        letter, raw, context = answer_question(question, options, picked, assemble_context)
        col1, col2 = st.columns(2)                               # 双列对比
        with col1:                                               # RAG 回答
            st.subheader("RAG 回答" + (f"（选项 {letter}）" if letter else ""))
            st.write(raw)
        with col2:                                               # 纯 LLM 回答
            st.subheader("纯 LLM 回答" if compare else "")
            if compare:
                st.write(call_llm(f"【问题】{question}\n请直接回答。",
                                  system="你是通用助手，请仅凭自身知识回答。"))
        st.subheader("检索到的知识片段")                          # 片段
        render_chunks(picked)                                    # 渲染
        with st.expander("送入 LLM 的完整上下文"):                 # 上下文
            st.text(context)                                     # 原文

    report = os.path.join(config.EVAL_DIR, "results.json")       # 评估结果
    if os.path.exists(report):                                   # 存在则展示
        with st.expander("6 个验收问题的评估结果"):                # 折叠
            data = json.load(open(report, encoding="utf-8"))     # 读取
            st.json(data["summary"])                             # 汇总
            st.dataframe([{"题号": c["id"], "题型": c["group"],    # 逐题
                           "标准": c["gold"], "预测": c["pred"],
                           "是否正确": "✓" if c["correct"] else "✗",
                           "耗时(s)": c["seconds"]} for c in data["cases"]],
                         use_container_width=True)


if __name__ == "__main__":                                        # 直接运行入口
    main()
