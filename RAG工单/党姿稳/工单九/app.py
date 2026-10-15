# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：Streamlit 交互界面——支持优化前/优化后/对比三种模式、中英文问答、检索证据与耗时展示
# 启动方式：streamlit run app.py   （推荐用 python start_app.py，可自动选择空闲端口）

import os                                                       # 导入 os 用于路径注册
import sys                                                      # 导入 sys 用于模块搜索路径
import time                                                     # 导入 time 用于统计交互耗时

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 把本目录加入模块搜索路径

import streamlit as st                                          # 导入 Streamlit 前端框架
import config                                                   # 导入本工单配置模块
import data_loader                                              # 导入语料与问题加载
import graph_builder                                            # 导入图谱构建
from build_index import build_index                             # 导入索引构建入口
from pipeline import GraphRAGPipeline                           # 导入问答管线

st.set_page_config(page_title="工单九 Graph RAG 优化",           # 设置页面标题
                   page_icon="📊", layout="wide")                # 设置图标与宽屏布局

MODE_LABELS = ["优化后 (Optimized)", "优化前 (Baseline)", "对比 (Compare)"]  # 模式选项（含中英）


@st.cache_resource(show_spinner="正在加载检索索引与知识图谱（首次较慢）...")  # 资源缓存：只初始化一次
def load_all():                                                 # 定义资源加载函数
    """加载检索索引与两版知识图谱，返回可复用的问答管线。"""
    retriever, chunks = build_index(use_cache=True)             # 加载或构建检索索引
    questions = data_loader.load_questions()                    # 加载验收问题（用于建图采样）
    graph_opt = graph_builder.build_graph(chunks, retriever.vectors, questions,  # 加载优化版图谱
                                          optimized=True, use_cache=True)  # 优先走缓存
    graph_base = graph_builder.build_graph(chunks, retriever.vectors, questions,  # 加载基线版图谱
                                           optimized=False, use_cache=True)  # 优先走缓存
    return GraphRAGPipeline(retriever, graph_opt, graph_base), chunks  # 返回管线与块列表


def render_timing(res):                                         # 定义耗时展示函数
    """渲染分阶段耗时与检索耗时指标。"""
    t = res["timing"]                                           # 取耗时汇总
    cols = st.columns(3)                                        # 三列布局
    cols[0].metric("检索耗时", "%.3f s" % res["retrieve_seconds"],  # 展示检索耗时（验收关键指标）
                   delta="达标" if res["retrieve_seconds"] <= config.RETRIEVE_TIME_BUDGET else "超时")  # 与预算对比
    cols[1].metric("端到端耗时", "%.3f s" % t["total"])          # 展示端到端耗时
    cols[2].metric("上下文块数", "%d" % len(res["contexts"]))     # 展示上下文块数量
    with st.expander("分阶段耗时明细"):                          # 折叠面板展示明细
        st.table({"阶段": [n for n, _ in t["stages"]],           # 阶段名列
                  "耗时(秒)": ["%.4f" % c for _, c in t["stages"]]})  # 耗时列


def render_result(res, title):                                  # 定义结果渲染函数
    """渲染单个模式的问答结果：答案、引用来源、图谱关系。"""
    st.subheader(title)                                         # 小标题
    st.markdown(res["answer"])                                  # 展示生成的答案（Markdown 渲染）
    render_timing(res)                                          # 展示耗时
    with st.expander("检索到的上下文片段（%d 条）" % len(res["contexts"])):  # 折叠面板展示证据
        for i, (txt, src) in enumerate(zip(res["contexts"], res["sources"]), 1):  # 遍历上下文与来源
            st.markdown("**[资料%d] %s 第%d页**" % (i, src["source"], src["page"]))  # 展示来源标签
            st.text(txt[:600])                                  # 展示片段正文（截断避免过长）
    if res["triples"]:                                          # 若存在图谱关系证据
        with st.expander("知识图谱关系（%d 条）" % len(res["triples"])):  # 折叠面板
            st.text("\n".join(res["triples"]))                  # 逐行展示关系三元组


def main():                                                     # 定义界面主函数
    st.title("金融文档 Graph RAG 智能问答（工单九 · 优化版）")     # 页面主标题
    st.caption("工单编号：人工智能NLP-RAG-Graph RAG 优化任务 ｜ "  # 副标题说明
               "评估指标：上下文精度 ≥0.8、上下文召回 ≥0.9、检索耗时 ≤3s")  # 验收标准提示

    with st.sidebar:                                            # 侧边栏
        st.header("设置")                                        # 侧边栏标题
        mode = st.radio("检索模式", MODE_LABELS, index=0)        # 模式单选
        lang_choice = st.selectbox("回答语言", ["自动检测 (Auto)", "中文", "English"])  # 语言选择
        topk = st.slider("返回上下文块数", 3, 10, config.FINAL_TOP_K)  # 上下文数量调节
        st.divider()                                            # 分割线
        st.markdown("**示例问题**")                              # 提示文字
        try:                                                    # 尝试加载示例问题
            for qa in data_loader.load_questions()[:4]:         # 取前 4 条验收问题
                if st.button(qa["question"][:40] + "...", key="ex%d" % qa["id"]):  # 生成候选问题按钮
                    st.session_state["pending"] = qa["question"]  # 点击后写入待提问队列
        except Exception as e:                                  # 容错
            st.warning("示例问题加载失败：%r" % e)                # 展示告警

    if "history" not in st.session_state:                       # 初始化会话状态
        st.session_state["history"] = []                        # 对话历史列表
    question = st.text_input("请输入你的问题（支持中文 / English）：",  # 文本输入框
                             value=st.session_state.pop("pending", ""))  # 默认值可来自示例按钮
    if not question:                                            # 未输入时结束本次渲染
        st.info("请输入问题，或点击左侧示例问题开始。")            # 提示信息
        return                                                  # 返回

    pipeline, chunks = load_all()                               # 加载索引与图谱（带缓存）
    lang = None                                                 # 语言参数初值（None 表示自动检测）
    if lang_choice == "中文":                                    # 强制中文
        lang = "zh"                                             # 指定为中文
    elif lang_choice == "English":                              # 强制英文
        lang = "en"                                             # 指定为英文
    config.FINAL_TOP_K = topk                                   # 应用滑块选择的上下文数量

    with st.spinner("正在检索并生成答案..."):                     # 加载提示
        t0 = time.time()                                        # 记录交互起始时间
        try:                                                    # 捕获运行时异常，保证界面不崩
            if mode.startswith("对比"):                          # 对比模式
                opt = pipeline.answer_optimized(question)       # 先跑优化后
                base = pipeline.answer_baseline(question)       # 再跑优化前
                col1, col2 = st.columns(2)                      # 双列并排展示
                with col1:                                      # 左列
                    render_result(opt, "优化后 (Optimized)")     # 展示优化后结果
                with col2:                                      # 右列
                    render_result(base, "优化前 (Baseline)")     # 展示优化前结果
                st.success("检索提速 %.3fs → %.3fs" % (            # 展示对比结论
                    base["retrieve_seconds"], opt["retrieve_seconds"]))  # 前后检索耗时
            else:                                               # 单模式
                if mode.startswith("优化后"):                    # 优化后模式
                    res = pipeline.answer_optimized(question)   # 执行优化后管线
                    render_result(res, "优化后结果 (Optimized)")  # 渲染结果
                else:                                           # 优化前模式
                    res = pipeline.answer_baseline(question)    # 执行基线管线
                    render_result(res, "优化前结果 (Baseline)")   # 渲染结果
        except Exception as e:                                  # 容错：任何异常都给出友好提示
            st.error("处理失败，请重试或更换问题。错误信息：%r" % e)  # 展示错误
            return                                              # 返回
    st.caption("本次交互总耗时 %.2f 秒" % (time.time() - t0))      # 展示交互总耗时
    st.session_state["history"].append(question)                # 把问题加入历史


if __name__ == "__main__":                                      # 兼容直接以 python app.py 调试的场景
    main()                                                      # 调用界面主函数
