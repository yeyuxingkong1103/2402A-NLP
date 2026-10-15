# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：Streamlit 交互界面——可选择使用 RAG 或 LightRAG 知识库检索并生成答案

import os                                                       # 导入 os 用于判断图谱文件是否存在
import json                                                     # 导入 json 用于读取知识图谱
import time                                                     # 导入 time 用于统计检索耗时
import streamlit as st                                          # 导入 Streamlit 界面框架
import config                                                   # 导入本工单配置模块
import corpus                                                   # 导入语料分块模块
import embedder                                                 # 导入嵌入编码模块
import rag_retriever                                            # 导入扁平 RAG 检索器
import lightrag_retriever                                       # 导入 LightRAG 检索器
import generator                                                # 导入答案生成模块

RAG_MODE = "RAG（向量 + BM25 混合检索）"                          # 扁平 RAG 模式名称
LR_MODE = "LightRAG（知识图谱双层级检索）"                        # LightRAG 模式名称


@st.cache_resource(show_spinner="正在加载语料、向量索引与知识图谱…")  # 资源只加载一次并全局复用
def load_resources():                                           # 定义资源加载函数
    """加载语料块、嵌入模型、扁平索引与 LightRAG 索引（图谱缺失时 LightRAG 不可用）。"""
    chunks = corpus.build_or_load_chunks()                      # 加载或构建语料块
    model = embedder.load_model()                               # 加载本地嵌入模型
    base = rag_retriever.build_index(chunks, model)             # 构建扁平 RAG 索引
    graph, lr = None, None                                      # 图谱与 LightRAG 索引默认不可用
    if os.path.isfile(config.GRAPH_FILE):                       # 图谱缓存存在时才构建
        with open(config.GRAPH_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开图谱
            graph = json.load(f)                                # 反序列化知识图谱
        lr = lightrag_retriever.build_index(model, graph, base)  # 构建 LightRAG 索引
    return chunks, model, base, lr, graph                       # 返回全部资源


def _hit_rows(hits):                                            # 定义检索结果表格构造函数
    """把检索结果整理为 Streamlit 表格数据与展开查看的正文列表。"""
    rows, texts = [], []                                        # 表格行与正文列表
    for h in hits:                                              # 遍历每条检索结果
        rows.append({"名次": h["rank"] + 1, "来源": h["source"],  # 名次与来源说明书
                     "页码": h["page"],                          # 来源页码
                     "召回路径": h.get("via", "混合"),             # 召回路径（LightRAG 三路标记）
                     "得分": round(float(h.get("score", 0.0)), 4)})  # 融合得分
        texts.append(h["text"])                                 # 收集正文用于展开显示
    return rows, texts                                          # 返回表格与正文


def _render_trace(trace):                                       # 定义 LightRAG 检索追溯渲染函数
    """展示 LightRAG 的关键词抽取与实体链接过程，体现图谱检索可解释性。"""
    if not trace:                                               # 无追溯信息时直接返回
        return                                                  # 结束渲染
    kw = trace.get("keywords", {})                              # 取关键词字典
    st.markdown("**关键词抽取**")                                 # 小节标题
    st.write("低层关键词（具体实体）：%s" % "、".join(kw.get("low_level_keywords", [])) or "（无）")  # 低层关键词
    st.write("高层关键词（抽象概念）：%s" % "、".join(kw.get("high_level_keywords", [])) or "（无）")  # 高层关键词
    st.markdown("**实体链接与图谱扩展**")                          # 小节标题
    st.write("直接命中实体：%s" % "、".join(trace.get("matched_entities", [])) or "（无）")  # 命中实体
    st.write("图谱邻域扩展命中：%s" % "、".join(trace.get("expanded_entities", [])) or "（无）")  # 扩展实体
    st.write("三路候选块数：局部 %d / 全局 %d / BM25 %d" % (trace.get("local_candidates", 0),  # 局部路候选数
                                                        trace.get("global_candidates", 0),  # 全局路候选数
                                                        trace.get("bm25_candidates", 0)))  # BM25 路候选数


def main():                                                     # 定义界面主函数
    """渲染完整交互页面：模式选择、问题输入、检索、生成与结果展示。"""
    st.set_page_config(page_title="招股说明书问答（RAG / LightRAG）", layout="wide")  # 页面基础配置
    st.title("招股说明书问答系统：RAG 与 LightRAG 对比")           # 页面主标题
    chunks, model, base, lr, graph = load_resources()           # 加载全部资源
    st.sidebar.header("检索设置")                                 # 侧边栏标题
    options = [RAG_MODE] + ([LR_MODE] if lr is not None else [])  # 可选模式（图谱缺失时禁用 LightRAG）
    mode = st.sidebar.radio("知识库", options, index=0)          # 单选检索方式
    top_k = st.sidebar.slider("上下文块数量", 3, 12, config.FINAL_TOP_K)  # 可调最终上下文规模
    with st.sidebar.expander("语料与图谱规模"):                    # 侧边栏信息折叠区
        st.write("语料块：%d" % len(chunks))                      # 显示语料块数
        st.write("语料字数：%d" % sum(len(c["text"]) for c in chunks))  # 显示语料总字数
        if graph is not None:                                   # 图谱可用时显示规模
            st.write("图谱实体：%d" % len(graph["entities"]))     # 显示实体总数
            st.write("图谱关系：%d" % len(graph["relations"]))    # 显示关系总数
        else:                                                   # 图谱缺失时给出提示
            st.warning("未找到知识图谱，请先运行 kg_build.py")     # 提示用户建图

    st.markdown("**选择一个工单测试问题，或自行输入问题**")         # 输入区提示
    preset = st.selectbox("工单测试问题", ["（自行输入）"] + [q["question"] for q in config.QUESTIONS])  # 预设问题下拉
    default = "" if preset == "（自行输入）" else preset           # 默认输入内容
    question = st.text_input("问题", value=default)              # 问题输入框
    if not st.button("检索并生成答案", type="primary"):           # 点击按钮才执行
        return                                                  # 未点击时结束渲染

    if not question.strip():                                    # 问题为空时提示
        st.warning("请输入问题")                                  # 输出告警
        return                                                  # 结束渲染

    with st.spinner("检索中…"):                                  # 检索阶段加载提示
        t0 = time.time()                                        # 记录检索起始时间
        if mode == LR_MODE and lr is not None:                  # LightRAG 模式
            hits, trace = lr.retrieve(question, top_k=top_k, with_trace=True)  # 图谱双层级检索
        else:                                                   # RAG 模式
            hits, trace = base.retrieve(model, question, top_k=top_k), None  # 向量+BM25 混合检索
        cost = time.time() - t0                                 # 计算检索耗时

    col1, col2, col3 = st.columns(3)                            # 顶部指标三列布局
    col1.metric("检索方式", "LightRAG" if (mode == LR_MODE and lr is not None) else "RAG")  # 当前模式
    col2.metric("检索耗时", "%.3f s" % cost)                     # 检索耗时
    col3.metric("召回块数", len(hits))                           # 召回上下文块数
    if cost > config.RETRIEVE_TIME_BUDGET:                      # 超过工单时间预算时告警
        st.error("检索耗时 %.3f 秒，超过 %.1f 秒预算" % (cost, config.RETRIEVE_TIME_BUDGET))  # 超时提示

    rows, texts = _hit_rows(hits)                               # 整理检索结果
    st.subheader("检索结果")                                     # 结果小节标题
    st.dataframe(rows, use_container_width=True, hide_index=True)  # 渲染检索结果表格
    with st.expander("查看各块正文"):                             # 正文折叠区
        for r, t in zip(rows, texts):                           # 遍历每条结果
            st.markdown("**第 %d 名 · %s 第 %d 页 · %s**" % (r["名次"], r["来源"], r["页码"], r["召回路径"]))  # 结果标题
            st.text(t)                                          # 展示块正文

    if trace:                                                   # LightRAG 模式展示检索追溯
        with st.expander("LightRAG 检索过程（关键词 → 实体 → 图谱扩展）"):  # 追溯折叠区
            _render_trace(trace)                                # 渲染追溯信息

    with st.spinner("生成答案中…"):                              # 生成阶段加载提示
        answer = generator.generate(question, hits)             # 调用 LLM 生成答案
    st.subheader("生成的答案")                                   # 答案小节标题
    st.markdown(answer)                                         # 渲染答案
    st.caption("答案完全基于检索到的招股说明书片段生成，已要求模型标注来源与页码。")  # 结果说明


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动界面
