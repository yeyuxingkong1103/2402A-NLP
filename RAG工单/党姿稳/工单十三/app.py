# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：Streamlit 界面，逐请求展示五个阶段的耗时明细与「检索结果是否在 3 秒内返回」的验收判定

import os                                                       # 导入 os 用于路径拼接
import json                                                     # 导入 json 用于读取结构化日志
import pandas as pd                                             # 导入 pandas 用于渲染阶段耗时表
import streamlit as st                                          # 导入 streamlit 用于构建界面
import config                                                   # 导入本工单配置模块
import resources                                                # 导入资源加载模块
import baseline_pipeline                                        # 导入优化前流水线
import optimized_pipeline                                       # 导入优化后流水线
from logger import StageTracer                                  # 导入阶段追踪器

st.set_page_config(page_title="RAG 检索性能跟踪", layout="wide")   # 设置页面标题与宽布局


@st.cache_resource(show_spinner="正在加载语料与嵌入模型……")        # 缓存共享资源，避免每次交互重载
def _load_shared():                                             # 定义共享资源加载函数
    """加载语料与嵌入模型，两条流水线共用同一份实例。"""
    chunks = resources.load_chunks()                            # 加载语料块
    model = resources.load_model()                              # 加载嵌入模型
    return chunks, model                                        # 返回共享资源


@st.cache_resource(show_spinner="正在构建流水线……")               # 缓存流水线实例
def _build(mode):                                               # 定义流水线构建函数
    """按模式构建流水线：optimized 为优化后，baseline 为优化前。"""
    chunks, model = _load_shared()                              # 取出共享资源
    if mode == "优化前（基线）":                                   # 基线模式
        pipe = baseline_pipeline.build(chunks=chunks, model=model)[0]  # 构建优化前流水线
    else:                                                       # 优化后模式
        pipe = optimized_pipeline.build(chunks=chunks, model=model, warm=True)  # 构建并预热
    return pipe                                                 # 返回流水线实例


def _recent_traces(limit=8):                                    # 定义最近访问记录读取函数
    """读取结构化阶段日志的最后若干条记录，用于展示最近的请求耗时。"""
    path = config.LOG_FILE                                      # 结构化日志路径
    if not os.path.exists(path):                                # 日志尚不存在
        return []                                               # 返回空列表
    with open(path, "r", encoding="utf-8") as f:                # 以 UTF-8 打开日志
        lines = f.readlines()[-limit * 6:]                       # 多读若干行（一条请求含五个阶段）
    rows = []                                                   # 输出记录
    for line in lines:                                          # 逐行解析
        line = line.strip()                                     # 去掉行尾换行
        if not line:                                            # 跳过空行
            continue                                            # 继续下一行
        try:                                                    # 容错解析，避免半行导致崩溃
            rows.append(json.loads(line))                       # 解析为字典
        except ValueError:                                      # JSON 不完整时忽略
            continue                                            # 继续下一行
    return rows[-limit:]                                        # 返回最后 limit 条阶段记录


def _stage_table(summary):                                      # 定义阶段耗时表渲染函数
    """把一次请求的五阶段耗时渲染为表格，并标出占检索口径的比重。"""
    ret = summary["retrieval_seconds"]                          # 检索口径总耗时
    rows = []                                                   # 表格数据行
    for name in config.STAGES:                                  # 按固定顺序遍历五个阶段
        sec = summary["stages"].get(name, 0.0)                  # 取该阶段耗时
        share = ("%.1f%%" % (100.0 * sec / ret)) if ret > 1e-9 else "-"  # 在检索口径中占比
        rows.append({"阶段": name, "耗时(s)": round(sec, 4),      # 追加行：阶段名与耗时
                     "占检索口径": share})                        # 追加占比列
    df = pd.DataFrame(rows)                                     # 构造数据框
    st.dataframe(df, hide_index=True, use_container_width=True)  # 渲染表格
    st.bar_chart(df.set_index("阶段")["耗时(s)"], height=240)      # 绘制阶段耗时柱状图


def _show_response(resp):                                       # 定义答案展示函数
    """展示模型答案与引用来源。"""
    st.markdown("**答案**")                                      # 小标题
    st.write(resp["answer"])                                    # 输出答案正文
    if resp["citations"]:                                       # 存在引用时展示
        st.markdown("**引用来源**")                               # 小标题
        for c in resp["citations"]:                             # 遍历引用条目
            st.write("- " + c)                                  # 逐条输出


def main():                                                     # 定义界面主函数
    """渲染页面：选择系统与问题，执行一次检索并展示阶段耗时与达标情况。"""
    config.ensure_dirs()                                        # 确保输出目录存在
    st.title("RAG 检索性能跟踪（工单十三）")                        # 页面主标题
    st.caption("验收标准：用户输入 query 后，检索结果返回时间 ≤ %.0f 秒" % config.RETRIEVAL_BUDGET)  # 验收说明
    with st.sidebar:                                            # 侧边栏放参数
        st.header("运行参数")                                    # 侧栏标题
        mode = st.radio("系统", ["优化后", "优化前（基线）"], index=0)  # 系统选择
        preset = st.selectbox("测试问题（工单十二验收集）",          # 问题下拉
                              ["自定义"] + ["[%d] %s" % (q["id"], q["query"]) for q in config.QUERIES],
                              index=1)                          # 默认选第一条预设
        user_q = st.text_area("自定义问题", value="", height=80,  # 自定义输入框
                              placeholder="选择预设问题或在此输入自定义问题")  # 占位提示
        run = st.button("执行检索", type="primary")               # 执行按钮
        st.divider()                                            # 分隔线
        st.caption("结构化日志：%s" % os.path.basename(config.LOG_FILE))  # 日志文件提示
    if preset != "自定义":                                       # 选中了预设问题
        query = preset.split("] ", 1)[1]                        # 去掉编号前缀取问题原文
    else:                                                       # 使用自定义输入
        query = user_q.strip()                                  # 去除首尾空白
    if not run and not query:                                   # 首次进入且无输入时不执行
        st.info("请在左侧选择一个测试问题，然后点击「执行检索」。")     # 提示信息
        return                                                  # 结束本次渲染
    if not query:                                               # 自定义模式但输入为空
        st.warning("请先输入问题。")                              # 警告提示
        return                                                  # 结束本次渲染
    pipe = _build(mode)                                         # 构建或复用流水线
    tracer = StageTracer(query=query)                           # 创建带 request_id 的追踪器
    with st.spinner("检索与生成中……"):                            # 执行期间显示加载提示
        resp = pipe.run(query, tracer)                          # 执行一次完整流程
    summary = tracer.summary()                                  # 取阶段耗时摘要
    c1, c2, c3, c4 = st.columns(4)                              # 四个指标卡片
    c1.metric("检索结果返回(s)", "%.4f" % summary["retrieval_seconds"])  # 检索口径耗时
    c2.metric("端到端总耗时(s)", "%.4f" % summary["total_seconds"])       # 全流程耗时
    c3.metric("验收判定", "达标" if summary["within_budget"] else "超时")  # 是否 ≤3 秒
    c4.metric("request_id", summary["request_id"])              # 请求标识（可据此查日志）
    if summary["within_budget"]:                                # 达标
        st.success("检索结果在 %.0f 秒内返回（%.4f 秒）" % (config.RETRIEVAL_BUDGET,
                                                          summary["retrieval_seconds"]))
    else:                                                       # 超时
        st.error("检索结果超出 %.0f 秒预算（%.4f 秒）" % (config.RETRIEVAL_BUDGET,
                                                        summary["retrieval_seconds"]))
    tab1, tab2, tab3 = st.tabs(["阶段耗时", "答案与引用", "最近阶段日志"])  # 三个标签页
    with tab1:                                                  # 阶段耗时页
        _stage_table(summary)                                   # 渲染阶段表与图
        st.caption("检索口径 = 查询处理与增强 + 检索 + 上下文组装与提示工程；"
                   "LLM 生成不属于「检索结果返回」时限范围。")     # 口径说明
    with tab2:                                                  # 答案页
        _show_response(resp)                                    # 展示答案与引用
    with tab3:                                                  # 日志页
        st.caption("来自 %s 的最后 8 条阶段记录（每行一个阶段，同 request_id 为同一次请求）。"
                   % os.path.basename(config.LOG_FILE))          # 数据来源说明
        st.dataframe(pd.DataFrame(_recent_traces()), hide_index=True,  # 渲染最近阶段记录
                     use_container_width=True)                  # 撑满容器宽度


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动界面
