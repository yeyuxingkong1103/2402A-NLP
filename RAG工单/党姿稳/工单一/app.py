# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
交互界面：使用 Gradio 提供问答界面，展示RAG回答、纯LLM回答及检索片段
"""
import os
import socket
import gradio as gr
from pdf_parser import build_chunks, load_chunks
from vector_store import VectorStore
from rag_chain import rag_answer, llm_only_answer
from config import CHUNKS_FILE


def find_available_port(start_port=7860, max_tries=20):
    """从 start_port 开始查找第一个可用端口，避免端口被占用导致启动失败"""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port  # 兜底，返回起始端口

# 全局向量库实例
vector_store = None


def init_store():
    """初始化向量库（文本块 + 向量）"""
    global vector_store
    if vector_store is None:
        print("[系统] 正在初始化知识库...")
        if os.path.exists(CHUNKS_FILE):
            chunks = load_chunks()
        else:
            chunks = build_chunks()
        vector_store = VectorStore(chunks)
        print("[系统] 知识库初始化完成！")
    return vector_store


def answer_question(question, show_contexts):
    """
    处理用户提问，返回 RAG 回答、纯LLM回答、检索片段
    """
    if not question or not question.strip():
        return "请输入问题", "请输入问题", "", ""

    store = init_store()

    # RAG 回答
    rag_ans, contexts, rag_time = rag_answer(question, store)

    # 纯 LLM 回答（对比）
    llm_ans, llm_time = llm_only_answer(question)

    # 格式化检索片段
    context_text = ""
    if show_contexts:
        for i, (c, s) in enumerate(contexts):
            context_text += f"【片段 {i+1} | 相似度: {s:.4f}】\n{c}\n\n"

    # 格式化对比信息
    comparison = (
        f"⏱️ RAG耗时: {rag_time:.2f}s\n"
        f"⏱️ 纯LLM耗时: {llm_time:.2f}s"
    )

    return rag_ans, llm_ans, context_text, comparison


def build_ui():
    """构建 Gradio 界面"""
    with gr.Blocks(title="基于PDF文档的问答系统 - RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📄 基于PDF文档的问答系统（RAG）
        **工单编号：人工智能NLP-RAG-基于PDF文档的问答系统**

        本系统基于《武汉兴图新科电子股份有限公司招股说明书》构建，支持：
        - 基于PDF文档内容的精准问答（RAG模式）
        - 与纯LLM回答的对比分析
        - 展示检索到的相关文档片段
        """)

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="请输入您的问题",
                    placeholder="例如：武汉兴图新科电子股份有限公司注册资本是多少？",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 提交问题", variant="primary")
                    show_ctx_checkbox = gr.Checkbox(
                        label="显示检索片段", value=True
                    )

                # 快捷问题按钮
                gr.Markdown("**快捷问题（验收问题）：**")
                quick_questions = [
                    "武汉兴图新科电子股份有限公司注册资本是多少？",
                    "武汉兴图新科电子股份有限公司法定代表人是谁？",
                    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
                    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                ]
                with gr.Row():
                    for q in quick_questions[:2]:
                        gr.Button(q).click(
                            lambda x=q: x, outputs=question_input
                        )
                with gr.Row():
                    for q in quick_questions[2:]:
                        gr.Button(q).click(
                            lambda x=q: x, outputs=question_input
                        )

            with gr.Column(scale=4):
                time_info = gr.Textbox(label="⏱️ 响应时间", interactive=False)

        with gr.Row():
            with gr.Column():
                rag_output = gr.Markdown(label="🤖 RAG回答（基于PDF文档）")
            with gr.Column():
                llm_output = gr.Markdown(label="💬 纯LLM回答（无文档参考）")

        context_output = gr.Textbox(label="📚 检索到的相关文档片段", lines=10)

        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, show_ctx_checkbox],
            outputs=[rag_output, llm_output, context_output, time_info]
        )
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, show_ctx_checkbox],
            outputs=[rag_output, llm_output, context_output, time_info]
        )

    return demo


if __name__ == "__main__":
    # 启动时预加载知识库
    init_store()
    demo = build_ui()
    # 自动查找可用端口（7860 起，冲突则依次递增）
    port = find_available_port(7860)
    print(f"[系统] 使用端口: {port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True
    )
