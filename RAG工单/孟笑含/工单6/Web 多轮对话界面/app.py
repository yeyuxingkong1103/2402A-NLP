# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：用户交互界面（Gradio 6.x + 混合检索配置）
功能：可配置检索模式 + 重排算法 + Top-K + 多轮对话
"""

import gradio as gr
from rag_qa_system import RAGQASystem
from conversation import Conversation


def create_ui(system: RAGQASystem):

    def chat_interface(question, history, conv_state,
                       mode, use_rerank, rerank_method, top_k):
        if not question.strip():
            return history, "", conv_state

        # 应用配置
        system.set_retrieval_mode(mode=mode, use_rerank=use_rerank,
                                    rerank_method=rerank_method)
        system.retrieval_config.rerank_top_k = int(top_k)

        if conv_state is None:
            conv_state = Conversation()

        r = system.answer_with_history(question, conv_state)
        answer = r["answer"]

        rewrite_info = ""
        if r.get("rewritten_query") and r["rewritten_query"] != question:
            rewrite_info = f"\n\n🔍 指代消解：{r['rewritten_query']}"

        pages = set(c["page"] for c in r.get("retrieved_contexts", []))
        if pages:
            answer += f"\n\n📚 参考来源：第 {', '.join(map(str, sorted(pages)))} 页"
        answer += f"{rewrite_info}\n\n⚙️ 检索模式：{mode} | 重排：{rerank_method}\n⏱️ 响应时间：{r['response_time']}秒"

        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", conv_state

    def clear_session():
        return [], "", None

    with gr.Blocks(title="基于PDF文档的问答系统（混合检索版）") as demo:
        gr.Markdown("# 基于PDF文档的问答系统（混合检索版）")
        gr.Markdown("### 工单编号：人工智能NLP-RAG-混合检索任务")

        conv_state = gr.State(None)

        with gr.Row():
            # 左侧：检索配置面板
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 检索配置")
                mode = gr.Radio(
                    choices=["vector", "fulltext", "hybrid"],
                    value="hybrid",
                    label="检索模式",
                    info="向量 / 全文 / 混合（推荐）"
                )
                use_rerank = gr.Checkbox(value=True, label="启用 Rerank 模型")
                rerank_method = gr.Dropdown(
                    choices=["cross_encoder", "tfidf", "adaptive"],
                    value="cross_encoder",
                    label="重排算法"
                )
                top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1,
                                    label="Top K")
                clear_btn = gr.Button("清空会话")

            # 右侧：对话区
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="对话记录")
                with gr.Row():
                    msg = gr.Textbox(label="输入问题", placeholder="支持指代词（他/这个公司/那XXX呢？）...", scale=4)
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                        "武汉兴图新科电子股份有限公司注册资本是多少？",
                        "那武汉力源信息技术股份有限公司呢？",
                        "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？",
                    ],
                    inputs=msg
                )

        inputs = [msg, chatbot, conv_state, mode, use_rerank, rerank_method, top_k]
        outputs = [chatbot, msg, conv_state]
        msg.submit(chat_interface, inputs, outputs)
        submit_btn.click(chat_interface, inputs, outputs)
        clear_btn.click(clear_session, outputs=[chatbot, msg, conv_state])

    return demo


if __name__ == "__main__":
    system = RAGQASystem([
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ])
    demo = create_ui(system)
    # AutoDL 开放端口 6006
    demo.launch(server_name="0.0.0.0", server_port=6006,
                share=False, show_error=True, theme=gr.themes.Soft())
