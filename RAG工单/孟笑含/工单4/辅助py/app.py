# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：用户交互界面（Gradio 6.x）
功能：Web 界面 + 中英切换 + Reranker 展示
"""

import gradio as gr
from rag_qa_system import RAGQASystem


def create_ui(system: RAGQASystem):
    """创建 Gradio 界面"""

    def chat_interface(question: str, history: list):
        if not question.strip():
            return history, ""
        r = system.answer(question, use_rag=True)
        answer = r["answer"]
        pages = set(c["page"] for c in r.get("retrieved_contexts", []))
        if pages:
            answer += "\n\n参考来源：第 " + ", ".join(map(str, sorted(pages))) + " 页"
        answer += "\n\n响应时间：" + str(r["response_time"]) + "秒"
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    with gr.Blocks(title="基于PDF文档的问答系统（优化版）") as demo:
        gr.Markdown("# 基于PDF文档的问答系统（优化版）")
        gr.Markdown("### 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化")

        with gr.Tab("问答"):
            chatbot = gr.Chatbot(height=500, label="对话记录")
            with gr.Row():
                msg = gr.Textbox(label="输入问题", placeholder="请输入问题（支持中英文）...", scale=4)
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
            submit_btn.click(chat_interface, [msg, chatbot], [chatbot, msg])
            gr.Examples(
                examples=[
                    "武汉兴图新科电子股份有限公司注册资本是多少？",
                    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
                    "What is the registered capital of Wuhan Xingtu Xinke Electronics?",
                    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                ],
                inputs=msg
            )

        with gr.Tab("优化说明"):
            gr.Markdown("""
## 优化方法（5 项）

| 方法 | 效果 |
|------|------|
| 1. 分块优化 | 页码前缀 + 章节识别 |
| 2. 混合检索 | bge-m3 向量 + BM25 + RRF |
| 3. Reranker 精排 | Top-20 → Top-3 |
| 4. Mock 升级 | 跨块汇总 + 数值提取 |
| 5. 多语言 | 中英通用 bge-m3 |
            """)

    return demo


if __name__ == "__main__":
    system = RAGQASystem(["./data/招股说明书1.pdf", "./data/招股说明书2.pdf"])
    demo = create_ui(system)
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True, theme=gr.themes.Soft())
