# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
模块：用户交互界面（Gradio 6.x 多轮版）
功能：支持多轮对话 + 指代消解 + 中英双语
"""

import gradio as gr
from rag_qa_system import RAGQASystem
from conversation import Conversation


def create_ui(system: RAGQASystem):

    def chat_interface(question: str, history: list, conv_state):
        if not question.strip():
            return history, "", conv_state

        if conv_state is None:
            conv_state = Conversation()

        r = system.answer_with_history(question, conv_state)
        answer = r["answer"]

        # 显示改写信息（如果改写了）
        rewrite_info = ""
        if r.get("rewritten_query") and r["rewritten_query"] != question:
            rewrite_info = f"\n\n🔍 指代消解：{r['rewritten_query']}"

        pages = set(c["page"] for c in r.get("retrieved_contexts", []))
        if pages:
            answer += f"\n\n📚 参考来源：第 {', '.join(map(str, sorted(pages)))} 页"
        answer += f"{rewrite_info}\n\n⏱️ 响应时间：{r['response_time']}秒"

        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", conv_state

    def clear_session():
        return [], "", None

    with gr.Blocks(title="基于PDF文档的问答系统（多轮版）") as demo:
        gr.Markdown("# 基于PDF文档的问答系统（多轮对话版）")
        gr.Markdown("### 工单编号：人工智能NLP-RAG-Query理解优化任务")

        conv_state = gr.State(None)

        with gr.Tab("多轮对话"):
            chatbot = gr.Chatbot(height=500, label="对话记录")
            with gr.Row():
                msg = gr.Textbox(label="输入问题", placeholder="支持指代词（他/这个公司/那XXX呢？）...", scale=4)
                submit_btn = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("清空会话", scale=1)

            msg.submit(chat_interface, [msg, chatbot, conv_state], [chatbot, msg, conv_state])
            submit_btn.click(chat_interface, [msg, chatbot, conv_state], [chatbot, msg, conv_state])
            clear_btn.click(clear_session, outputs=[chatbot, msg, conv_state])

            gr.Examples(
                examples=[
                    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                    "他参与的哪个工程荣获了国家科技进步一等奖？",
                    "这个公司的法定代表人是谁？",
                    "那武汉力源信息技术股份有限公司呢？",
                    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
                ],
                inputs=msg
            )

        with gr.Tab("工单说明"):
            gr.Markdown("""
## 工单5 — Query理解优化

### 核心能力
- **多轮对话**：支持连续对话，保留上下文
- **指代消解**：
  - "他/她/它" → 上一轮提到的公司
  - "这个公司/该企业" → 上一轮提到的公司
  - "那XXX呢？" → 继承上一轮问题类型，替换公司
- **多语言**：中英双语问答

### 测试对话示例
1. 报告期内，武汉兴图新科来自军用领域的收入分别是多少？
2. **他**参与的哪个工程荣获了国家科技进步一等奖？
3. **这个公司**的法定代表人是谁？
4. **那武汉力源信息技术股份有限公司呢**？
5. 武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？
            """)

    return demo


if __name__ == "__main__":
    system = RAGQASystem([
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ])
    demo = create_ui(system)
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, show_error=True, theme=gr.themes.Soft())
