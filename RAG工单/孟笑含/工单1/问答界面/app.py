# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：用户交互界面
功能：基于Gradio的Web界面
"""

import gradio as gr
from rag_qa_system import RAGQASystem


def create_ui(system: RAGQASystem):
    """创建Gradio界面"""
    
    def chat_interface(question: str, history: list):
        """聊天接口（Gradio 6.x messages 格式）"""
        if not question.strip():
            return history, ""

        result = system.answer(question, use_rag=True)
        answer = result["answer"]

        sources = ""
        if "retrieved_contexts" in result:
            pages = set(ctx["page"] for ctx in result["retrieved_contexts"])
            sources = f"\\n\\n📚 参考来源：第 {', '.join(map(str, sorted(pages)))} 页"

        response = f"{answer}{sources}\\n\\n⏱️ 响应时间：{result['response_time']}秒"

        # Gradio 6.x messages 格式
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        return history, ""


    def upload_pdf(file):
        """上传PDF并重建索引"""
        if file is None:
            return "请选择PDF文件"
        # 此处可扩展为动态加载
        return f"已接收文件：{file.name}"
    
    with gr.Blocks(title="基于PDF文档的问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📄 基于PDF文档的问答系统")
        gr.Markdown("### 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统")
        
        with gr.Tab("💬 问答"):
            chatbot = gr.Chatbot(height=500, label="对话记录")
            with gr.Row():
                msg = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    scale=4
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
            submit_btn.click(chat_interface, [msg, chatbot], [chatbot, msg])
            
            gr.Examples(
                examples=[
                    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
                    "武汉兴图新科电子股份有限公司注册资本是多少？",
                    "武汉兴图新科电子股份有限公司法定代表人是谁？",
                ],
                inputs=msg
            )
        
        with gr.Tab("📤 文档管理"):
            pdf_file = gr.File(label="上传PDF文档", file_types=[".pdf"])
            upload_btn = gr.Button("上传并解析")
            upload_output = gr.Textbox(label="处理结果")
            upload_btn.click(upload_pdf, pdf_file, upload_output)
        
        with gr.Tab("📊 评估报告"):
            gr.Markdown("### RAG vs 纯LLM 对比评估")
            eval_btn = gr.Button("运行批量评估")
            eval_output = gr.Dataframe(
                headers=["问题", "RAG答案", "LLM答案", "胜出方", "提升幅度"],
                label="评估结果"
            )
            
            def run_eval():
                results = system.batch_evaluate([
                    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
                    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
                ])
                data = []
                for r in results:
                    data.append([
                        r["question"],
                        r["rag_answer"][:100],
                        r["llm_answer"][:100],
                        r["comparison"]["winner"],
                        f"{r['comparison']['improvement']:.2%}"
                    ])
                return data
            
            eval_btn.click(run_eval, outputs=eval_output)
    
    return demo




if __name__ == "__main__":
    system = RAGQASystem("./data/招股说明书1.pdf")
    demo = create_ui(system)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
