# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
模块：Web 界面（含图谱 Tab）
功能：问答 + 知识图谱可视化 + 多轮对话
"""

import os
import gradio as gr
from rag_qa_system import RAGQASystem
from conversation import Conversation


def create_ui(system: RAGQASystem):

    def chat_interface(question, history, conv_state,
                       mode, use_rerank, rerank_method, top_k, use_graph):
        if not question.strip():
            return history, "", conv_state

        system.set_retrieval_mode(mode=mode, use_rerank=use_rerank, rerank_method=rerank_method)
        system.retrieval_config.rerank_top_k = int(top_k)

        if conv_state is None:
            conv_state = Conversation()

        r = system.answer_with_history(question, conv_state)
        answer = r["answer"]

        # 图谱信息
        if use_graph:
            triples = r.get("graph_triples", [])
            if triples:
                answer += f"\n\n🔗 知识图谱：检索到 {len(triples)} 条三元组"
                for t in triples[:3]:
                    answer += f"\n  • {t['head']} —{t['relation']}→ {t['tail']}"

        pages = set(c["page"] for c in r.get("retrieved_contexts", []))
        if pages:
            answer += f"\n\n📚 参考来源：第 {', '.join(map(str, sorted(pages)))} 页"
        answer += f"\n\n⏱️ 响应时间：{r['response_time']}秒"

        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", conv_state

    def clear_session():
        return [], "", None

    with gr.Blocks(title="Graph RAG 金融问答系统") as demo:
        gr.Markdown("# Graph RAG 金融问答系统")
        gr.Markdown("### 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答")

        conv_state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("💬 问答"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## ⚙️ 配置")
                        mode = gr.Radio(["vector", "fulltext", "hybrid"], value="hybrid", label="检索模式")
                        use_rerank = gr.Checkbox(value=True, label="启用 Rerank")
                        rerank_method = gr.Dropdown(["cross_encoder", "tfidf", "adaptive"], value="cross_encoder", label="重排算法")
                        top_k = gr.Slider(1, 10, value=3, step=1, label="Top K")
                        use_graph = gr.Checkbox(value=True, label="启用 Graph RAG")
                        clear_btn = gr.Button("清空会话")

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500, label="对话记录")
                        with gr.Row():
                            msg = gr.Textbox(label="输入问题", placeholder="支持指代词...", scale=4)
                            submit_btn = gr.Button("发送", variant="primary", scale=1)
                        gr.Examples(
                            examples=[
                                "武汉兴图新科电子股份有限公司注册资本是多少？",
                                "武汉力源信息技术股份有限公司法定代表人是谁？",
                                "那武汉力源信息技术股份有限公司呢？",
                            ],
                            inputs=msg
                        )

                inputs = [msg, chatbot, conv_state, mode, use_rerank, rerank_method, top_k, use_graph]
                outputs = [chatbot, msg, conv_state]
                msg.submit(chat_interface, inputs, outputs)
                submit_btn.click(chat_interface, inputs, outputs)
                clear_btn.click(clear_session, outputs=[chatbot, msg, conv_state])

            with gr.Tab("🕸️ 知识图谱"):
                gr.Markdown("## 知识图谱可视化")
                gr.Markdown("下方为抽取的实体关系交互图。如未显示，请点击 [下载 HTML] 在本地浏览器打开。")
                if os.path.exists("graph.html"):
                    gr.File(value="graph.html", label="📥 下载图谱 HTML")
                    gr.HTML('<iframe src="/gradio_api/file=graph.html" width="100%" height="800px" style="border:none;background:#1a1a1a;"></iframe>')
                else:
                    gr.Markdown("⚠️ graph.html 不存在，请先运行 graph_visualizer.py")

            with gr.Tab("📊 图谱统计"):
                if system.graph_retriever:
                    stats = system.graph_retriever.get_stats()
                    gr.Markdown(f"""
## 知识图谱统计

- **节点数**：{stats['nodes']}
- **边数**：{stats['edges']}

### 实体类型
- 公司、人物、机构、指标、数值、年份、地点、产品、报告

### 关系类型
- 持股、控股、任职、实现、报告、拥有、属于、同比、包含
                    """)

    return demo


if __name__ == "__main__":
    system = RAGQASystem([
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ])
    demo = create_ui(system)
    demo.launch(server_name="0.0.0.0", server_port=6006,
                share=False, show_error=True, theme=gr.themes.Soft())
