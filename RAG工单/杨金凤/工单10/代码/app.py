# ============================================================
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 文件：app.py
# 说明：Gradio 多轮对话界面（Gradio 5.x messages 格式）
# ============================================================
import gradio as gr
from rag_engine import RAGEngine

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/2_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/models/BAAI--bge-reranker-base/snapshots/master",
)


def user_submit(user_message, history):
    if not user_message or not user_message.strip():
        return "", history
    history = history or []
    history = history + [{"role": "user", "content": user_message}]
    return "", history


def bot_respond(history, retrieval_mode, hybrid_weight, rerank_method):
    if not history:
        return history
    user_message = history[-1]["content"]

    # 提取历史对（messages 格式）
    engine_history = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            engine_history.append((history[i]["content"], history[i+1]["content"]))
            i += 2
        else:
            i += 1

    answer, _ = engine.ask(
        user_message,
        history=engine_history if engine_history else None,
        retrieval_mode=retrieval_mode,
        hybrid_weight=hybrid_weight,
        rerank_method=rerank_method,
    )
    history = history + [{"role": "assistant", "content": answer}]
    return history


def clear_history():
    return []


with gr.Blocks(title="基于《招股说明书》的 RAG 问答系统（混合检索版）") as demo:
    gr.Markdown("## 基于《招股说明书》的 RAG 问答系统（混合检索版）")
    gr.Markdown(
        "支持**向量检索 / 全文检索 / 混合检索**三种模式，"
        "**BGE / TF-IDF / LLM** 三种重排算法，并支持多轮对话与中英文问答。"
    )

    with gr.Accordion("⚙️ 检索策略配置", open=True):
        with gr.Row():
            retrieval_mode = gr.Radio(
                choices=[("向量检索", "vector"), ("全文检索", "fulltext"), ("混合检索（推荐）", "hybrid")],
                value="hybrid",
                label="检索模式",
            )
            rerank_method = gr.Dropdown(
                choices=[("BGE Reranker", "bge"), ("TF-IDF Reranker", "tfidf"),
                         ("LLM Reranker", "llm"), ("不重排", "none")],
                value="bge",
                label="重排算法",
            )
        hybrid_weight = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
            label="混合检索权重（向量占比，1-该值为 BM25 占比）",
        )

    chatbot = gr.Chatbot(label="对话", height=450)

    with gr.Row():
        msg = gr.Textbox(
            label="请输入问题 / Enter your question",
            placeholder="例如：平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？",
            scale=5, lines=1,
        )
        submit_btn = gr.Button("提交 / Submit", scale=1)

    clear_btn = gr.Button("清空对话 / Clear")

    submit_btn.click(
        user_submit, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot_respond,
        [chatbot, retrieval_mode, hybrid_weight, rerank_method],
        chatbot,
    )
    msg.submit(
        user_submit, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot_respond,
        [chatbot, retrieval_mode, hybrid_weight, rerank_method],
        chatbot,
    )
    clear_btn.click(clear_history, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)