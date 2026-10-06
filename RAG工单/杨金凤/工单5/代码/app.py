# ============================================================
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# 项目名称：PDF文档的Query理解优化
# 文件：app.py
# 说明：Gradio 多轮对话界面（tuples 格式，兼容 Gradio 4/5）
# ============================================================
import gradio as gr
from rag_engine import RAGEngine

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/2_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)


def user_submit(user_message, history):
    """用户提交：把用户消息加入历史"""
    if not user_message or not user_message.strip():
        return "", history
    history = history or []
    history = history + [[user_message, None]]
    return "", history


def bot_respond(history):
    """助手响应：从历史中提取用户问题，调用 RAG 引擎"""
    if not history:
        return history
    user_message = history[-1][0]

    # 提取历史对（仅已完成问答的对）
    engine_history = []
    for pair in history[:-1]:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            q, a = pair
            if q and a:
                engine_history.append((q, a))

    answer, _ = engine.ask(user_message, history=engine_history if engine_history else None)
    history[-1][1] = answer
    return history


def clear_history():
    return None


with gr.Blocks(title="基于《招股说明书》的 RAG 问答系统（多轮）") as demo:
    gr.Markdown("## 基于《招股说明书》的 RAG 问答系统（多轮对话版）")
    gr.Markdown(
        "基于大语言模型与检索增强生成（RAG）技术，支持**多轮对话**（如「他参与的哪个工程？」「那力源呢？」）。\n\n"
        "**支持中英文问答**：中文或英文提问均可。"
    )

    chatbot = gr.Chatbot(
        label="对话",
        height=500,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="请输入问题 / Enter your question",
            placeholder="例如：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
            scale=5,
            lines=1,
        )
        submit_btn = gr.Button("提交 / Submit", scale=1)

    clear_btn = gr.Button("清空对话 / Clear")

    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    clear_btn.click(clear_history, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)