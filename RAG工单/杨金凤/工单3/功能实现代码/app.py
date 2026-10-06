# ============================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 项目名称：基于PDF文档的问答系统优化
# 文件：app.py
# 说明：Gradio 问答界面，支持中英文问答、来源展示、反馈机制
# ============================================================
import gradio as gr
from rag_engine import RAGEngine

# 全局加载一次引擎
engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/2_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)


def chat_fn(question):
    if not question or not question.strip():
        return "请输入问题后再提交。", ""
    answer, docs = engine.ask(question)
    if docs:
        sources = "\n\n".join(
            [f"[片段 {i+1}]\n{d.page_content[:200]}..." for i, d in enumerate(docs[:3])]
        )
    else:
        sources = "（本次无检索结果）"
    return answer, sources


def feedback_fn(choice):
    print(f"[用户反馈] {choice}")
    return f"已收到反馈：{choice}"


with gr.Blocks(title="基于《招股说明书》的 RAG 问答系统") as demo:
    gr.Markdown("## 基于《招股说明书1.pdf》的问答系统")
    gr.Markdown(
        "基于大语言模型与检索增强生成（RAG）技术，针对招股说明书内容进行问答。\n\n"
        "**支持中英文问答**：中文或英文提问均可（Supports Chinese and English questions）。"
    )

    with gr.Row():
        question = gr.Textbox(
            label="请输入问题 / Enter your question",
            placeholder="例如：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？\n"
                        "or: What is the registered capital of the company?",
            lines=2,
            scale=4,
        )
        ask_btn = gr.Button("提交 / Submit", scale=1)

    answer = gr.Textbox(label="回答 / Answer", lines=10)

    with gr.Accordion("查看检索到的原文片段 / View retrieved source snippets", open=False):
        sources = gr.Textbox(label="来源片段 / Source snippets", lines=12)

    gr.Markdown("### 回答反馈 / Feedback")
    with gr.Row():
        good_btn = gr.Button("👍 准确 / Accurate")
        bad_btn = gr.Button("👎 不准确 / Inaccurate")
    fb_out = gr.Textbox(label="反馈状态 / Feedback status")

    ask_btn.click(chat_fn, inputs=question, outputs=[answer, sources])
    question.submit(chat_fn, inputs=question, outputs=[answer, sources])
    good_btn.click(lambda: feedback_fn("准确"), outputs=fb_out)
    bad_btn.click(lambda: feedback_fn("不准确"), outputs=fb_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)