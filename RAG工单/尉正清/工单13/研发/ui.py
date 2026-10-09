# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""Gradio 界面搭建

从 app.py 拆出来只为行数：app.py 同时放业务处理函数和界面搭建会超过
300 行的单文件上限。处理函数（知识库管理、提问、反馈、语音）都留在 app.py，
本模块只负责把它们接到控件上。

依赖方向是单向的 ui -> app；app.py 只在 __main__ 里导入本模块，不构成循环。
"""
import gradio as gr

from app import (L, MODES, ask, clear_chat, delete_kb, feedback, init_engine,
                 list_kbs, load_kb, transcribe_audio)
from config import RERANK_ALPHA, RERANK_METHOD, RETRIEVAL_MODE, TEST_QUESTIONS
from rerank import available as rerank_options


# ---------- 界面 ----------
def build_ui():
    choices = [f"{q['id']} | {q['question']}" for q in TEST_QUESTIONS]
    rag_label, llm_label = L("mode_rag"), L("mode_llm")

    with gr.Blocks(title="基于PDF文档的问答系统") as demo:
        gr.Markdown(
            "# 基于 PDF 文档的问答系统\n"
            "向量模型：本地 BGE-M3（Dense+Sparse 混合检索） | "
            "工单编号：人工智能NLP-RAG-功能测试及评估"
        )

        with gr.Row():
            pdf_in = gr.File(label=L("upload"), file_types=[".pdf"], file_count="multiple")
            with gr.Column():
                use_cache = gr.Checkbox(value=True, label="使用缓存 / Use cache")
                init_btn = gr.Button(L("init"), variant="primary")

        status = gr.Textbox(label=L("status"), value="未初始化 / Not initialized",
                            interactive=False)

        # 知识库管理
        with gr.Row():
            kb_dropdown = gr.Dropdown(choices=list_kbs(), label=L("kb_select"),
                                      value=None, scale=3)
            kb_load = gr.Button(L("kb_load"))
            kb_del = gr.Button(L("kb_delete"), variant="stop")
            kb_msg = gr.Textbox(label=L("status"), interactive=False, scale=2)

        test_dropdown = gr.Dropdown(choices=choices, label=L("pick_q"), value=None)

        with gr.Row():
            question = gr.Textbox(label=L("question"), lines=2, scale=4)
            mode = gr.Radio([rag_label, llm_label], value=rag_label,
                            label=L("mode"), scale=2)

        # 检索策略配置（工单6）：策略 / 向量权重 / 重排算法
        with gr.Row():
            strategy = gr.Radio([label for _, label in MODES],
                                value=dict((v, l) for v, l in MODES)[RETRIEVAL_MODE],
                                label="检索策略 / Retrieval strategy", scale=3)
            alpha = gr.Slider(0, 1, value=RERANK_ALPHA, step=0.1,
                              label="向量权重 alpha（混合检索用）/ Vector weight", scale=2)
            rerank_sel = gr.Dropdown([label for _, label in rerank_options()],
                                     value=dict(rerank_options()).get(RERANK_METHOD, "不重排"),
                                     label="重排算法 / Rerank", scale=2)

        with gr.Row():
            audio_in = gr.Audio(label=L("audio"), sources=["microphone"],
                                type="filepath", scale=2)
            with gr.Column(scale=1):
                use_qu = gr.Checkbox(value=True, label="Query 理解 / Query understanding")
                show_ctx = gr.Checkbox(value=True, label="显示上下文 / Show context")

        chat = gr.Chatbot(label="对话历史 / Conversation", height=260)

        with gr.Row():
            ask_btn = gr.Button(L("ask"), variant="primary", scale=3)
            clear_btn = gr.Button("清空对话 / Clear", scale=1)

        answer = gr.Textbox(label=L("answer"), lines=8)
        context = gr.Textbox(label=L("context"), lines=8)
        meta = gr.Textbox(label=L("meta"))

        with gr.Row():
            good_btn = gr.Button(L("good"))
            bad_btn = gr.Button(L("bad"))
            fb_msg = gr.Textbox(label="反馈 / Feedback", interactive=False, scale=3)

        _mode_map = dict((label, value) for value, label in MODES)
        _rank_map = dict((label, value) for value, label in rerank_options())
        inputs = [question, use_qu, show_ctx, mode,
                  strategy, alpha, rerank_sel, chat]
        outputs = [answer, context, meta, chat]
        def _ask(*a):
            q, use_qu, show_ctx, md, strat, al, rk = a[:7]
            hist = a[7] if len(a) > 7 else []
            return ask(q, use_qu, show_ctx, md, _mode_map.get(strat, "hybrid"),
                       al, _rank_map.get(rk, "none"), hist)

        ask_btn.click(_ask, inputs=inputs, outputs=outputs)
        question.submit(_ask, inputs=inputs, outputs=outputs)
        clear_btn.click(clear_chat, None, [chat, status])

        test_dropdown.change(
            lambda c: c.split("|", 1)[1].strip() if c else "", [test_dropdown], [question])
        audio_in.change(transcribe_audio, [audio_in], [question])
        init_btn.click(init_engine, [pdf_in, use_cache], [status])
        kb_load.click(load_kb, [kb_dropdown], [kb_msg])
        kb_del.click(delete_kb, [kb_dropdown], [kb_msg, kb_dropdown])
        good_btn.click(lambda q, a: feedback("good", q, a), [question, answer], [fb_msg])
        bad_btn.click(lambda q, a: feedback("bad", q, a), [question, answer], [fb_msg])

    return demo
