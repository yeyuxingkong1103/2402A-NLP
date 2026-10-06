# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""Gradio 交互界面：文字/语音输入、RAG 与纯 LLM 对比、知识库管理、反馈机制"""
import json
import os
import shutil
import time
from pathlib import Path

# gradio 默认会向自己的服务器回传匿名统计，网络不通时会在后台线程抛
# ConnectTimeout，把部署日志弄得很吓人。这里关掉。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from config import CACHE_ROOT, FEEDBACK_LOG, I18N, TEST_QUESTIONS, TOP_K
from document import build_chunks, load_pdf
from rag_engine import LLMError, RAGEngine
from vector_store import BGEM3VectorStore

_engine = None
_whisper = None


def L(key):
    """界面文案中英并列（验收标准要求支持中英文）。"""
    zh, en = I18N[key]
    return f"{zh} / {en}"


# ---------- 知识库管理 ----------
def list_kbs():
    return sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir()) \
        if CACHE_ROOT.exists() else []


def load_kb(name):
    global _engine
    if name not in list_kbs():
        return "请选择有效的知识库"
    try:
        store = BGEM3VectorStore()
        if not store.load(CACHE_ROOT / name):
            return f"知识库 {name} 不完整，请重新初始化"
        _engine = RAGEngine(store, top_k=TOP_K)
        return f"已加载知识库：{name} | {len(store.chunks)} 个块"
    except Exception as exc:                                   # noqa: BLE001
        return f"加载失败：{type(exc).__name__}: {exc}"


def delete_kb(name):
    if name not in list_kbs():
        return "请选择有效的知识库", gr.update(choices=list_kbs())
    shutil.rmtree(CACHE_ROOT / name, ignore_errors=True)
    return f"已删除：{name}", gr.update(choices=list_kbs(), value=None)


# ---------- 初始化知识库 ----------
def init_engine(pdf_file, use_cache=True, progress=gr.Progress()):
    global _engine
    if not pdf_file:
        return "请先上传 PDF，再点击【初始化知识库】"

    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        return f"未找到上传的 PDF：{pdf_path}"

    cache_dir = CACHE_ROOT / pdf_path.stem
    try:
        store = BGEM3VectorStore()
        if use_cache and cache_dir.exists():
            progress(0.3, desc="加载缓存索引 ...")
            if store.load(cache_dir):
                _engine = RAGEngine(store, top_k=TOP_K)
                return f"从缓存加载完成：{pdf_path.name} | {len(store.chunks)} 个块"

        progress(0.1, desc="解析 PDF ...")
        pages, tables = load_pdf(pdf_path)
        progress(0.4, desc="文本分块 ...")
        chunks = build_chunks(pages, tables)
        progress(0.6, desc="BGE-M3 编码中 ...")
        store.build(chunks)
        try:
            store.save(cache_dir)
        except OSError as exc:
            print(f"[警告] 缓存保存失败：{exc}")

        _engine = RAGEngine(store, top_k=TOP_K)
        return (f"初始化完成：{pdf_path.name} | {len(pages)} 页文字、"
                f"{len(tables)} 个表格，共 {len(chunks)} 个块")
    except Exception as exc:                                   # noqa: BLE001
        _engine = None
        return f"初始化失败：{type(exc).__name__}: {exc}"


# ---------- 提问 ----------
def ask(question, use_qu, show_ctx, mode):
    if _engine is None:
        return "请先上传 PDF 并点击【初始化知识库】", "", ""
    if not question or not question.strip():
        return "请输入问题 / Please enter a question", "", ""

    try:
        if mode.startswith("RAG"):
            res = _engine.answer(question, use_query_understanding=use_qu)
            ctx = "\n\n".join(f"[第{c['page']}页-{c['type']}] {c['text'][:300]}"
                              for c, _ in res["contexts"]) if show_ctx else ""
            meta = (f"耗时 {res['elapsed']:.2f}s | 意图：{res['intent']} | "
                    f"语言：{res['lang']} | 消歧：{'、'.join(res['ambiguous']) or '无'} | "
                    f"实体：{'、'.join(res['entities']) or '无'} | "
                    f"数字：{'、'.join(res['numbers']) or '无'} | "
                    f"子问题：{len(res['sub_questions'])} 个")
        else:
            res = _engine.answer_without_rag(question)
            ctx = ""
            meta = f"耗时 {res['elapsed']:.2f}s | 纯 LLM 回答（未检索文档）"
    except LLMError as exc:
        return f"[大模型调用失败] {exc}", "", ""
    except Exception as exc:                                   # noqa: BLE001
        return f"[出错] {type(exc).__name__}: {exc}", "", ""

    return res["answer"], ctx, meta


# ---------- 反馈机制 ----------
def feedback(kind, question, answer):
    with FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rating": kind, "question": question or "",
            "answer": (answer or "")[:500],
        }, ensure_ascii=False) + "\n")
    return f"{L('feedback_ok')}（{kind}）"


# ---------- 语音转文字 ----------
def transcribe_audio(audio_path):
    global _whisper
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return "[未安装 faster-whisper：pip install faster-whisper]"
    if _whisper is None:                       # 只加载一次，避免每次录音都重载模型
        _whisper = WhisperModel("small", device="auto", compute_type="int8")
    try:
        segments, _ = _whisper.transcribe(audio_path)   # 不指定语言，中英自动识别
        return "".join(s.text for s in segments).strip()
    except Exception as exc:                                   # noqa: BLE001
        return f"[语音识别失败] {exc}"


# ---------- 界面 ----------
def build_ui():
    choices = [f"{q['id']} | {q['question']}" for q in TEST_QUESTIONS]
    rag_label, llm_label = L("mode_rag"), L("mode_llm")

    with gr.Blocks(title="基于PDF文档的问答系统") as demo:
        gr.Markdown(
            "# 基于 PDF 文档的问答系统\n"
            "向量模型：本地 BGE-M3（Dense+Sparse 混合检索） | "
            "工单编号：人工智能NLP-RAG-基于PDF文档的问答系统"
        )

        with gr.Row():
            pdf_in = gr.File(label=L("upload"), file_types=[".pdf"], file_count="single")
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

        with gr.Row():
            audio_in = gr.Audio(label=L("audio"), sources=["microphone"],
                                type="filepath", scale=2)
            with gr.Column(scale=1):
                use_qu = gr.Checkbox(value=True, label="Query 理解 / Query understanding")
                show_ctx = gr.Checkbox(value=True, label="显示上下文 / Show context")

        ask_btn = gr.Button(L("ask"), variant="primary")
        answer = gr.Textbox(label=L("answer"), lines=8)
        context = gr.Textbox(label=L("context"), lines=8)
        meta = gr.Textbox(label=L("meta"))

        with gr.Row():
            good_btn = gr.Button(L("good"))
            bad_btn = gr.Button(L("bad"))
            fb_msg = gr.Textbox(label="反馈 / Feedback", interactive=False, scale=3)

        inputs = [question, use_qu, show_ctx, mode]
        outputs = [answer, context, meta]
        ask_btn.click(ask, inputs=inputs, outputs=outputs)
        question.submit(ask, inputs=inputs, outputs=outputs)

        test_dropdown.change(
            lambda c: c.split("|", 1)[1].strip() if c else "", [test_dropdown], [question])
        audio_in.change(transcribe_audio, [audio_in], [question])
        init_btn.click(init_engine, [pdf_in, use_cache], [status])
        kb_load.click(load_kb, [kb_dropdown], [kb_msg])
        kb_del.click(delete_kb, [kb_dropdown], [kb_msg, kb_dropdown])
        good_btn.click(lambda q, a: feedback("good", q, a), [question, answer], [fb_msg])
        bad_btn.click(lambda q, a: feedback("bad", q, a), [question, answer], [fb_msg])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
