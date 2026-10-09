# 工单编号：人工智能NLP-RAG-金融问答系统部署
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
# 这里只导入 app.py 自己用得到的。RERANK_ALPHA / RERANK_METHOD /
# RETRIEVAL_MODE / rerank_options 是界面（ui.py）用的，由 ui.py 自己导入 ——
# 界面拆分时它们被留在了这里，成了永远用不到的死导入。
from config import (ANSWER_DOC, ANSWER_PAGES, CACHE_ROOT, FEEDBACK_LOG,
                    SERVER_NAME, SERVER_PORT,
                    HYBRID_FUSION, I18N, TEST_QUESTIONS, TOP_K)
from dialogue import Conversation
from document import build_chunks, company_name, load_pdf
from image_parser import parse_pdf_images
from rag_engine import LLMError, MultiDocEngine, RAGEngine
from vector_store import BGEM3VectorStore

_engine = None
_whisper = None
_conv = Conversation()      # 多轮对话上下文（工单5）
_last_chunks = []           # 上一轮用到的文档块，反馈时要记下来（工单6）


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
def init_engine(pdf_files, use_cache=True, progress=gr.Progress()):
    """初始化知识库：支持同时加载多份 PDF，按公司名路由检索。"""
    global _engine
    if not pdf_files:
        return "请先上传 PDF，再点击【初始化知识库】"
    if isinstance(pdf_files, (str, Path)):
        pdf_files = [pdf_files]

    multi, built = MultiDocEngine(), []
    for item in pdf_files:
        pdf_path = Path(item if isinstance(item, str) else item.name)
        if not pdf_path.exists():
            return f"未找到上传的 PDF：{pdf_path}"
        company = company_name(pdf_path)
        try:
            store = BGEM3VectorStore()
            cache_dir = CACHE_ROOT / pdf_path.stem
            if use_cache and cache_dir.exists():
                progress(0.3, desc=f"加载缓存 {pdf_path.name} ...")
                if not store.load(cache_dir):
                    store = _build_store(pdf_path, cache_dir, progress)
            else:
                store = _build_store(pdf_path, cache_dir, progress, use_cache)
            multi.add(company, RAGEngine(store, top_k=TOP_K))
            built.append(f"{company}({len(store.chunks)}块)")
        except Exception as exc:                               # noqa: BLE001
            _engine = None
            return f"初始化失败：{pdf_path.name} -> {type(exc).__name__}: {exc}"

    _engine = multi
    _conv.companies = [c for c, _ in multi.entries]
    _conv.reset()
    return "初始化完成：" + "；".join(built)


def _build_store(pdf_path, cache_dir, progress, use_cache=True):
    progress(0.1, desc=f"解析 {pdf_path.name} ...")
    pages, tables = load_pdf(pdf_path)
    # 工单4：图表页交多模态模型转成文字，否则图里的信息检索不到
    progress(0.3, desc="图像内容解析中 ...")
    images = parse_pdf_images(pdf_path, cache_dir=cache_dir,
                              progress=lambda m: progress(0.3, desc=m))
    progress(0.6, desc="BGE-M3 编码中 ...")
    store = BGEM3VectorStore()
    store.build(build_chunks(pages, tables) + images)
    if use_cache:
        try:
            store.save(cache_dir)
        except OSError as exc:
            print(f"[警告] 缓存保存失败：{exc}")
    return store



# ---------- 检索策略（工单6）----------
MODES = [("vector", "向量检索（召回+重排）"), ("fulltext", "全文检索"),
         ("hybrid", "混合检索")]


def _apply_strategy(mode, alpha, rerank_name):
    """把界面选的检索策略应用到所有已加载的知识库引擎上。"""
    if _engine is None:
        return
    for _, eng in getattr(_engine, "entries", []):
        eng.mode = mode
        eng.alpha = alpha
        if eng.retriever.rerank_name != rerank_name:
            eng.retriever.rerank_name = rerank_name
            eng.retriever._reranker = None      # 换重排算法要丢掉旧的实例
    if not getattr(_engine, "entries", None) and hasattr(_engine, "mode"):
        _engine.mode, _engine.alpha = mode, alpha



# ---------- 提问 ----------
def _precision_note(question, contexts):
    """产出物要求演示时显示「检索精确度」。

    精确度 = 召回块里落在标准答案页上的比例。只有工单指定的问题有标准答案页，
    其它自由提问没有可比对的基准，返回空串不显示。

    知识库是 9 份年报合一，页码会跨文档重复，所以必须按**文档 + 页码**
    一起比对；块上没有 doc 字段时（自己上传单份 PDF 建的库）退化为只比页码。
    """
    qid = next((q["id"] for q in TEST_QUESTIONS
                if q["question"].strip() == (question or "").strip()), None)
    if qid is None or qid not in ANSWER_PAGES or not contexts:
        return ""
    want_doc, want = ANSWER_DOC.get(qid), set(ANSWER_PAGES[qid])
    pairs = [(c.get("doc"), c["page"]) for c, _ in contexts
             if want_doc is None or c.get("doc") in (None, want_doc)]
    hit = [p for _, p in pairs if p in want]
    return (f" | 检索精确度 {len(hit)}/{len(pairs)} = "
            f"{len(hit) / len(pairs):.0%}"
            f"（答案页 {sorted(want)}，实际命中 {sorted(set(hit))}）")


def ask(question, use_qu, show_ctx, mode, strategy, alpha, rerank_name,
        history=None):
    """回答一轮提问。

    工单5：先把问题交给对话上下文改写（指代消解 / 省略补全），
    用改写后的自足问题去检索；界面上同时保留原始问句，方便对照。
    """
    if _engine is None:
        return "请先上传 PDF 并点击【初始化知识库】", "", "", history or []
    if not question or not question.strip():
        return "请输入问题 / Please enter a question", "", "", history or []

    _apply_strategy(strategy, alpha, rerank_name)
    global _last_chunks
    turn = _conv.resolve(question)
    resolved = turn["resolved"] or question
    history = list(history or [])

    try:
        if mode.startswith("RAG"):
            res = _engine.answer(resolved, use_query_understanding=use_qu)
            ctx = "\n\n".join(f"[第{c['page']}页-{c['type']}] {c['text'][:300]}"
                              for c, _ in res["contexts"]) if show_ctx else ""
            meta = (f"耗时 {res['elapsed']:.2f}s | 意图：{res['intent']} | "
                    f"语言：{res['lang']} | 消歧：{'、'.join(res['ambiguous']) or '无'} | "
                    f"实体：{'、'.join(res['entities']) or '无'} | "
                    f"数字：{'、'.join(res['numbers']) or '无'} | "
                    f"子问题：{len(res['sub_questions'])} 个 | "
                    f"检索策略：{strategy}"
                    f"（alpha={alpha:.1f}，融合={HYBRID_FUSION}，重排={rerank_name}） | "
                    f"召回 {res['recall_count']} 块")
            meta += _precision_note(resolved, res["contexts"])
        else:
            res = _engine.answer_without_rag(resolved)
            ctx = ""
            meta = f"耗时 {res['elapsed']:.2f}s | 纯 LLM 回答（未检索文档）"
    except LLMError as exc:
        return f"[大模型调用失败] {exc}", "", "", history
    except Exception as exc:                                   # noqa: BLE001
        return f"[出错] {type(exc).__name__}: {exc}", "", "", history

    # 多轮改写信息只在真的改写过时才提示，避免刷屏
    if turn["strategy"].startswith("原样"):
        meta = f"多轮：{turn['strategy']} | " + meta
    else:
        meta = f"多轮改写：{question} → {resolved}（{turn['strategy']}）\n" + meta

    _last_chunks = [f"{c.get('page')}:{c.get('text', '')[:40]}"
                    for c, _ in res.get("contexts", [])]
    _conv.add_turn(question, resolved, res["answer"])
    history += [(question, res["answer"])]
    return res["answer"], ctx, meta, history


def clear_chat():
    """清空对话上下文（工单5 的多轮对话需要一个重置入口）。"""
    _conv.reset()
    return [], "对话已清空，下一问将作为新话题处理"


# ---------- 反馈机制 ----------
def feedback(kind, question, answer):
    """记录一次反馈。

    要把**这一轮用到的文档块**一起记下来 —— 反馈自适应重排器就是靠它
    给块加权减权的。不记的话那个重排器永远拿不到数据，等于不生效。
    """
    with FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rating": kind, "question": question or "",
            "answer": (answer or "")[:500],
            "chunks": list(_last_chunks),
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


if __name__ == "__main__":
    # 界面在 ui.py 里，放到 __main__ 里导入是为了避开循环依赖：
    # ui 需要本模块的处理函数，本模块只在真正启动时才知道 ui 的存在。
    from ui import build_ui

    # 开启队列并允许并发请求。GPU 前向在 vector_store 里已加锁串行化，
    # 接口层并发不会互相挤爆显存，也能避免高并发下界面被拖死。
    # 监听地址与端口走 config，容器部署时用 HOST / PORT 环境变量改。
    build_ui().queue(max_size=32, default_concurrency_limit=4).launch(
        server_name=SERVER_NAME, server_port=SERVER_PORT)
