# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
app/streamlit_app_v4.py —— 工单四 图像内容解析与检索 Streamlit 界面（新增文件，不改动 v3）

启动：streamlit run app/streamlit_app_v4.py --server.port 8504
功能：中英文切换 / 文档选择 / 答案+引用(文本/表格/图像) / Markdown表格 /
     图像展示(caption+OCR+VQA) / 图像解析结果查看页 / 响应时间 / 点赞点踩评论
"""
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()                                   # 工单四：加载 .env（DEEPSEEK_API_KEY）

st.set_page_config(page_title="图像内容解析及检索优化 v4", page_icon="🖼️",
                   layout="wide")

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"      # 工单四：工单编号
DOCS = ["全部", "招股说明书1", "招股说明书2"]

# ---------------- 工单四：中英文 i18n ----------------
I18N = {
    "中文": {"page_qa": "💬 智能问答", "page_img": "🖼️ 图像解析结果查看",
             "doc": "选择文档", "question": "请输入您的问题：",
             "ask": "提交问题", "answer": "答案", "refs": "引用来源",
             "ref_text": "文本", "ref_table": "表格", "ref_image": "图像",
             "latency": "响应时间(ms)", "retrieval": "检索(ms)", "llm": "生成(ms)",
             "fb_up": "👍 点赞", "fb_down": "👎 点踩", "fb_comment": "评论",
             "fb_saved": "反馈已保存，感谢！", "caption": "图像描述(caption)",
             "ocr": "图内文字(OCR)", "vqa": "图表问答(VQA)", "page_col": "页码",
             "path_col": "图像路径", "empty": "知识库暂无该图像解析结果",
             "examples": "快捷问题"},
    "English": {"page_qa": "💬 Q&A", "page_img": "🖼️ Image Parsing Viewer",
                "doc": "Select document", "question": "Enter your question:",
                "ask": "Ask", "answer": "Answer", "refs": "References",
                "ref_text": "Text", "ref_table": "Table", "ref_image": "Image",
                "latency": "Latency(ms)", "retrieval": "Retrieval(ms)",
                "llm": "Generation(ms)", "fb_up": "👍 Like", "fb_down": "👎 Dislike",
                "fb_comment": "Comment", "fb_saved": "Feedback saved. Thanks!",
                "caption": "Caption", "ocr": "OCR text", "vqa": "Chart VQA",
                "page_col": "Page", "path_col": "Image path",
                "empty": "No parsed images for this document yet",
                "examples": "Quick questions"},
}
# 工单四：快捷问题（覆盖 id5/id6 图像题、时序文本题、表格金标题与英文题，
# 对应 16 题验收集中的图像/文本/表格/中英文四类场景）
QUICK_QA = [
    ("id5 组织结构图", "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"),
    ("id6 增长图", "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"),
    ("文本题示例", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"),
    ("表格题示例", "本次发行的募集资金总额是多少？"),
    ("English example", "How many shares are issued in this offering?"),
]


@st.cache_resource(show_spinner="加载 RAG v4 引擎（首次约1-2分钟）...")
def load_engine():
    """工单四：缓存 RAGEngineV4（复用工单三文本/表格链路 + 图像三路检索）"""
    from src.rag_engine_v4 import RAGEngineV4
    engine = RAGEngineV4(top_k=5)
    # 工单四：启动时串行预热 bge/reranker/CLIP，避免首问并行懒加载
    # 重复载入模型导致 8GB 显存超卖（人工智能NLP-RAG-图像内容解析及检索优化）
    engine.warmup()
    return engine


def save_feedback(payload: dict) -> str:
    """工单四：点赞/点踩/评论 → data/feedback/feedback_*.json"""
    fb_dir = Path("data/feedback")
    fb_dir.mkdir(parents=True, exist_ok=True)
    path = fb_dir / f"feedback_{datetime.now():%Y%m%d}.json"
    records = []
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
    records.append(payload)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return str(path)


def render_images(images: list, t: dict):
    """工单四：图像引用展示（图像 + caption/OCR/VQA 折叠区）"""
    for im in images:
        ref_id, path = im.get("ref_id", ""), im.get("path", "")
        with st.expander(f"{ref_id} | {im.get('doc_id','')} 第{im.get('page','?')}页 "
                         f"| score={im.get('score', 0)}", expanded=True):
            p = Path(path)
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.warning(f"{t['path_col']}: {path}（文件缺失）")
            if im.get("caption"):
                st.markdown(f"**{t['caption']}**：{im['caption']}")
            if im.get("ocr_text"):
                st.markdown(f"**{t['ocr']}**：\n\n{im['ocr_text'][:500]}")
            if im.get("vqa_text"):
                st.markdown(f"**{t['vqa']}**：{im['vqa_text'][:500]}")


def render_table_ref(t_ref: dict, t: dict):
    """工单四：表格引用 Markdown 展示（表格类问题）"""
    with st.expander(f"{t_ref.get('ref_id','表')} | {t_ref.get('doc_id','')} "
                     f"第{t_ref.get('page','?')}页 | score={t_ref.get('score',0)}"):
        content = t_ref.get("preview", "")
        st.markdown(f"```\n{content}\n```") if content else st.info("无内容")


# ================= 工单四：侧栏 =================
st.sidebar.title("PDF 智能问答 v4")
st.sidebar.caption(f"工单编号：{WORK_ORDER}")
lang = st.sidebar.radio("🌐 Language / 语言", list(I18N.keys()), horizontal=True)
t = I18N[lang]
page = st.sidebar.radio("", [t["page_qa"], t["page_img"]])
doc_choice = st.sidebar.selectbox(t["doc"], DOCS)
doc_id = None if doc_choice == "全部" else doc_choice

# ================= 工单四：页面一 智能问答 =================
if page == t["page_qa"]:
    st.header(t["page_qa"])
    st.sidebar.divider()
    st.sidebar.subheader(t["examples"])
    # 工单四：radio on_change 显式回填 preset；text_area 带 session_state key，
    # 保证用户手动键入的问题（如英文题、表格金标题）提交时不被 preset 覆盖
    def _apply_quick():
        st.session_state["v4_query"] = dict(QUICK_QA)[st.session_state["v4_quick"]]

    quick = st.sidebar.radio(t["examples"], [q[0] for q in QUICK_QA],
                             label_visibility="collapsed", key="v4_quick",
                             on_change=_apply_quick)
    default_q = dict((k, v) for k, v in QUICK_QA)[quick]

    query = st.text_area(t["question"], value=default_q, height=90, key="v4_query")
    if st.button(t["ask"], type="primary"):
        if not query.strip():
            st.warning(t["question"])
        else:
            engine = load_engine()
            t0 = time.time()
            with st.spinner("检索文本/表格/图像并生成答案..."):
                result = engine.ask(query, doc_id=doc_id, use_image=True)
            total_ms = (time.time() - t0) * 1000

            st.subheader(t["answer"])
            st.markdown(result["answer"])

            # 工单四：图像引用（图像+caption/OCR/VQA）
            images = result.get("retrieved_images", [])
            if images:
                st.subheader(f"{t['refs']} · {t['ref_image']}")
                render_images([{**im, "ref_id": f"图{i}"} for i, im in
                               enumerate(images, 1)], t)

            # 工单四：表格引用（Markdown）
            tables = result.get("retrieved_tables", [])
            if tables:
                st.subheader(f"{t['refs']} · {t['ref_table']}")
                for i, tb in enumerate(tables, 1):
                    render_table_ref({**tb, "ref_id": f"表{i}"}, t)

            # 工单四：文本引用（页码+原文片段）
            texts = result.get("retrieved_text_chunks", [])
            if texts:
                st.subheader(f"{t['refs']} · {t['ref_text']}")
                for i, c in enumerate(texts, 1):
                    with st.expander(f"资料{i} | {c.get('doc_id','')} "
                                     f"{t['page_col']}{c.get('page','?')} "
                                     f"| score={c.get('rrf_score', c.get('rerank_score', 0))}"):
                        st.markdown((c.get("content") or "")[:400])

            # 工单四：响应时间（总/检索/生成）
            b = result.get("breakdown", {})
            c1, c2, c3 = st.columns(3)
            c1.metric(t["latency"], f"{total_ms:.0f}")
            c2.metric(t["retrieval"], f"{b.get('retrieve_ms', 0):.0f}")
            c3.metric(t["llm"], f"{b.get('llm_ms', 0):.0f}")

            # 工单四：点赞/点踩/评论
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 4])
            comment = col3.text_input(t["fb_comment"], key="fb_comment")
            if col1.button(t["fb_up"]):
                p = save_feedback({"ts": datetime.now().isoformat(), "lang": lang,
                                   "query": query, "answer": result["answer"],
                                   "rating": "up", "comment": comment,
                                   "latency_ms": round(total_ms, 1),
                                   "work_order": WORK_ORDER})
                st.success(f"{t['fb_saved']} ({p})")
            if col2.button(t["fb_down"]):
                p = save_feedback({"ts": datetime.now().isoformat(), "lang": lang,
                                   "query": query, "answer": result["answer"],
                                   "rating": "down", "comment": comment,
                                   "latency_ms": round(total_ms, 1),
                                   "work_order": WORK_ORDER})
                st.success(f"{t['fb_saved']} ({p})")

# ================= 工单四：页面二 图像解析结果查看 =================
else:
    st.header(t["page_img"])
    view_doc = st.selectbox(t["doc"], DOCS[1:])
    # 工单四：优先读解析 JSON（caption/VQA），缺则回退提取清单
    parsed_path = Path(f"data/image_descriptions/{view_doc}_images_parsed.json")
    manifest_path = Path(f"data/images/{view_doc}_images.json")
    images = []
    if parsed_path.exists():
        images = json.loads(parsed_path.read_text(encoding="utf-8")).get("images", [])
    elif manifest_path.exists():
        images = json.loads(manifest_path.read_text(encoding="utf-8")).get("images", [])
    if not images:
        st.info(t["empty"])
    else:
        st.caption(f"{len(images)} 张图像")
        page_no = st.selectbox(t["page_col"],
                               sorted({im.get("page", 0) for im in images}))
        for im in [x for x in images if x.get("page", 0) == page_no]:
            with st.expander(f"{im.get('image_id','')} | "
                             f"{t['page_col']}{im.get('page','?')} "
                             f"| {im.get('width','?')}x{im.get('height','?')}"):
                p = Path(im.get("path", ""))
                if p.exists():
                    st.image(str(p), use_container_width=True)
                if im.get("caption"):
                    st.markdown(f"**{t['caption']}**：{im['caption']}")
                if im.get("ocr_text"):
                    st.markdown(f"**{t['ocr']}**：\n\n{im['ocr_text'][:600]}")
                for qa in (im.get("vqa_qa") or [])[:6]:
                    st.markdown(f"- **VQA-Q**: {qa.get('q','')}  \n"
                                f"  **VQA-A**: {qa.get('a','')}")
