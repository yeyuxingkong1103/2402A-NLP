# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
app/streamlit_app_v3.py —— 工单三前端（端口 8503）

启动：streamlit run app/streamlit_app_v3.py --server.port 8503
功能：
  1. 中英文界面切换（侧栏）
  2. 多文档选择（招股说明书1 / 招股说明书2 / 全部）
  3. 表格+文本融合检索（RAGEngineV3 直连，无需 API）
  4. 表格类问题：Markdown 表格展示检索到的表格
  5. 引用来源：文本引用 + 表格引用（页码 + 表格 ID + 相关度）
  6. 响应时间指标（总/检索/生成）
  7. 路由信息：text_only / table_only / hybrid
  8. 14 个测试问题快捷按钮
  9. 表格解析结果查看页面：选 PDF → 浏览解析出的表格
  10. 反馈按钮：点赞/点踩/评论
"""
import json
import os
import sys
import time
from pathlib import Path

# 工单三：项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="PDF 表格解析与检索优化（工单三）",
    page_icon="📊", layout="wide",
)

# 工单三：环境变量（显存优化）
os.environ.setdefault("RAG_EMBED_DEVICE", "cuda")
os.environ.setdefault("RAG_EMBED_BATCH_SIZE", "8")
os.environ.setdefault("RAG_EMBED_MAX_SEQ", "512")

from dotenv import load_dotenv
load_dotenv()

# ================= i18n =================
I18N = {
    "zh": {
        "title": "📊 PDF 表格解析与检索优化",
        "sub": "工单三：表格解析 · 表格感知检索 · 多文档 · 中英文 · 重排序",
        "lang_label": "界面语言 / Language",
        "answer_lang": "回答语言",
        "doc_select": "选择文档",
        "page_select": "页面导航",
        "page_qa": "💬 智能问答",
        "page_tables": "📊 表格解析查看",
        "qa": "💬 智能问答",
        "question": "请输入您的问题（支持中英文）：",
        "ask": "🚀 提问",
        "refs": "📚 引用来源",
        "text_refs": "文本引用",
        "table_refs": "表格引用",
        "latency": "总耗时",
        "retrieve": "检索",
        "generate": "生成",
        "route": "路由",
        "quick_questions": "⚡ 快捷问题（14 题）",
        "no_answer": "请输入问题后点击提问",
        "route_text": "纯文本",
        "route_table": "纯表格",
        "route_hybrid": "混合",
        "feedback": "本条回答是否有帮助？",
        "good": "👍 点赞",
        "bad": "👎 点踩",
        "comment": "✍️ 补充评论",
        "submit": "提交反馈",
        "fb_ok": "感谢您的反馈！",
        "table_viewer": "选择 PDF 查看解析出的表格",
        "table_count": "共解析出 {n} 张表格",
        "table_page": "表格页码",
        "prev": "⬅ 上一张",
        "next": "下一张 ➡",
        "headers": "表头",
        "rows": "数据行",
        "caption": "表标题",
        "no_tables": "该文档未解析出表格",
    },
    "en": {
        "title": "📊 PDF Table Parsing & Retrieval Optimization",
        "sub": "Ticket 3: Table Parsing · Table-aware Retrieval · Multi-doc · Bilingual · Rerank",
        "lang_label": "界面语言 / Language",
        "answer_lang": "Answer Language",
        "doc_select": "Select Document",
        "page_select": "Navigation",
        "page_qa": "💬 Ask",
        "page_tables": "📊 Table Viewer",
        "qa": "💬 Ask",
        "question": "Enter your question (Chinese or English):",
        "ask": "🚀 Ask",
        "refs": "📚 References",
        "text_refs": "Text References",
        "table_refs": "Table References",
        "latency": "Total",
        "retrieve": "Retrieval",
        "generate": "Generation",
        "route": "Route",
        "quick_questions": "⚡ Quick Questions (14)",
        "no_answer": "Enter a question and click Ask",
        "route_text": "Text Only",
        "route_table": "Table Only",
        "route_hybrid": "Hybrid",
        "feedback": "Was this answer helpful?",
        "good": "👍 Upvote",
        "bad": "👎 Downvote",
        "comment": "✍️ Comment",
        "submit": "Submit",
        "fb_ok": "Thanks for your feedback!",
        "table_viewer": "Select a PDF to view parsed tables",
        "table_count": "{n} tables parsed",
        "table_page": "Table index",
        "prev": "⬅ Prev",
        "next": "Next ➡",
        "headers": "Headers",
        "rows": "Rows",
        "caption": "Caption",
        "no_tables": "No tables parsed for this document",
    },
}
st.session_state.setdefault("ui_lang", "zh")
lang_ui = st.session_state["ui_lang"]
T = I18N[lang_ui]

# ================= 14 个测试问题 =================
QUICK_QUESTIONS = [
    ("T01 发行股数", "本次发行的发行股数是多少？", "招股说明书1"),
    ("T02 募集资金", "本次发行的募集资金总额是多少？", "招股说明书1"),
    ("T03 关联方", "公司的主要关联方有哪些？", "招股说明书1"),
    ("T04 持股比例", "公司前五大股东的持股比例是多少？", "招股说明书1"),
    ("T05 军用收入", "报告期内，公司来自军用领域的收入分别是多少？", "招股说明书1"),
    ("T06 营业收入", "报告期内公司营业收入分别是多少？", "招股说明书1"),
    ("T07 前五大客户", "前五大客户占营业收入的比例是多少？", "招股说明书1"),
    ("T08 力源发行股数", "武汉力源信息本次发行的发行股数是多少？", "招股说明书2"),
    ("T09 力源募投", "力源信息的募集资金投向哪些项目？", "招股说明书2"),
    ("T10 力源收入构成", "力源信息报告期内主营业务收入构成是什么？", "招股说明书2"),
    ("T11 EN Shares", "How many shares are issued in this offering?", "招股说明书1"),
    ("T12 EN Funds", "What is the total amount of funds raised?", "招股说明书1"),
    ("T13 EN Related", "Who are the main related parties of the company?", "招股说明书1"),
    ("T14 EN Shareholder", "What is the shareholding ratio of top five shareholders?", "招股说明书1"),
]

# ================= 侧栏 =================
with st.sidebar:
    st.header("⚙️ 设置 / Settings")
    lang_choice = st.radio(T["lang_label"], ["中文", "English"], horizontal=True)
    st.session_state["ui_lang"] = {"中文": "zh", "English": "en"}[lang_choice]
    if st.session_state["ui_lang"] != lang_ui:
        st.rerun()

    # 工单三：页面导航
    page = st.radio(T["page_select"],
                    [T["page_qa"], T["page_tables"]],
                    horizontal=True)
    st.divider()

    ans_lang = st.radio(
        T["answer_lang"],
        ["自动检测 / Auto", "中文", "English"],
        horizontal=True,
    )
    st.session_state["answer_lang"] = {
        "自动检测 / Auto": None, "中文": "zh", "English": "en",
    }[ans_lang]

    doc_choice = st.selectbox(
        T["doc_select"],
        ["全部 / All", "招股说明书1", "招股说明书2"],
    )
    doc_id = None if "All" in doc_choice or "全部" in doc_choice else doc_choice

    top_k = st.slider("Top-K", 1, 15, 5)
    st.divider()

    # 快捷问题（仅问答页显示）
    if page == T["page_qa"]:
        st.subheader(T["quick_questions"])
        for label, q, doc in QUICK_QUESTIONS:
            if st.button(label, key=f"qq_{label}", use_container_width=True):
                st.session_state["q_box"] = q
                st.session_state["qq_doc"] = doc
                st.rerun()

# 自动设置快捷问题的 doc_id
if "qq_doc" in st.session_state and page == T["page_qa"]:
    doc_id = st.session_state["qq_doc"]

# ================= RAG 引擎（懒加载） =================
@st.cache_resource
def get_engine():
    """工单三：懒加载 RAGEngineV3"""
    from src.rag_engine_v3 import RAGEngineV3
    return RAGEngineV3(top_k=5, use_rerank=True)


# ================= 辅助：表格渲染为 Markdown =================
def render_table_as_markdown(table_chunk):
    """工单三：将检索到的表格结构渲染为 Markdown 表格"""
    # 表格 chunk 可能含 rows / headers / table_text
    headers = table_chunk.get("headers") or table_chunk.get("metadata", {}).get("headers")
    rows = table_chunk.get("rows") or table_chunk.get("metadata", {}).get("rows")

    # 如果没有结构化 rows，尝试从 table_text 解析
    if not rows and table_chunk.get("content"):
        return None  # 无法渲染为表格，用文本展示

    if not headers:
        if rows and len(rows) > 0:
            headers = [f"col_{i+1}" for i in range(len(rows[0]))]
        else:
            return None

    # 构建 Markdown 表格
    md = "| " + " | ".join(str(h) for h in headers) + " |\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in (rows or [])[:20]:  # 最多 20 行
        # 补齐列数
        row = list(row) + [""] * (len(headers) - len(row))
        md += "| " + " | ".join(str(c)[:50] for c in row) + " |\n"
    return md


# ================= 页面 1：智能问答 =================
if page == T["page_qa"]:
    st.title(T["title"])
    st.caption(T["sub"])
    st.subheader(T["qa"])
    question = st.text_area(T["question"], height=90, key="q_box")

    col1, _ = st.columns([1, 4])
    with col1:
        ask_clicked = st.button(T["ask"], type="primary",
                                use_container_width=True)

    # 提问处理
    if ask_clicked and question.strip():
        st.session_state["fb_rating"] = 0
        with st.spinner("检索中... / Retrieving..."):
            try:
                from src.multilingual_v3 import bilingual_ask_rag
                engine = get_engine()
                lang_override = st.session_state.get("answer_lang")

                t0 = time.time()
                result = bilingual_ask_rag(
                    engine, question.strip(),
                    doc_id=doc_id, top_k=top_k,
                    lang_override=lang_override,
                )
                result["total_ms"] = (time.time() - t0) * 1000
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"错误 / Error: {e}")
                st.session_state["last_result"] = None

    # 结果展示
    result = st.session_state.get("last_result")
    if result:
        # 答案
        st.subheader("📝 " + ("回答" if lang_ui == "zh" else "Answer"))
        st.markdown(result.get("answer", ""))

        # 指标
        bd = result.get("breakdown", {})
        route = result.get("route", {})
        route_label = {
            "text_only": T["route_text"],
            "table_only": T["route_table"],
            "hybrid": T["route_hybrid"],
        }.get(route.get("route", ""), route.get("route", ""))

        m = st.columns(5)
        m[0].metric(T["latency"], f"{result.get('total_ms', 0):.0f} ms")
        m[1].metric(T["retrieve"],
                    f"{bd.get('retrieve_ms', 0):.0f} ms")
        m[2].metric(T["generate"],
                    f"{bd.get('llm_ms', 0):.0f} ms")
        m[3].metric(T["route"], route_label)
        m[4].metric("Lang", result.get("lang", "?").upper())

        # 翻译信息
        if result.get("translated"):
            st.info(
                f"🔍 {('英文问题已翻译为中文检索' if lang_ui=='zh' else 'English question translated to Chinese for retrieval')}: "
                f"`{result.get('search_query', '')}`"
            )

        # 引用
        refs = result.get("references", [])
        text_refs = [r for r in refs if r.get("type") == "text"]
        table_refs = [r for r in refs if r.get("type") == "table"]

        # 表格引用：以 Markdown 表格形式展示
        if table_refs:
            with st.expander(f"📊 {T['table_refs']}（{len(table_refs)}）",
                              expanded=True):
                for ref in table_refs:
                    st.markdown(
                        f"**[{ref['ref_id']}]** "
                        f"{('第' if lang_ui=='zh' else 'page ')}{ref.get('page', '?')}"
                        f"{'页' if lang_ui=='zh' else ''} · "
                        f"`{ref.get('table_id', '')}` · "
                        f"score={ref.get('score', 0):.4f}"
                    )
                    # 工单三：尝试从 retrieved_tables 获取完整表格结构
                    retrieved_tables = result.get("retrieved_tables", [])
                    table_data = None
                    for t in retrieved_tables:
                        if t.get("table_id") == ref.get("table_id"):
                            table_data = t
                            break

                    md_table = render_table_as_markdown(table_data) if table_data else None
                    if md_table:
                        st.markdown(md_table)
                    else:
                        st.text(ref.get("preview", "")[:300])
                    st.divider()

        # 文本引用
        if text_refs:
            with st.expander(f"📄 {T['text_refs']}（{len(text_refs)}）"):
                for ref in text_refs:
                    st.markdown(
                        f"**[{ref['ref_id']}]** "
                        f"{('第' if lang_ui=='zh' else 'page ')}{ref.get('page', '?')}"
                        f"{'页' if lang_ui=='zh' else ''} · "
                        f"score={ref.get('score', 0):.4f}"
                    )
                    st.text(ref.get("preview", "")[:300])
                    st.divider()

        # ================= 反馈 =================
        st.session_state.setdefault("fb_rating", 0)
        st.caption(T["feedback"])
        fb_cols = st.columns([1, 1, 4])
        with fb_cols[0]:
            if st.button(T["good"], key="fb_good",
                         use_container_width=True,
                         type="primary" if st.session_state["fb_rating"] == 1 else "secondary"):
                st.session_state["fb_rating"] = 1
        with fb_cols[1]:
            if st.button(T["bad"], key="fb_bad",
                         use_container_width=True,
                         type="primary" if st.session_state["fb_rating"] == -1 else "secondary"):
                st.session_state["fb_rating"] = -1
        with fb_cols[2]:
            fb_comment = st.text_input(T["comment"], key="fb_comment_input",
                                       label_visibility="collapsed")
            if st.button(T["submit"], key="fb_submit"):
                # 工单三：保存反馈到本地 JSON
                feedback_dir = Path(PROJECT_ROOT) / "data" / "feedback"
                feedback_dir.mkdir(parents=True, exist_ok=True)
                fb_file = feedback_dir / f"feedback_{int(time.time())}.json"
                fb_data = {
                    "question": result.get("original_query", question),
                    "answer": result.get("answer", "")[:500],
                    "rating": st.session_state["fb_rating"],
                    "comment": fb_comment,
                    "route": result.get("route", {}).get("route"),
                    "latency_ms": result.get("total_ms"),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                fb_file.write_text(
                    json.dumps(fb_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success(T["fb_ok"])

    elif not ask_clicked:
        st.caption(T["no_answer"])


# ================= 页面 2：表格解析查看 =================
elif page == T["page_tables"]:
    st.title(T["title"])
    st.caption(T["sub"])
    st.subheader(T["page_tables"])

    # 选择 PDF
    table_doc = st.selectbox(
        T["table_viewer"],
        ["招股说明书1", "招股说明书2"],
    )

    # 加载表格 JSON
    table_file = Path(PROJECT_ROOT) / "data" / "tables" / f"{table_doc}_tables.json"
    if not table_file.exists():
        st.warning(T["no_tables"])
    else:
        data = json.loads(table_file.read_text(encoding="utf-8"))
        tables = data.get("tables", [])
        st.caption(T["table_count"].format(n=len(tables)))

        if tables:
            # 表格导航
            nav_cols = st.columns([1, 2, 1])
            st.session_state.setdefault("table_idx", 0)

            with nav_cols[0]:
                if st.button(T["prev"], use_container_width=True):
                    st.session_state["table_idx"] = max(0, st.session_state["table_idx"] - 1)

            with nav_cols[1]:
                idx = st.number_input(
                    T["table_page"], min_value=1, max_value=len(tables),
                    value=st.session_state["table_idx"] + 1, step=1,
                )
                st.session_state["table_idx"] = int(idx) - 1

            with nav_cols[2]:
                if st.button(T["next"], use_container_width=True):
                    st.session_state["table_idx"] = min(
                        len(tables) - 1, st.session_state["table_idx"] + 1
                    )

            idx = st.session_state["table_idx"]
            tbl = tables[idx]

            # 表格元信息
            st.markdown(f"**{T['table_page']}**: {idx + 1} / {len(tables)}")
            st.markdown(
                f"**table_id**: `{tbl.get('table_id', '')}` · "
                f"**page**: {tbl.get('page_range', tbl.get('page', '?'))} · "
                f"**rows**: {len(tbl.get('rows', []))} · "
                f"**cols**: {tbl.get('cols', '?')}"
            )
            if tbl.get("caption"):
                st.markdown(f"**{T['caption']}**: {tbl['caption']}")
            if tbl.get("header_inferred"):
                st.caption("⚠️ " + ("表头为自动推断" if lang_ui == "zh"
                                      else "Headers were inferred"))

            st.divider()

            # 渲染表格
            headers = tbl.get("headers") or []
            rows = tbl.get("rows") or []
            if not headers and rows:
                headers = [f"col_{i+1}" for i in range(len(rows[0]))] if rows else []

            if headers and rows:
                md = "| " + " | ".join(str(h) for h in headers) + " |\n"
                md += "|" + "|".join(["---"] * len(headers)) + "|\n"
                for row in rows[:30]:  # 最多 30 行
                    row = list(row) + [""] * (len(headers) - len(row))
                    md += "| " + " | ".join(str(c)[:80] for c in row) + " |\n"
                st.markdown(md)
                if len(rows) > 30:
                    st.caption(f"... +{len(rows) - 30} rows")
            else:
                st.text(tbl.get("table_text", "")[:500])

            # 合并单元格信息
            merged = tbl.get("merged_cells", [])
            if merged:
                with st.expander(f"🔗 合并单元格（{len(merged)}）"):
                    for mc in merged[:10]:
                        st.text(str(mc))

            # table_text（自然语言描述）
            table_texts = data.get("table_texts", [])
            for tt in table_texts:
                if tt.get("table_id") == tbl.get("table_id"):
                    with st.expander("📝 table_text（自然语言描述）"):
                        st.text(tt.get("table_text", "")[:500])
                    break
