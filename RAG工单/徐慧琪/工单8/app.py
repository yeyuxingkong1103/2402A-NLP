# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
工单08 金融问答系统界面（Streamlit）

页签：
  1. 问答        —— 文字 + 语音输入，输出答案/检索结果/图谱结构/响应时间
  2. 研报生成    —— 主题 → 结构化研报（图谱骨架 + 文本证据）
  3. 图谱可视化  —— 交互式浏览（vis.js），离线回退静态图
  4. 实体关系    —— 查看/新增/删除实体与关系，创建思路讲解
  5. 评测结果    —— eval_question 逐题结果与与07对比
  6. 系统状态    —— 模型清单、健康检查、图谱统计

启动：streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from src import (config, dialogue, graph_qa, graph_viz, speech, ui_helpers,
                 upload_index)
from src.kg_schema import ENTITY_TYPES, RELATION_TYPES, Entity, Relation
from src.kg_store import KnowledgeGraph

st.set_page_config(page_title="金融问答 · Graph RAG（工单08）",
                   page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="正在加载知识图谱与模型…")
def _boot():
    return graph_qa.warmup()


def _get_kg():
    return graph_qa.get_kg()


def _tab_qa() -> None:
    st.subheader("智能问答")

    col_lang, col_strategy = st.columns([1, 2])
    with col_lang:
        lang = st.radio("语言 / Language", ["中文", "English"],
                        horizontal=True, key="qa_lang")
    with col_strategy:
        strategy = st.selectbox("检索策略", ["hybrid", "vector", "fulltext"],
                                key="qa_strategy")

    qlang = "zh" if lang == "中文" else "en"

    st.markdown("#### 语音输入")
    components.html(speech.speech_component_html(
        lang="zh-CN" if qlang == "zh" else "en-US"), height=90)
    st.caption(speech.browser_support_note())

    st.markdown("#### 📄 上传文档（可选）")
    up = st.file_uploader("上传 PDF，就这份文档提问", type=["pdf"], key="qa_upload",
                          help="上传的文档只用于本次会话的提问，不写入知识图谱与语料库；"
                               "答案同样标注页码。")
    upload_ready = False
    if up is not None:
        sig = (up.name, int(getattr(up, "size", 0)))
        if st.session_state.get("upload_sig") != sig:
            with st.spinner(f"正在解析《{up.name}》…"):
                try:
                    idx = upload_index.build_index_from_pdf(up.getvalue(), up.name)
                    st.session_state["upload_index"] = idx
                    st.session_state["upload_sig"] = sig
                    st.success(f"文档已就绪：《{up.name}》共 {idx.n_chunks} 个文本块"
                               + ("（超长，已截断到上限）" if idx.truncated else ""))
                except upload_index.UploadError as exc:
                    st.session_state.pop("upload_index", None)
                    st.session_state["upload_sig"] = None
                    st.error(f"上传失败：{exc}")
        idx = st.session_state.get("upload_index")
        if idx is not None:
            upload_ready = st.checkbox(
                f"只就上传文档《{idx.filename}》提问（不与知识图谱混答）",
                value=True, key="qa_use_upload")
    else:
        st.session_state.pop("upload_index", None)
        st.session_state["upload_sig"] = None

    question = st.text_input("请输入问题 / Ask a question",
                             key="qa_input",
                             placeholder="例如：平安银行2019年末的不良贷款率是多少？")

    c1, c2 = st.columns([1, 1])
    ask_clicked = c1.button("提问", type="primary", key="qa_ask")
    if c2.button("清空对话历史", key="qa_clear"):
        dialogue.reset_state("ui")
        st.success("已清空")

    if ask_clicked and question.strip():
        with st.spinner("检索中…"):
            if upload_ready and st.session_state.get("upload_index") is not None:
                res = graph_qa.ask_uploaded(question.strip(),
                                            st.session_state["upload_index"])
            else:
                res = graph_qa.ask(question.strip(), kg=_get_kg(),
                                   state=dialogue.get_state("ui"),
                                   strategy=strategy)
        d = res.to_dict()
        st.session_state["last_result"] = d
        st.session_state["last_question"] = question.strip()

    d = st.session_state.get("last_result")
    if not d:
        st.info("请输入问题后点击「提问」。")
        return

    st.markdown("### 答案")
    st.markdown(ui_helpers.render_answer_md(d, d.get("lang", "zh")))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("检索模式", d.get("mode", "—"))
    m2.metric("图谱事实", len(d.get("graph_facts") or []))
    m3.metric("文本块", len(d.get("text_hits") or []))
    m4.metric("总耗时(s)", f"{(d.get('timings') or {}).get('total', 0):.2f}")

    tab1, tab2, tab3 = st.tabs(["🔎 检索结果（文本块）", "🕸 知识图谱结构", "🛠 调试"])
    with tab1:
        st.markdown(ui_helpers.render_hits_md(d.get("text_hits") or []))
    with tab2:
        st.markdown("**图谱事实**")
        st.markdown(ui_helpers.render_facts_md(d.get("graph_facts") or []))
        sub = d.get("subgraph") or {}
        nodes, edges = sub.get("nodes") or [], sub.get("edges") or []
        if nodes:
            components.html(
                graph_viz.to_vis_html(nodes, edges, height=460,
                                      title=f"子图（模式：{sub.get('mode')}）"),
                height=520)
        else:
            st.info("本次未检索到图谱子图。")
    with tab3:
        st.json(d.get("timings") or {})
        if d.get("degraded"):
            st.warning(d.get("degraded_reason", ""))
        st.download_button("导出答案与来源",
                           ui_helpers.answer_with_sources(d),
                           file_name="answer.md", key="qa_dl")


def _tab_report() -> None:
    st.subheader("金融研报生成")
    topic = st.text_input("研报主题", value="银行板块2019年资产质量对比",
                          key="rp_topic")
    kg = _get_kg()
    all_companies = ([kg.g.nodes[n]["name"] for n in kg.nodes_of_type("Company")]
                     if kg else [])
    picked = st.multiselect("涉及公司（留空=全部）", sorted(all_companies),
                            key="rp_companies")
    st.caption("研报模式：图谱提供确定性数据骨架，文本块提供原文证据，"
               "每个数字均可回溯到文档与页码。")

    if st.button("生成研报", type="primary", key="rp_go"):
        with st.spinner("正在生成…"):
            rep = graph_qa.generate_report(topic, kg=kg,
                                           companies=picked or None)
        st.session_state["last_report"] = rep.to_dict()

    rep = st.session_state.get("last_report")
    if not rep:
        return

    if rep.get("degraded"):
        st.warning(f"降级：{rep.get('degraded_reason')}")

    st.markdown(rep.get("markdown", ""))
    if rep.get("source_table"):
        st.markdown("### 数据来源表（每个数字可回溯）")
        st.dataframe(rep["source_table"], width="stretch")
    st.download_button("导出研报 Markdown", rep.get("markdown", ""),
                       file_name="report.md", key="rp_dl")


def _tab_graph() -> None:
    st.subheader("知识图谱可视化")
    kg = _get_kg()
    if kg is None:
        st.error(f"图谱不可用：{graph_qa.status().get('graph_error')}")
        st.code("python build_kg.py --layer1", language="bash")
        return

    st.caption(f"图谱规模：{kg.g.number_of_nodes()} 个实体 / "
               f"{kg.g.number_of_edges()} 条关系。"
               "多类型实体用不同颜色区分，关系用带标签连线表示。")

    companies = sorted(kg.g.nodes[n]["name"] for n in kg.nodes_of_type("Company"))
    c1, c2, c3 = st.columns(3)
    picked = c1.multiselect("公司筛选", companies,
                            default=companies[:3], key="kg_companies")
    types = c2.multiselect("实体类型筛选", list(ENTITY_TYPES.keys()),
                           default=[], key="kg_types")
    hops = c3.slider("邻居跳数", 1, 3, 1, key="kg_hops")

    nodes, edges = graph_viz.filter_subgraph(
        kg, companies=picked or None, types=types or None, hops=hops,
        max_nodes=config.KG_SUBGRAPH_MAX_NODES * 2)
    st.write(f"子图：{len(nodes)} 节点 / {len(edges)} 边")

    view = st.radio("渲染方式", ["交互式（vis.js）", "静态（matplotlib）"],
                    horizontal=True, key="kg_view")
    if view.startswith("交互"):
        components.html(graph_viz.to_vis_html(nodes, edges, height=600,
                                              title="知识图谱"),
                        height=700)
    else:
        png = os.path.join(config.KG_DIR, "view.png")
        graph_viz.to_matplotlib_png(nodes, edges, png, dpi=120)
        st.image(png)

    if nodes:
        st.markdown("#### 节点详情")
        labels = {n["id"]: n["label"] for n in nodes}
        pick = st.selectbox("选择节点查看属性与来源",
                            list(labels), key="kg_node")
        st.json(next(n for n in nodes if n["id"] == pick))


def _tab_entities() -> None:
    st.subheader("实体与关系管理")
    kg = _get_kg()
    if kg is None:
        st.error("图谱不可用")
        return

    t1, t2, t3 = st.tabs(["实体浏览", "新增实体/关系", "创建思路讲解"])

    with t1:
        etype = st.selectbox("实体类型", list(ENTITY_TYPES.keys()),
                             key="ent_type")
        ids = kg.nodes_of_type(etype)
        rows = [{"id": n, "name": kg.g.nodes[n]["name"],
                 "度数": kg.g.degree(n),
                 "别名": "、".join(kg.g.nodes[n].get("aliases", [])[:3]),
                 "属性": json.dumps(kg.g.nodes[n].get("attrs", {}),
                                    ensure_ascii=False)[:120]}
                for n in ids[:500]]
        st.write(f"共 {len(ids)} 个 `{etype}` 实体"
                 + ("（仅显示前 500）" if len(ids) > 500 else ""))
        st.dataframe(rows, width="stretch")

        if st.button("删除该类型下所有节点（演示用）", key="ent_del_all"):
            st.warning("演示环境：删除操作仅作用于内存副本，"
                       "重新加载页面即恢复（不会改动磁盘图谱）。")

    with t2:
        st.markdown("**新增实体**")
        with st.form("add_entity", clear_on_submit=True):
            name = st.text_input("名称（规范名）", key="ae_name")
            etype2 = st.selectbox("类型", list(ENTITY_TYPES.keys()), key="ae_type")
            aliases = st.text_input("别名（逗号分隔）", key="ae_alias")
            if st.form_submit_button("添加到内存图谱"):
                e = Entity(f"{etype2}:{name}", name, etype2,
                           aliases=[a.strip() for a in aliases.split(",") if a.strip()])
                try:
                    kg.add_entity(e)
                    st.success(f"已添加 {e.entity_id}（内存；"
                               f"持久化请写入 data/kg 并重启）")
                except ValueError as exc:
                    st.error(str(exc))

        st.markdown("**新增关系**")
        with st.form("add_relation", clear_on_submit=True):
            head = st.text_input("头实体 ID", key="ar_head")
            rtype = st.selectbox("关系类型", list(RELATION_TYPES.keys()),
                                 key="ar_type")
            tail = st.text_input("尾实体 ID", key="ar_tail")
            evidence = st.text_input("原文依据（LLM 抽取必须提供）", key="ar_ev")
            if st.form_submit_button("添加关系"):
                ok = kg.add_relation(Relation(head, rtype, tail, evidence=evidence,
                                              confidence=1.0 if evidence else 0.99))
                st.success("已添加" if ok else "被拒绝（端点不存在/类型非法/重复）")

    with t3:
        st.markdown("""
### 实体、关系是怎么从 PDF 里建出来的

**第一步：定本体（先看语料，而不是先定模型）**

9 份年报有固定的章节结构 ——「公司治理」「董事监事高级管理人员」「股东情况」
「组织结构图」「主要会计数据」「业务概要」「风险管理」。这些章节反复出现、格式稳定，
因此可稳定抽取的实体与关系是有限且可枚举的：

| 章节 | 稳定可抽的实体 | 稳定可抽的关系 |
| --- | --- | --- |
| 公司治理 / 股东 | 公司、人物 | `SHAREHOLDER_OF`、`CONTROLS`、`SERVES_AS` |
| 组织结构图 | 部门、板块 | `BELONGS_TO` |
| 主要会计数据 | 指标 | `HAS_METRIC`、`REPORTED_IN` |
| 业务概要 | 业务/产品 | `OPERATES` |
| 风险管理 | 指标、事件 | `HAS_METRIC`、`INVESTS_IN` |

**第二步：分层抽取 —— 词典打底，LLM 补长尾**

- **第 1 层（词典 + 正则，全量 7757 块）**：公司名/指标名高度可枚举，
  用词典做最长匹配零成本、零幻觉、可复现；金额/百分比/报告期用正则精确切分。
  这一层给出图谱的**可信骨架**；子类口径（企业贷款不良率、零售不良率、
  行标大型企业贷款不良率…）由口径守卫拒绝，不冒充公司总口径。
- **第 2 层（LLM 定向抽取，约 2000~3000 块）**：先用三条信号
  （章节白名单 / 图谱触发词 / 实体密度）筛出最可能含关系的块，
  再让 qwen2.5:3b 抽取控股链、任职、投资等长尾关系。

**第三步：三道防幻觉闸门**

1. 实体类型必须在本体内（10 类），否则丢弃；
2. 关系类型必须在 10 类内，且头尾类型符合本体约束（如 `HAS_METRIC` 必须是
   公司→指标），否则丢弃；
3. **关系的 `evidence` 必须真实出现在原文中**（逐字比对），否则丢弃。

**第四步：多文档融合**

实体 ID 为 `{类型}:{规范名}`，因此「平安银行股份有限公司」「平安银行」
「平安银行股份」经别名表归一会落到**同一个节点**；同一实体在 9 份年报中出现多次时，
边累加来源而不重复建点。同一指标同期间若出现不同值，**保留全部并标记冲突**，
不静默覆盖 —— 金融口径差异是真实存在的。

**第五步：每条边都带 provenance**

`source_doc`（文档）+ `page`（页码）+ `evidence`（原文片段）三元组随边存储，
这就是工单「返回答案所在的文本块」的实现基础：图谱给出的每个结论都能落到具体页。
""")


def _tab_eval() -> None:
    st.subheader("评测结果（eval_question）")
    gt_path = os.path.join("data", "eval", "ground_truth.json")
    if os.path.isfile(gt_path):
        gt = json.load(io.open(gt_path, encoding="utf-8"))
        st.write(f"问题集共 {len(gt)} 题（工单07 的 10 题 + 工单08 扩展 "
                 f"{len(gt) - 10} 题）")
        st.dataframe([{"编号": f"Q{it['id']:02d}", "类型": it["qtype"],
                       "问题": it["question"][:80],
                       "来源": "07" if it.get("origin") == "wo07" else "08扩展"}
                      for it in gt], width="stretch")

    res_path = os.path.join("data", "eval", "graph_qa_results.json")
    if os.path.isfile(res_path):
        st.markdown("### 逐题结果")
        res = json.load(io.open(res_path, encoding="utf-8"))
        st.dataframe([{"编号": f"Q{r['id']:02d}", "模式": r.get("mode"),
                       "耗时(s)": round((r.get("timings") or {})
                                       .get("total", 0), 2),
                       "降级": r.get("degraded"),
                       "答案": (r.get("answer") or "")[:100]} for r in res],
                     width="stretch")
    else:
        st.info("尚无逐题结果。运行：`python graph_qa_cli.py --all`")

    cmp_path = os.path.join("data", "eval", "compare_wo07.json")
    if os.path.isfile(cmp_path):
        st.markdown("### 与工单07 的对比")
        st.json(json.load(io.open(cmp_path, encoding="utf-8")))
    else:
        st.info("尚无对比结果。运行：`python src/compare_with_wo07.py`")


def _tab_status() -> None:
    st.subheader("系统状态")
    st.markdown("#### 知识图谱")
    st.json(graph_qa.status())

    st.markdown("#### 模型")
    lm_path = os.path.join("data", "local_models.json")
    if os.path.isfile(lm_path):
        lm = json.load(io.open(lm_path, encoding="utf-8"))
        st.write(f"扫描到模型目录 {len(lm['models'])} 个，"
                 f"Ollama 模型 {len(lm['ollama']['models'])} 个（**零下载**）")
        st.dataframe([{"角色": m["role"], "名称": m["name"],
                       "路径": m["path"],
                       "GB": round(m["size_bytes"] / 1024 ** 3, 2)}
                      for m in lm["models"]], width="stretch")
    else:
        st.info("运行 `python scan_local_models.py` 生成模型清单")

    st.markdown("#### 检索与精排")
    try:
        from src import retriever, strategies
        st.json({"strategies": strategies.describe(),
                 "retriever": retriever.stats()})
    except Exception as exc:                       # noqa: BLE001
        st.warning(f"检索模块状态不可用：{exc}")

    st.markdown("#### 日志")
    log_dir = config.LOG_DIR
    if os.path.isdir(log_dir):
        files = sorted(os.listdir(log_dir))[-5:]
        st.write(f"日志目录：`{log_dir}`")
        for f in files:
            st.caption(f)
    else:
        st.info("暂无日志")


def main() -> None:
    st.title("📊 金融问答系统 · Graph RAG")
    st.caption("工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答")

    boot = _boot()
    if not boot.get("graph", {}).get("ok"):
        st.warning(f"图谱未就绪：{boot.get('graph', {}).get('error')} — "
                   f"请先运行 `python build_kg.py --layer1`")

    tabs = st.tabs(["💬 问答", "📝 研报生成", "🕸 图谱可视化",
                    "🔗 实体关系", "📈 评测结果", "⚙️ 系统状态"])
    with tabs[0]:
        _tab_qa()
    with tabs[1]:
        _tab_report()
    with tabs[2]:
        _tab_graph()
    with tabs[3]:
        _tab_entities()
    with tabs[4]:
        _tab_eval()
    with tabs[5]:
        _tab_status()


if __name__ == "__main__":
    main()
