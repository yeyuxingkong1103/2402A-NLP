# -*- coding: utf-8 -*-
"""工单7 **Gradio 版**测试评估台（原计划设计的技术栈）。

工单编号：人工智能NLP-RAG-功能测试及评估

与「部署/界面/测试评估台.py」的关系
---------------------------------
* ``测试评估台.py`` 是**纯标准库 http.server** 版界面，作为**零依赖兜底**保留
  （本机最初误判「沙箱封网、不能装包」，故先做了它；该判断后来被实测推翻，见下）。
* 本文件是**任务书原计划设计**的 Gradio 版，二者**共用同一套后端**：
  - 引擎：直接复用 ``测试评估台.EngineBox``（持有唯一 ``QAEngine``、后台预热、串行问答）
  - 数据：直接复用 ``测试评估台.build_eval_payload / build_flow_payload / build_health_payload``
  因此不存在「两套实现口径不一致」的问题 —— 换界面不换逻辑。

⚠️ 关于依赖（实测更正，2026-10-08）
--------------------------------
本仓库一度把「沙箱封禁外网出口 / pip 会挂死」当作既成事实，于是把 Gradio 列为**技术栈偏离**。
经 captain 复测，该结论错误：
  * Python（OpenSSL 3.5.6 + certifi）对 pypi / 清华源 / 阿里源 / hf-mirror / github **全部 HTTP 200**；
  * 失败的只是 ``curl.exe`` 与 PowerShell ``Invoke-WebRequest``（schannel
    ``SEC_E_NO_CREDENTIALS``，Windows 系统 TLS 凭证不可用），与「封网」无关；
  * 但 ``pip install --target`` 在本机沙箱下确实会 ``PermissionError``（pip 自己的临时解包目录被拒），
    故改用 ``工具/不依赖pip装包.py``（下 wheel → zipfile 解包）把**原计划设计的依赖**装到
    ``研发/第三方库``：**gradio 6.29.1 / pandas 3.0.6 / pdfplumber 0.11.10**（含依赖约束求解与 lockfile）。

用法：
    python 部署/界面/gradio_app.py                     # 默认 http://127.0.0.1:7860
    python 部署/界面/gradio_app.py --port 7862
    python 部署/界面/gradio_app.py --no-warmup         # 不预热（首问较慢，仅演示用）

退出码：0 正常退出；3 依赖缺失（gradio 未安装）；4 索引缺失。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
REPO_ROOT = Path(__file__).resolve().parents[2]          # ...\工单7
DASHBOARD = REPO_ROOT / "部署" / "界面" / "测试评估台.py"
DEFAULT_TOP_K = 10


def _load_dashboard() -> Any:
    """把标准库版界面当**后端模块**加载（复用其 EngineBox 与 payload 构造函数）。

    ⚠️ 路径顺序（实测踩坑）：``研发`` 放最前，``研发/第三方库`` 放**最后**。
    第三方库里有 numpy 2.4.6，而 venv 的 faiss 是针对旧 numpy 编译的 C 扩展；
    第三方库若在 sys.path 前面，numpy 2.x 会覆盖旧版 →
    ``import faiss`` 抛 ``AttributeError: _ARRAY_API not found`` → 向量检索整体失败。
    """
    for p in (str(REPO_ROOT / "研发"), str(REPO_ROOT / "研发" / "第三方库")):
        if p not in sys.path:
            sys.path.append(p)
    if not DASHBOARD.is_file():
        raise FileNotFoundError(f"找不到后端模块：{DASHBOARD}")
    spec = importlib.util.spec_from_file_location("gd7_dashboard", DASHBOARD)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {DASHBOARD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gd7_dashboard"] = module
    spec.loader.exec_module(module)
    return module


def _require_gradio() -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:                    # noqa: PERF203
        print("❌ 未安装 gradio（本工单把它装在 研发/第三方库）。", file=sys.stderr)
        print("   请先运行： python 工具/不依赖pip装包.py gradio --with-deps", file=sys.stderr)
        raise SystemExit(3) from exc
    return gr


# ---------------------------------------------------------------------------
# 展示层
# ---------------------------------------------------------------------------
def _fmt_hits(payload: dict[str, Any]) -> str:
    """把 Top-K 命中渲染成 Markdown 表（含页码/类型/分数）。"""
    hits = list(payload.get("evidence") or payload.get("chunks") or [])
    if not hits:
        return "_（本次没有检索命中）_"
    lines = ["| # | chunk_id | 文件 | 页 | 类型 | 最终分 | 片段（前 160 字） |",
             "|---|---|---|---|---|---|---|"]
    for h in hits[:10]:
        text = str(h.get("text_head") or "").replace("\n", " ").replace("|", "\\|")[:160]
        fname = str(h.get("file_name") or "")[:34]
        lines.append(f"| {h.get('rank')} | `{h.get('chunk_id')}` | {fname} | {h.get('page')} | "
                     f"{h.get('type')} | {h.get('final_score')} | {text} |")
    return "\n".join(lines)


def _fmt_latency(payload: dict[str, Any]) -> str:
    lat = dict(payload.get("latency_ms") or {})
    total = lat.get("total")
    flag = "✅ ≤3000ms" if isinstance(total, (int, float)) and total <= 3000 else "⚠️ 超 3000ms"
    return (f"**检索** {lat.get('retrieve')} ms ｜ **生成** {lat.get('generate')} ms ｜ "
            f"**首字** {lat.get('first_token')} ms ｜ **总计** {lat.get('total')} ms ｜ {flag}　"
            f"（含调度墙钟 {payload.get('wall_ms')} ms）\n\n"
            f"**生成方式** `{payload.get('generation_method')}` ｜ **后端** "
            f"{payload.get('backend')}:{payload.get('model')} ｜ **拒答** "
            f"{payload.get('refused')}（{payload.get('refusal_reason') or '—'}）")


def _answers_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    """评估看板表格（17 题）。"""
    out: list[list[Any]] = []
    for r in rows:
        out.append([
            r.get("id"), r.get("question_type"), r.get("company"), r.get("lang"),
            (str(r.get("answer") or "")[:90]),
            r.get("coverage"), "是" if r.get("judged") else "否",
            r.get("recall5"), r.get("recall10"),
            "是" if r.get("page_hit") else "否",
            "是" if r.get("cross_corpus_leak") else "否",
            r.get("total_ms"), r.get("generation_method"),
        ])
    return out


def _summary_md(summary: dict[str, Any]) -> str:
    if not summary:
        return "_（评估汇总尚未生成）_"
    acc = summary.get("准确率") or {}
    rec = summary.get("检索召回") or {}
    lat = summary.get("响应时间ms") or {}
    sla = summary.get("SLA") or {}
    ref = summary.get("拒答") or {}
    leak = summary.get("跨语料泄漏") or {}
    cfg = summary.get("配置") or {}
    return (
        f"### 汇总指标（{summary.get('生成时间', '—')}）\n\n"
        f"| 指标 | 值 |\n|---|---|\n"
        f"| 严格准确率 | **{acc.get('严格判对题数')}/{summary.get('总数')} = {acc.get('严格准确率')}**"
        f"（判对 {acc.get('判对题号')}） |\n"
        f"| 宽松准确率（含部分正确） | **{acc.get('宽松准确率(严格+部分)')}**"
        f"（部分正确 {acc.get('部分正确题号')}） |\n"
        f"| 平均关键词覆盖率 | {acc.get('平均关键词覆盖率')} |\n"
        f"| recall@5 / recall@10 | **{rec.get('平均recall@5')}** / **{rec.get('平均recall@10')}** |\n"
        f"| 响应时间 均值 / P90 / 最大 | {lat.get('平均')} / {lat.get('P90')} / {lat.get('最大')} ms |\n"
        f"| SLA ≤{sla.get('阈值ms')}ms 达标率 | **{sla.get('达标率')}**（{sla.get('达标题数')} 题）"
        f"{'，超时 ' + str(sla.get('超时题号')) if sla.get('超时题号') else ''} |\n"
        f"| 拒答率 | {ref.get('拒答率')}（{ref.get('拒答题号')}，原因 {ref.get('按原因')}） |\n"
        f"| 跨语料泄漏 | top-3 **{leak.get('题数')}** 题 {leak.get('题号')}｜引用级 "
        f"**{leak.get('引用级泄漏题数')}** 题 {leak.get('引用级泄漏题号')} |\n"
        f"| 索引规模 | {cfg.get('索引chunk_total')} 块（{cfg.get('embed_model')}） |\n"
        f"\n> 完整逐题证据与根因见 `测试/测试数据/检索问题分析.md`。"
    )


def _flow_md(flow: dict[str, Any]) -> str:
    steps = list(flow.get("steps") or [])
    if not steps:
        return "_（流程数据未就绪）_"
    parts = [f"### 测试评估流程（{flow.get('work_order')}）\n"]
    for i, s in enumerate(steps, 1):
        title = s.get("title") or s.get("name") or f"步骤 {i}"
        detail = s.get("detail") or s.get("desc") or ""
        items = s.get("items") or s.get("metrics") or []
        parts.append(f"**{i}. {title}**\n")
        if detail:
            parts.append(f"{detail}\n")
        if isinstance(items, list):
            for it in items[:8]:
                if isinstance(it, dict):
                    parts.append(f"- {it.get('label') or it.get('name')}：{it.get('value')}")
                else:
                    parts.append(f"- {it}")
        parts.append("")
    parts.append(f"\n> 仓库：`{flow.get('repo_root')}`｜数据快照：{flow.get('generated_at')}")
    return "\n".join(parts)


def build_app(gr: Any, dash: Any, *, box: Any, top_k_default: int = DEFAULT_TOP_K) -> Any:
    """组装 Gradio Blocks（三标签页，与标准库版一一对应）。"""
    examples = [[q["q"]] for q in (dash._example_questions() or [])][:6]

    with gr.Blocks(title=f"工单7 测试评估台（Gradio）· {WORK_ORDER}") as app:
        gr.Markdown(
            f"# 工单7 测试评估台 · {WORK_ORDER}\n"
            "Gradio 版界面（任务书原计划设计）。后端与 `部署/界面/测试评估台.py` **完全共用**："
            "同一个 `QAEngine`、同一套评估产物，换界面不换逻辑。"
        )
        with gr.Tabs():
            # ---------------- 标签页 1：问答测试 ----------------
            with gr.Tab("问答测试"):
                with gr.Row():
                    with gr.Column(scale=3):
                        q_box = gr.Textbox(label="问题（中英文均可）", lines=3,
                                           placeholder="例如：平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？")
                        with gr.Row():
                            lang_dd = gr.Dropdown(["auto", "zh", "en"], value="auto", label="语言")
                            topk_sl = gr.Slider(1, 20, value=top_k_default, step=1, label="Top-K")
                            ask_btn = gr.Button("提问", variant="primary")
                        gr.Examples(examples=examples, inputs=q_box, label="题库示例（点击填入）")
                    with gr.Column(scale=2):
                        status_md = gr.Markdown("_引擎状态加载中…_")
                answer_md = gr.Markdown(label="答案")
                latency_md = gr.Markdown()
                with gr.Accordion("Top-K 检索证据", open=True):
                    hits_md = gr.Markdown("_尚未提问_")

                def _on_ask(question: str, lang: str, top_k: float):
                    if not str(question or "").strip():
                        return "**请输入问题**（空输入会被闸门拒答为「不清楚」）", "", "_（无）_"
                    try:
                        payload = box.ask(str(question), lang=str(lang), top_k=int(top_k))
                    except Exception as exc:                       # noqa: BLE001
                        return f"❌ 调用失败：{type(exc).__name__}: {exc}", "", ""
                    head = f"### 答案\n\n{payload.get('answer')}"
                    if payload.get("refused"):
                        head += (f"\n\n> 拒答原因：`{payload.get('refusal_reason')}`"
                                 f"（{payload.get('refusal_detail')}）")
                    cites = payload.get("cited_pages") or []
                    if cites:
                        head += f"\n\n**引用页**：{cites}"
                    return head, _fmt_latency(payload), _fmt_hits(payload)

                ask_btn.click(_on_ask, inputs=[q_box, lang_dd, topk_sl],
                              outputs=[answer_md, latency_md, hits_md])
                q_box.submit(_on_ask, inputs=[q_box, lang_dd, topk_sl],
                             outputs=[answer_md, latency_md, hits_md])

            # ---------------- 标签页 2：评估看板 ----------------
            with gr.Tab("评估看板"):
                refresh_btn = gr.Button("刷新产物", variant="secondary")
                summary_md = gr.Markdown("_加载中…_")
                table = gr.Dataframe(
                    headers=["题号", "题型", "对应公司", "lang", "答案（前 90 字）", "覆盖率",
                             "判对", "recall@5", "recall@10", "页命中", "跨语料泄漏",
                             "总耗时ms", "生成方式"],
                    wrap=True, interactive=False, label="17 题逐题结果")
                analysis_md = gr.Markdown(label="检索问题分析", value="_加载中…_")

                def _load_eval():
                    ev = dash.build_eval_payload()
                    if not ev.get("ready"):
                        return ("### ⚠️ 评估产物未就绪\n\n请先运行：\n\n```\n"
                                "python 研发/评估/运行评估.py\n```"), [], "_（无）_"
                    ana = ev.get("analysis") or {}
                    body = ana.get("preamble_html") or ""
                    for name, text in (ana.get("sections") or [])[:6]:
                        body += f"\n\n### {name}\n\n{text[:1500]}"
                    return (_summary_md(ev.get("summary") or {}),
                            _answers_table(ev.get("rows") or []),
                            body or "_（分析文档为空）_")

                refresh_btn.click(_load_eval, outputs=[summary_md, table, analysis_md])
                app.load(_load_eval, outputs=[summary_md, table, analysis_md])

            # ---------------- 标签页 3：测试流程演示 ----------------
            with gr.Tab("测试流程演示"):
                flow_btn = gr.Button("刷新流程数据", variant="secondary")
                flow_md = gr.Markdown("_加载中…_")
                health_json = gr.JSON(label="健康检查 / 环境")

                def _load_flow():
                    return _flow_md(dash.build_flow_payload()), dash.build_health_payload()

                flow_btn.click(_load_flow, outputs=[flow_md, health_json])
                app.load(_load_flow, outputs=[flow_md, health_json])

        gr.Markdown(
            "---\n"
            f"**工单编号**：{WORK_ORDER}　｜　"
            "**界面版本**：Gradio（原计划设计）＋ 标准库兜底版（`测试评估台.py`）　｜　"
            "**推理后端**：本地 Ollama（`qwen2.5:3b` 生成 / `bge-m3` 嵌入）\n\n"
            "> 提示：Ollama 是 CPU 单实例，问答内部已串行化；界面首次提问前请等待预热完成。"
        )
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单7：{WORK_ORDER} —— Gradio 测试评估台")
    parser.add_argument("--port", type=int, default=7860, help="监听端口（默认 7860）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--no-warmup", action="store_true", help="不预热（首问会慢，仅调试）")
    parser.add_argument("--share", action="store_true", help="生成公网链接（Gradio share）")
    args = parser.parse_args(argv)

    gr = _require_gradio()
    dash = _load_dashboard()

    cfg_dir = Path(str(getattr(dash, "REPO_ROOT", REPO_ROOT))) / "研发" / "data" / "index"
    manifest = cfg_dir / "index_manifest.json"
    print("=" * 78)
    print(f"工单7 测试评估台（Gradio）｜{WORK_ORDER}")
    print(f"仓库  ：{REPO_ROOT}")
    print(f"gradio：{gr.__version__}")
    print(f"索引  ：{manifest}（{'存在' if manifest.is_file() else '缺失'}）")
    print("=" * 78)
    if not manifest.is_file():
        print("❌ 索引缺失：请先运行 研发/scripts/build_index.py（见 README §五 步骤 1）", file=sys.stderr)
        return 4

    box = dash.EngineBox()
    if not args.no_warmup:
        box.start_warmup(with_generate=True)     # 后台预热：不预热时首问可达 7.5 s（必违约）
    else:
        box.state = "ready"
        box.engine = None
        print("⚠️ --no-warmup：问答会在首次调用时初始化（首问延迟显著变大）")

    app = build_app(gr, dash, box=box)
    print(f"启动：http://{args.host}:{args.port}/  （Ctrl+C 停止）")
    # Gradio 6 起 theme 从 Blocks 构造器移到 launch()（旧写法会打 UserWarning）
    launch_kwargs: dict[str, Any] = {"server_name": args.host, "server_port": args.port,
                                     "share": bool(args.share), "show_error": True,
                                     "inbrowser": False, "prevent_thread_lock": False}
    try:
        app.queue().launch(theme=gr.themes.Soft(), **launch_kwargs)
    except TypeError:                            # 兼容 Gradio 5 及更早（theme 在 Blocks 上）
        app.queue().launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已停止]", file=sys.stderr)
        raise SystemExit(0)
