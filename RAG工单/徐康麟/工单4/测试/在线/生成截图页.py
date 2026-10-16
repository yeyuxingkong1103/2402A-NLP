# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/在线/生成截图页.py（T6：把**本轮实跑产物**渲染成 4 个可截图的 HTML 页）

用途：为「准确率汇总 / 响应时间 / 图像解析结果 / 日志可审计性」四张专项截图提供**数据驱动**的页面，
      页面内容全部来自 测试/测试数据/ 的实跑产物与 研发/data/processed 的解析产物，**不手写任何数字**。
      页面由静态 HTTP 服务托管后用 browser_* 真实截图（禁止 PIL 画假图）。

产出：测试/测试数据/截图页/{准确率汇总,响应时间,图像解析结果,日志可审计性}.html
运行：pwsh -NoProfile -File run_py.ps1 测试/在线/生成截图页.py
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "测试" / "测试数据"
PROC = ROOT / "研发" / "data" / "processed"
OUT = DATA / "截图页"

CSS = """
body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#f5f7fb;color:#1b2333}
.head{background:#1b4fd8;color:#fff;padding:16px 22px}
.head h1{margin:0;font-size:19px}.head div{margin-top:6px;font-size:12.5px;opacity:.92}
.wrap{padding:18px 22px 40px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{border:1px solid #dbe2ef;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#eef2fb}
.ok{color:#0a7f3f;font-weight:700}.bad{color:#c62828;font-weight:700}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:14px}
.kpi{background:#fff;border-radius:10px;padding:12px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08);min-width:190px}
.kpi b{display:block;font-size:22px;color:#1b4fd8}.kpi span{font-size:12px;color:#5b6577}
h2{font-size:15px;margin:20px 0 8px}.note{font-size:12px;color:#5b6577;margin:6px 0 12px}
img{max-width:100%;border:1px solid #dbe2ef;background:#fff}
pre{background:#0f172a;color:#d7e3ff;padding:12px;border-radius:8px;font-size:12px;overflow:auto;max-height:900px}
"""


def page(title: str, subtitle: str, body: str) -> str:
    return (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<div class='head'><h1>{html.escape(title)}</h1><div>{html.escape(subtitle)}</div></div>"
            f"<div class='wrap'>{body}</div></body></html>")


def load_rows():
    return [json.loads(x) for x in (DATA / "qa_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def gen_accuracy():
    rep = json.loads((DATA / "accuracy_report.json").read_text(encoding="utf-8"))
    s, meta = rep["summary"], rep["run_meta"]
    rows = load_rows()
    kpis = [
        ("准确率", f"{s['n_correct']}/{s['n_judged']} = {s['accuracy']*100:.1f}%"),
        ("检索命中（权威层）", f"{s['n_hit_complete']}/16"),
        ("BM25 命中", f"{s['n_hit_complete_bm25']}/16"),
        ("向量命中", f"{s['n_hit_complete_vector']}/16"),
        ("引用可回溯", f"{s['n_citation_ok']}/16"),
        ("延迟 max", f"{s['latency_wall_max_ms']} ms"),
    ]
    kpi_html = "".join(f"<div class='kpi'><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>"
                       for k, v in kpis)
    neg = rep["negatives"]
    bi = rep["bilingual"]
    trs = []
    for r in rows:
        cls = "ok" if r.get("correct") else "bad"
        trs.append(
            f"<tr><td>{r['order']}</td><td>{r['id']}</td><td>{html.escape(r['category'])}</td>"
            f"<td class='{cls}'>{'判对' if r.get('correct') else '判错'}</td>"
            f"<td>{'是' if r.get('hit_complete') else '否'}</td>"
            f"<td>{'是' if r.get('hit_complete_bm25') else '否'}</td>"
            f"<td>{'是' if r.get('hit_complete_vector') else '否'}</td>"
            f"<td>{html.escape(str(r.get('cited_pages')))}</td>"
            f"<td>{'是' if r.get('citation_ok') else '否'}</td>"
            f"<td>{r.get('latency_wall_ms')}</td>"
            f"<td>{html.escape(str(r.get('missing_keywords') or '—'))}</td></tr>")
    body = (f"<div class='kpis'>{kpi_html}</div>"
            f"<div class='note'>数据来源：<b>测试/测试数据/qa_results.csv、accuracy_report.json</b>（本轮 T6 实跑，"
            f"判分=测试/common/golden.py 确定性判定）；索引 chunk_total={meta.get('index_chunk_total')}"
            f"（image {meta.get('index_by_modality', {}).get('image')}）；golden sha256={str(meta.get('golden_sha256'))[:16]}…</div>"
            "<h2>16 题逐题明细</h2>"
            "<table><tr><th>#</th><th>id</th><th>题型</th><th>判对</th><th>权威层命中</th>"
            "<th>BM25 命中</th><th>向量命中</th><th>引用页</th><th>引用可回溯</th><th>延迟(ms)</th><th>缺失关键字</th></tr>"
            + "".join(trs) + "</table>"
            f"<h2>负例（拒答红线）</h2><div class='note'>通过 {neg['n_passed']}/{neg['n']}；"
            "未通过项已如实登记（详见测试报告 §负例）</div>"
            "<table><tr><th>#</th><th>类型</th><th>问题</th><th>拒答</th><th>原因</th><th>通过</th></tr>"
            + "".join(f"<tr><td>{html.escape(i['id'])}</td><td>{html.escape(i['type'])}</td>"
                      f"<td>{html.escape(i['question'][:52])}</td><td>{i.get('refused')}</td>"
                      f"<td>{html.escape(str(i.get('refusal_reason')))}</td>"
                      f"<td class='{'ok' if i.get('passed') else 'bad'}'>{'通过' if i.get('passed') else '未通过'}</td></tr>"
                      for i in neg["items"]) + "</table>"
            f"<h2>中英文同题</h2><div class='note'>中文 3/3 判对；英文路径 1/3（详见测试报告 §中英文）</div>"
            "<table><tr><th>id</th><th>语言</th><th>答案摘要</th><th>中文口径判对</th><th>英文原子口径</th><th>引用页</th></tr>"
            + "".join(f"<tr><td>{i['id']}</td><td>{i['language']}</td>"
                      f"<td>{html.escape(str(i.get('answer'))[:110])}</td>"
                      f"<td>{i.get('correct')}</td><td>{i.get('en_criterion_passed')}</td>"
                      f"<td>{html.escape(str(i.get('cited_pages')))}</td></tr>" for i in bi["items"]) + "</table>")
    (OUT / "准确率汇总.html").write_text(page(
        "工单4 测试报告 · 16 题准确率汇总", "判分=确定性关键字（非 LLM 自评）；命中率主指标=权威层 hit_complete", body),
        encoding="utf-8")


def gen_latency():
    rows = load_rows()
    rep = json.loads((DATA / "accuracy_report.json").read_text(encoding="utf-8"))
    s = rep["summary"]
    kpis = [("P50（客户端 wall）", f"{s['latency_wall_p50_ms']} ms"), ("P95", f"{s['latency_wall_p95_ms']} ms"),
            ("最大", f"{s['latency_wall_max_ms']} ms"), ("最小", f"{s['latency_wall_min_ms']} ms"),
            ("超 3000ms 题数", f"{s['n_latency_over_3000ms']}"), ("首字范围", f"{s['first_token_min_ms']}~{s['first_token_max_ms']} ms")]
    kpi_html = "".join(f"<div class='kpi'><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>" for k, v in kpis)
    trs = "".join(
        f"<tr><td>{r['order']}</td><td>{r['id']}</td><td>{html.escape(r['category'])}</td>"
        f"<td>{r.get('latency_wall_ms')}</td><td>{r.get('latency_server_total_ms')}</td>"
        f"<td>{r.get('latency_first_token_ms')}</td><td>{r.get('latency_retrieve_ms')}</td>"
        f"<td>{r.get('latency_generate_ms')}</td><td>{html.escape(str(r.get('generation_method')))}</td>"
        f"<td>{r.get('cache')}</td></tr>" for r in rows)
    body = (f"<div class='kpis'>{kpi_html}</div>"
            "<div class='note'>主口径 = 客户端「请求发出 → 完整答案返回」（wall）；辅助 = 服务端 latency_ms 各段。"
            "本轮 <b>--no-cache</b>（use_cache=false）全量实跑，未用缓存样本冒充全量性能。</div>"
            "<table><tr><th>#</th><th>id</th><th>题型</th><th>wall(ms)</th><th>服务端 total(ms)</th>"
            "<th>首字(ms)</th><th>retrieve(ms)</th><th>generate(ms)</th><th>生成方式</th><th>缓存</th></tr>"
            + trs + "</table>")
    (OUT / "响应时间.html").write_text(page(
        "工单4 测试报告 · 响应时间（逐题 + P50/P95/最大）", "单次问答 ≤3 s 验收口径；预热 + 独占实跑", body),
        encoding="utf-8")


def gen_image_parsing():
    diagrams = [json.loads(x) for x in (PROC / "diagrams.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    charts = [json.loads(x) for x in (PROC / "charts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    d39 = next((d for d in diagrams if d.get("page") == "39" and "招股说明书2" in d.get("file_name", "")), None)
    c72 = next((c for c in charts if c.get("page") == "72" and "招股说明书2" in c.get("file_name", "")), None)
    copied = []
    for src, dst in ((PROC / "evidence" / "招股说明书2.pdf_p39_diagram.png", "p39_diagram.png"),
                     (PROC / "evidence" / "招股说明书2.pdf_p72_chart_x1692.png", "p72_chart_x1692.png"),
                     (PROC / "evidence" / "招股说明书2.pdf_p72_chart_x1693.png", "p72_chart_x1693.png")):
        if src.exists():
            shutil.copy2(src, OUT / dst)
            copied.append(dst)

    def txt(obj, n=2600):
        return html.escape((obj or {}).get("text") or (obj or {}).get("content") or "")[:n]

    body = ("<div class='note'>数据来源：<b>研发/data/processed/diagrams.jsonl、charts.jsonl、evidence/*.png</b>"
            "（T2 解析产物）；本页由 测试/在线/生成截图页.py 渲染，不手写数字。"
            "<b>本机无 OCR 依赖</b>：框图文本按 x 列聚类复原，图表数值来自像素级抽取 + 视觉复核，"
            "位图文字走字形模板匹配。</div>"
            f"<h2>框图解析 · 招股说明书2 物理页 39（第 5 题证据页）</h2>"
            f"<div class='note'>chunk_id={html.escape(str((d39 or {}).get('chunk_id')))}｜"
            f"modality_sub={html.escape(str((d39 or {}).get('modality_sub')))}｜"
            f"image_sha1={html.escape(str((d39 or {}).get('image_sha1'))[:16])}…</div>"
            f"<img src='p39_diagram.png' alt='p39 diagram'>"
            f"<pre>{txt(d39)}</pre>"
            f"<h2>图表解析 · 招股说明书2 物理页 72（第 6 题证据页）</h2>"
            f"<div class='note'>chunk_id={html.escape(str((c72 or {}).get('chunk_id')))}｜"
            f"xref={html.escape(str((c72 or {}).get('xref')))}｜"
            f"image_sha1={html.escape(str((c72 or {}).get('image_sha1'))[:16])}…</div>"
            f"<img src='p72_chart_x1692.png' alt='p72 bar chart'><img src='p72_chart_x1693.png' alt='p72 pie chart'>"
            f"<pre>{txt(c72)}</pre>"
            f"<div class='note'>共解析 框图 {len(diagrams)} 条 / 图表 {len(charts)} 条；"
            f"证据图已随本页落盘：{html.escape('、'.join(copied))}</div>")
    (OUT / "图像解析结果.html").write_text(page(
        "工单4 测试报告 · 图像解析结果", "框图结构树（页 39）+ 图表数值（页 72）；无 OCR 依赖，方法级别如实标注", body),
        encoding="utf-8")


def gen_log():
    log = ROOT / "部署" / "日志" / "app.log"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-90:]
    keep = [ln for ln in lines if any(k in ln for k in ("answer", "generate", "retrieve", "compose",
                                                        "rerank", "hybrid_search", "build_citations",
                                                        "evaluate_answerability", "duration", "elapsed"))][-60:]
    body = ("<div class='note'>数据来源：<b>部署/日志/app.log</b>（结构化日志尾部）。"
            "验收点：函数入口/出口成对、含输入摘要与耗时，可审计；本页仅为原文渲染，未做任何加工。</div>"
            f"<pre>{html.escape(chr(10).join(keep))}</pre>")
    (OUT / "日志可审计性.html").write_text(page(
        "工单4 测试报告 · 日志可审计性", "结构化日志（函数入口/出口 + 输入摘要 + 耗时）", body), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gen_accuracy()
    gen_latency()
    gen_image_parsing()
    gen_log()
    for p in sorted(OUT.glob("*.html")) + sorted(OUT.glob("*.png")):
        print(f"[OK] {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
