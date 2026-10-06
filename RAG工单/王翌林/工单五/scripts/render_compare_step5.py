# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/render_compare_step5.py —— 工单四：评估结果渲染为 HTML 对比页（供截图）
import json
from pathlib import Path

r = json.loads(Path("docs/eval_v4_results.json").read_text(encoding="utf-8"))
sm, rows = r["summary"], r["rows"]
WO = "人工智能NLP-RAG-图像内容解析及检索优化"

rows_html = ""
for mode, label in (("pure_llm", "纯LLM"), ("v3", "工单三(v3)"), ("v4", "工单四(v4)")):
    a = sm[mode]["all"]
    cls = ' class="v4"' if mode == "v4" else ""
    rows_html += (f"<tr{cls}><td>{label}</td><td>{a['accuracy']*100:.1f}%</td>"
                  f"<td>{a['faithfulness']}</td><td>{a['relevance']}</td>"
                  f"<td>{a['ctx_precision']}</td><td>{a['ctx_recall']}</td>"
                  f"<td>{a['avg_latency_ms']:.0f}</td></tr>")
qt_html = ""
for qt, name in (("text", "文本题"), ("table", "表格题"), ("image", "图像题")):
    for mode in ("v3", "v4"):
        if qt in sm[mode]:
            a = sm[mode][qt]
            cls = ' class="v4"' if mode == "v4" else ""
            qt_html += (f"<tr><td>{name}-{mode}</td>"
                        f"<td>{a['accuracy']*100:.1f}%</td>"
                        f"<td>{a['faithfulness']}</td><td>{a['ctx_recall']}</td>"
                        f"<td>{a['avg_latency_ms']:.0f}</td></tr>")

# 工单四：图像题按题分组，v3/v4 答案对照（完整答案，便于截图佐证）
img_rows = ""
for qid in (105, 106):
    sub = [x for x in rows if x["id"] == qid and x["qtype"] == "image"]
    if not sub:
        continue
    q = sub[0]["question"]
    img_rows += f"<tr><th colspan=5 style=text-align:left>IMG-{qid-100}：{q}</th></tr>"
    for x in sub:
        cls = ' class="v4"' if x["mode"] == "v4" else ""
        img_rows += (f"<tr{cls}><td>{x['mode']}</td><td>{x['answer']}</td>"
                     f"<td>{'✓' if x['accuracy'] else '✗'} {' '.join(x['kws_hit'])}</td>"
                     f"<td>{x['latency_ms']:.0f}</td><td>{x['n_images']}</td></tr>")

# 工单四：图像解析结果样例（重点图 caption/OCR/VQA 摘要 + 缩略图）
parsed_file = Path("data/image_descriptions/招股说明书2_images_parsed.json")
sample_html = ""
if parsed_file.exists():
    parsed = json.loads(parsed_file.read_text(encoding="utf-8"))
    items = parsed if isinstance(parsed, list) else parsed.get("images", parsed.get("items", []))
    key_ids = {"img_008", "img_011", "img_012"}
    for im in items:
        if str(im.get("image_id")) not in key_ids:
            continue
        rel = im.get("path", "")
        rel_md = rel.replace("../", "") if rel.startswith("../") else rel
        src = f"../{rel_md}"  # 工单四：html 位于 docs/，图片相对路径需回退一级
        ocr = (im.get("ocr_text") or "").replace("\n", " / ")[:150]
        vqa = [qa for qa in (im.get("vqa_qa") or []) if isinstance(qa, dict)]
        # 工单四：兼容 vqa_qa 实际字段 q/a（image_vqa.py 输出）
        vqa_html = "".join(
            f"<li><b>{qa.get('q') or qa.get('question','')}</b> → "
            f"{qa.get('a') or qa.get('answer','')}</li>"
            for qa in vqa[:3]) or "<li>（无 VQA 结果）</li>"
        cap = (im.get("caption") or "").strip() or "（无 caption）"
        sample_html += (
            f"<div style='display:flex;gap:16px;margin:12px 0;border:1px solid #ddd;padding:10px'>"
            f"<img src='{src}' style='width:300px;object-fit:contain'>"
            f"<div style='flex:1;font-size:13px'><b>image_id:</b> {im.get('image_id')} ｜ "
            f"<b>page:</b> {im.get('page')}<br><b>caption:</b> {cap}<br>"
            f"<b>OCR:</b> {ocr}<br><b>VQA:</b><ul style=margin:4px>{vqa_html}</ul></div></div>")

v3a, v4a = sm["v3"]["all"]["accuracy"], sm["v4"]["all"]["accuracy"]
lift = (v4a - v3a) / max(v3a, 0.01) * 100
lat = {m: sm[m]["all"]["avg_latency_ms"] for m in ("pure_llm", "v3", "v4")}

html = f"""<!DOCTYPE html><html lang=zh><meta charset=utf-8>
<title>工单四 图像解析与检索优化对比</title>
<style>body{{font-family:'Microsoft YaHei';margin:24px;max-width:1200px;background:#fff;color:#222}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:6px 10px;font-size:14px;color:#222}}th{{background:#eef}}
.v4{{background:#e8f5e9}}h2{{border-left:4px solid #2e7d32;padding-left:8px;margin-top:28px}}
small{{color:#666}}</style>
<h1>图像解析与检索优化对比报告 <small>工单编号：{WO}</small></h1>
<p>评估时间：{r['ts']} ｜ 16 题（T01-T14 + IMG-05/06）｜ 三模式对照 ｜ {r.get('note','')}</p>
<h2>一、整体指标对比（工单三 vs 工单四）</h2>
<table><tr><th>模式</th><th>准确率</th><th>忠实度</th><th>答案相关性</th>
<th>上下文精度</th><th>上下文召回</th><th>平均响应ms</th></tr>{rows_html}</table>
<p><b>准确率提升：v4 {v4a*100:.1f}% vs v3 {v3a*100:.1f}%（{(v4a-v3a)*100:+.1f}pp，相对提升 {lift:+.0f}%）</b></p>
<h2>二、分题型准确率（文本 / 表格 / 图像）</h2>
<table><tr><th>题型-模式</th><th>准确率</th><th>忠实度</th><th>上下文召回</th><th>平均响应ms</th></tr>{qt_html}</table>
<h2>三、响应时间对比（ms）</h2>
<table><tr><th>模式</th><th>平均响应</th><th>说明</th></tr>
<tr><td>纯LLM</td><td>{lat['pure_llm']:.0f}</td><td>无检索基线</td></tr>
<tr><td>工单三(v3)</td><td>{lat['v3']:.0f}</td><td>文本+表格双路 RRF + 重排 + LLM</td></tr>
<tr class=v4><td>工单四(v4)</td><td>{lat['v4']:.0f}</td><td>+ 图像三子通道召回与图像感知路由</td></tr></table>
<p><b>v4 相对 v3 延迟增量：{lat['v4']-lat['v3']:+.0f}ms</b>（图像通道召回 ~0.3s，热启动后单问 ≤3s 达标）</p>
<h2>四、图像类问题逐题答案对照（工单三 vs 工单四）</h2>
<table><tr><th>模式</th><th>答案</th><th>关键词命中</th><th>延迟ms</th><th>召回图像数</th></tr>{img_rows}</table>
<h2>五、图像解析结果样例（招股说明书2 重点图）</h2>
{sample_html or '<p>解析文件缺失</p>'}
</html>"""
out = Path("docs/compare_v4.html")
out.write_text(html, encoding="utf-8")
print(f"OK -> {out.resolve()}")
