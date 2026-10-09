# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""把检索对比结果渲染成报告页并截图（产出物 3 的可视化证据）

工单格式要求「测试：图片 一定要多截图」。知识图谱那部分由 `图谱截图.py`
驱动 neo4j Browser 截，而**检索结果对比**没有一个现成的界面可截
（本工单不要求做 Web UI，见 设计说明 第六节），所以这里把
`results/{rag,lightrag,ragas}.json` 里的**真实数据**渲染成报告页再截。

⚠️ 渲染出来的是**真实结果的视图**，不是另做的界面 —— 页面上也标了这句话，
免得看的人误以为有这么一个实时系统。

用法（rag_gd 环境有 playwright）：
    D:/Anaconda/envs/rag_gd/python.exe 对比截图.py
截图输出到 测试/截图/
"""
import html
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHOTS = HERE / "截图"
TMP = HERE / "_render"

# 截哪几题：挑「两套系统表现不同」的典型，比全部 16 题更能说明问题
CASES = [
    (543, "注册资本（事实型，两套都该答对）"),
    (3, "关联方与持股比例（关系型，图谱的主场）"),
    (795, "国家科技进步一等奖（叙述型，答案跨多段）"),
]

CSS = """
body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;padding:32px 40px;
     background:#fafafa;color:#1a1a1a;font-size:15px;line-height:1.7}
h1{font-size:26px;margin:0 0 6px} h2{font-size:19px;margin:28px 0 10px;
   border-left:4px solid #c2410c;padding-left:10px}
.sub{color:#666;font-size:13px;margin-bottom:20px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:14px;
      box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{border:1px solid #e2e2e2;padding:8px 12px;text-align:left;vertical-align:top}
th{background:#f3f4f6;font-weight:600}
.q{background:#fff;border:1px solid #e2e2e2;border-radius:6px;padding:18px 22px;
   margin-bottom:22px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.qid{color:#c2410c;font-weight:700;font-size:15px}
.gold{background:#f0f9ff;border-left:3px solid #0369a1;padding:8px 12px;
      margin:10px 0;font-size:14px}
.ans{margin-top:12px} .ans b{display:inline-block;min-width:88px}
.rag b{color:#0369a1} .lr b{color:#c2410c}
.txt{background:#f9fafb;border:1px solid #eee;border-radius:4px;padding:10px 14px;
     margin-top:6px;white-space:pre-wrap;font-size:14px}
.up{color:#15803d;font-weight:700} .down{color:#b91c1c;font-weight:700}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:10px 14px;
      font-size:13px;color:#92400e;margin-bottom:22px}
"""


def esc(t):
    return html.escape(str(t or ""), quote=False)


def load(kb):
    return json.loads((RESULTS / f"{kb}.json").read_text(encoding="utf-8"))["records"]


def ragas_table():
    p = RESULTS / "ragas.json"
    if not p.exists():
        return "<p>（未运行 ragas_eval.py）</p>"
    r = json.loads(p.read_text(encoding="utf-8"))
    names = {"faithfulness": "回答忠于上下文", "answer_relevancy": "回答切题程度",
             "context_precision": "检索上下文排序质量", "context_recall": "标准答案覆盖率"}
    rows = []
    for k, cn in names.items():
        a, b = r.get("rag", {}).get(k), r.get("lightrag", {}).get(k)
        if a is None or b is None:
            continue
        d = b - a
        cls = "up" if d > 0 else "down"
        rows.append(f"<tr><td>{esc(k)}</td><td>{esc(cn)}</td><td>{a:.4f}</td>"
                    f"<td>{b:.4f}</td><td class='{cls}'>{d:+.4f}</td></tr>")
    return ("<table><tr><th>指标</th><th>含义</th><th>RAG</th>"
            "<th>LightRAG</th><th>差异</th></tr>" + "".join(rows) + "</table>")


def build_report_page():
    """RAGAS 指标对比 + 逐题汇总。"""
    rag, lr = load("rag"), load("lightrag")
    lr_by = {r["id"]: r for r in lr}
    rows = []
    for r in rag:
        b = lr_by.get(r["id"])
        if not b:
            continue
        rows.append(f"<tr><td>{r['id']}</td><td>{esc(r['question'][:46])}…</td>"
                    f"<td>{r['seconds']}s / {len(r['contexts'])}段</td>"
                    f"<td>{b['seconds']}s / {len(b['contexts'])}段</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>RAG vs LightRAG 对比报告</h1>
<div class="sub">工单编号：人工智能NLP-RAG项目-LightRAG优化　|
16 道工单指定问题　|　语料：招股说明书1（兴图新科 548页）+ 招股说明书2（力源信息 350页）</div>
<div class="note">本页由 <code>测试/对比截图.py</code> 从 results/*.json 的真实数据渲染而成，
用于留档；本工单没有做 Web 界面，页面上不是实时系统。</div>
<h2>一、RAGAS 指标对比</h2>
{ragas_table()}
<h2>二、逐题检索结果对比</h2>
<table><tr><th>id</th><th>问题</th><th>RAG</th><th>LightRAG</th></tr>
{''.join(rows)}</table>
</body></html>"""


def build_answer_page():
    """逐题并把两套系统的回答并排 —— 检索对比最直观的一页。"""
    rag = {r["id"]: r for r in load("rag")}
    lr = {r["id"]: r for r in load("lightrag")}
    blocks = []
    for qid, tag in CASES:
        a, b = rag.get(qid), lr.get(qid)
        if not a or not b:
            continue
        blocks.append(f"""
<div class="q">
  <div class="qid">id={qid}　{esc(tag)}</div>
  <div style="margin-top:6px"><b>{esc(a['question'])}</b></div>
  <div class="gold"><b>标准答案</b>：{esc(a.get('gold_answer'))}</div>
  <div class="ans rag"><b>RAG</b>（{a['seconds']}s，召回 {len(a['contexts'])} 段）
    <div class="txt">{esc(a['answer'])}</div></div>
  <div class="ans lr"><b>LightRAG</b>（{b['seconds']}s，召回 {len(b['contexts'])} 段原文
    + {b.get('kg_chars', 0):,} 字符图谱数据）
    <div class="txt">{esc(b['answer'])}</div></div>
</div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>检索结果对比 · 逐题明细</h1>
<div class="sub">同一批语料、同一个大模型配置，唯一变量是检索机制</div>
<div class="note">本页由 <code>测试/对比截图.py</code> 从 results/*.json 的真实回答渲染而成，
不是实时系统界面。</div>
{''.join(blocks)}
</body></html>"""


def main():
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    pages = [("09-对比报告-RAGAS与逐题", build_report_page()),
             ("10-检索对比-逐题回答", build_answer_page())]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        for name, html_text in pages:
            f = TMP / f"{name}.html"
            f.write_text(html_text, encoding="utf-8")
            print(f"[截图] {name}", flush=True)
            page.goto(f.as_uri(), wait_until="load")
            page.wait_for_timeout(600)
            page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
        browser.close()

    got = sorted(SHOTS.glob("*.png"))
    print(f"\n完成：截图目录共 {len(got)} 张")
    for g in got:
        print(f"  {g.name}  ({g.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
