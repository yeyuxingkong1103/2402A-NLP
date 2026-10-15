"""工单16：真实训练结果浏览页，不在打开页面时重新训练或伪造推理。"""

import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI()


@app.get("/api/report")
def report():
    return json.loads((ROOT / "测试/真实微调结果.json").read_text(encoding="utf-8"))


@app.get("/")
def index():
    return HTMLResponse(
        """<!doctype html><meta charset="utf-8"><style>body{font:16px/1.8 Microsoft YaHei;color:#203b45;background:#f1f5f6;margin:40px}.card{background:white;padding:25px;margin:20px 0;border-radius:12px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccd9df}</style><h1>工单16 · 专用视觉语言模型真实LoRA实验</h1><p>设计 · 研发 · 测试 · 优化 · 部署</p><div class="card"><h2>数据隔离与训练</h2><p id="data"></p><p>下列为已实际运行的结果。20条训练、4条验证、8条测试，按专利文档隔离。低分辨率适配8GB显存，小样本结果不能代表全量工业能力。</p><table id="metrics"></table></div><div class="card"><h2>训练损失</h2><pre id="loss"></pre></div><div class="card"><h2>未见测试题的原始输出</h2><pre id="samples"></pre></div><script>fetch('/api/report').then(r=>r.json()).then(r=>{data.textContent=r.model+' / '+r.method+' / epochs='+r.epochs;metrics.innerHTML='<tr><th>指标</th><th>微调前</th><th>微调后</th></tr>';for(const k of ['accuracy','terminology_accuracy','drawing_reasoning_accuracy']){const row=document.createElement('tr');for(const x of [k,r.before[k],r.after[k]]){const td=document.createElement('td');td.textContent=x;row.append(td)}metrics.append(row)}loss.textContent=JSON.stringify(r.training,null,2);samples.textContent=JSON.stringify(r.after.samples,null,2)})</script>"""
    )
