# -*- coding: utf-8 -*-
"""
知识图谱可视化 + 「实体/关系创建思路」说明材料生成
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

对应验收标准「知识图谱可视化：可视化展示 PDF 中的文档解析出来的知识图谱数据中的实体、关系」，
以及演示视频要求「展示 Graph RAG 构建的知识图谱可视化，以及实体、关系创建思路的讲解」。

本脚本产出三类可视化/说明材料：

  1. 静态图（matplotlib）
       results/graph/knowledge_graph.png            —— spring 布局，按实体类型着色
       results/graph/knowledge_graph_circular.png   —— 环形布局，便于看清社区聚集
  2. 交互式 HTML（零 Python 依赖，浏览器打开即可用）
       results/graph/knowledge_graph.html
       · 首选 ECharts（CDN）；若离线导致 CDN 不可用，自动降级为内置的
         Canvas 力导向渲染器（纯手写，无任何外部依赖），保证一定能打开
       · 支持：节点拖拽、滚轮缩放、画布平移、按实体类型着色与过滤、
         搜索高亮、点击节点查看属性与该实体的全部关系
  3. 抽取思路说明（Markdown，供技术文档与演示视频解说取材）
       results/extraction_examples.md
       · 逐类型给出「原文例句 → 抽取结果」对照
       · 实体归一化/别名消解实例
       · 社区检测与社区摘要实例
       results/graph/graph_data.json  —— 前端数据快照（供 serve.py 复用）

运行：
    python 工单08-GraphRAG金融问答/src/visualize_graph.py
    python .../visualize_graph.py --max-nodes 200 --no-static   # 只出 HTML
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core.graph_rag import FIN_ENTITY_TYPES, FIN_RELATION_TYPES   # noqa: E402

from build_graph import EXTRACT_CACHE, GRAPH_PATH                     # noqa: E402
from prepare_corpus import (                                          # noqa: E402
    CORPUS_CHUNKS, REPORT_DIR, RESULTS_DIR, load_chunks, load_graph,
)

HTML_PATH = REPORT_DIR / "knowledge_graph.html"
DATA_JSON = REPORT_DIR / "graph_data.json"
PNG_PATH = REPORT_DIR / "knowledge_graph.png"
PNG_CIRCULAR = REPORT_DIR / "knowledge_graph_circular.png"
EXAMPLES_MD = RESULTS_DIR / "extraction_examples.md"

# 实体类型的固定配色（与 matplotlib 静态图保持一致的观感）
TYPE_COLORS = {
    "公司": "#2563eb", "机构": "#7c3aed", "人物": "#db2777",
    "金融产品": "#059669", "业务板块": "#0891b2", "财务指标": "#ea580c",
    "风险类型": "#dc2626", "行业": "#65a30d", "地区": "#0d9488",
    "监管政策": "#9333ea", "时间": "#64748b", "其他": "#9ca3af",
}


# ---------------------------------------------------------------------------
# 一、前端数据载荷
# ---------------------------------------------------------------------------
def build_payload(kg, max_nodes: int = 300) -> dict:
    """把 KnowledgeGraph 转成前端可直接消费的 {nodes, links, categories, stats}。"""
    ents = sorted(kg.entities.values(), key=lambda e: -e.degree)
    keep = {e.name for e in ents[:max_nodes]}

    # 类型按出现频次排序，保证图例顺序稳定
    type_order = [t for t, _ in
                  Counter(e.type for e in kg.entities.values()).most_common()]
    for t in FIN_ENTITY_TYPES:                       # 补齐未出现的类型，图例完整
        if t not in type_order:
            type_order.append(t)
    cat_index = {t: i for i, t in enumerate(type_order)}

    nodes = [{
        "id": e.name, "name": e.name, "type": e.type,
        "category": cat_index.get(e.type, len(type_order) - 1),
        "desc": (e.description or "")[:300],
        "degree": e.degree,
        "size": min(10 + e.degree * 1.6, 48),
        "color": TYPE_COLORS.get(e.type, "#9ca3af"),
        "aliases": sorted(e.aliases)[:10],
    } for e in ents[:max_nodes]]

    links = [{
        "source": r.source, "target": r.target, "type": r.type,
        "desc": (r.description or "")[:200], "weight": r.weight,
    } for r in getattr(kg, "relations_merged", [])
        if r.source in keep and r.target in keep]

    # 社区归属（用于「按社区着色」视图）
    comm_of = {}
    for cid, members in kg.communities.items():
        for m in members:
            comm_of[m] = cid
    for n in nodes:
        n["community"] = comm_of.get(n["id"], -1)

    return {
        "nodes": nodes,
        "links": links,
        "categories": type_order,
        "colors": [TYPE_COLORS.get(t, "#9ca3af") for t in type_order],
        "communities": [
            {"id": cid, "size": len(m), "summary":
                (getattr(kg, "community_summaries", {}) or {}).get(cid, "")[:400],
             "members": m[:15]}
            for cid, m in sorted(kg.communities.items(), key=lambda x: -len(x[1]))
        ],
        "stats": kg.stats(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(GRAPH_PATH),
    }


# ---------------------------------------------------------------------------
# 二、交互式 HTML（ECharts 优先 + 手写 Canvas 力导向兜底）
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>金融知识图谱 · Graph RAG（工单08）</title>
<style>
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;font-family:"Microsoft YaHei","PingFang SC",sans-serif;
            background:#0f172a;color:#e2e8f0;overflow:hidden}
  #top{height:56px;display:flex;align-items:center;gap:16px;padding:0 18px;
       background:#1e293b;border-bottom:1px solid #334155}
  #top h1{font-size:16px;margin:0;font-weight:600;white-space:nowrap}
  #top .stat{font-size:12px;color:#94a3b8}
  #top .stat b{color:#38bdf8}
  #main{display:flex;height:calc(100% - 56px)}
  #side{width:280px;flex-shrink:0;background:#1e293b;border-right:1px solid #334155;
        padding:14px;overflow-y:auto;font-size:12.5px}
  #side h3{font-size:12px;color:#94a3b8;margin:16px 0 8px;letter-spacing:.5px}
  #side h3:first-child{margin-top:0}
  #search{width:100%;padding:7px 10px;border-radius:6px;border:1px solid #334155;
          background:#0f172a;color:#e2e8f0;font-size:12.5px;outline:none}
  #search:focus{border-color:#38bdf8}
  .legend{display:flex;align-items:center;gap:7px;padding:4px 6px;border-radius:5px;
          cursor:pointer;user-select:none}
  .legend:hover{background:#273449}
  .legend.off{opacity:.32}
  .dot{width:11px;height:11px;border-radius:50%;flex-shrink:0}
  .legend .cnt{margin-left:auto;color:#64748b;font-size:11px}
  #canvasWrap{flex:1;position:relative;background:radial-gradient(circle at 50% 40%,#16233c,#0b1120)}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas.dragging{cursor:grabbing}
  #hint{position:absolute;left:12px;bottom:12px;font-size:11.5px;color:#64748b;
        background:rgba(15,23,42,.75);padding:6px 10px;border-radius:6px;line-height:1.7}
  #panel{width:320px;flex-shrink:0;background:#1e293b;border-left:1px solid #334155;
         padding:14px;overflow-y:auto;font-size:12.5px;line-height:1.75}
  #panel h3{font-size:13px;margin:0 0 10px;color:#38bdf8}
  #panel .kv{margin-bottom:9px}
  #panel .k{color:#94a3b8;font-size:11.5px}
  #panel .tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;
              background:#334155;color:#cbd5e1;margin:2px 4px 2px 0}
  #panel .rel{padding:6px 8px;border-radius:6px;background:#0f172a;margin-bottom:6px;
              border-left:3px solid #38bdf8}
  #panel .rel .t{color:#38bdf8}
  #panel .rel .d{color:#94a3b8;font-size:11.5px}
  .empty{color:#64748b;font-size:12px}
  #err{position:absolute;top:10px;right:14px;font-size:11.5px;color:#fbbf24;
       background:rgba(15,23,42,.85);padding:5px 10px;border-radius:6px;display:none}
</style>
</head>
<body>
<div id="top">
  <h1>金融知识图谱 · Graph RAG</h1>
  <span class="stat">实体 <b id="s-ent">0</b></span>
  <span class="stat">关系 <b id="s-rel">0</b></span>
  <span class="stat">社区 <b id="s-com">0</b></span>
  <span class="stat">显示 <b id="s-show">0</b> 节点</span>
  <span class="stat" style="margin-left:auto" id="renderer">渲染器：检测中…</span>
</div>
<div id="main">
  <div id="side">
    <h3>搜索实体</h3>
    <input id="search" placeholder="输入名称后回车，如：拨备覆盖率">
    <h3>实体类型（点击过滤）</h3>
    <div id="legend"></div>
    <h3>显示</h3>
    <div class="legend" id="btn-labels"><span class="dot" style="background:#94a3b8"></span>显示标签</div>
  </div>
  <div id="canvasWrap">
    <canvas id="cv"></canvas>
    <div id="err">ECharts 加载失败，已切换到内置渲染器</div>
    <div id="hint">拖拽节点可调整位置 ｜ 空白处拖动平移 ｜ 滚轮缩放 ｜ 点击节点查看属性与关系</div>
  </div>
  <div id="panel"><h3>详情</h3><div class="empty">点击左侧图例过滤类型，点击画布中的节点查看实体属性、别名与全部关系；点击连线查看关系描述。</div></div>
</div>
<script id="graph-data" type="application/json">__GRAPH_DATA__</script>
<script>
const GRAPH = JSON.parse(document.getElementById('graph-data').textContent);
const TYPE_COLORS = __TYPE_COLORS__;

// ---------------------------------------------------------------------
// 公共：详情面板
// ---------------------------------------------------------------------
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

function showNode(n){
  const rels = GRAPH.links.filter(l=>l.source===n.id||l.target===n.id);
  let h = '<h3>实体详情</h3>';
  h += '<div class="kv"><div class="k">名称</div><b>'+esc(n.name)+'</b></div>';
  h += '<div class="kv"><div class="k">类型</div><span class="tag" style="background:'+
       (TYPE_COLORS[n.type]||'#64748b')+'">'+esc(n.type)+'</span></div>';
  h += '<div class="kv"><div class="k">度数（连接关系数）</div>'+n.degree+'</div>';
  if(n.community>=0) h += '<div class="kv"><div class="k">所属社区</div>社区 '+n.community+'</div>';
  if(n.aliases&&n.aliases.length)
    h += '<div class="kv"><div class="k">别名</div>'+n.aliases.map(a=>'<span class="tag">'+esc(a)+'</span>').join('')+'</div>';
  h += '<div class="kv"><div class="k">属性描述（由 LLM 从原文抽取并累加）</div>'+(esc(n.desc)||'<span class="empty">无</span>')+'</div>';
  h += '<h3>关系（'+rels.length+'）</h3>';
  h += rels.slice(0,80).map(r=>{
      const out = r.source===n.id;
      const other = out? r.target : r.source;
      return '<div class="rel"><div><b>'+esc(n.name)+'</b> <span class="t">--['+esc(r.type)+']--&gt;</span> <b>'+esc(other)+'</b>'+
             (out?'':' <span class="empty">(入边)</span>')+'</div>'+
             (r.desc?'<div class="d">'+esc(r.desc)+'</div>':'')+'</div>';
    }).join('') || '<div class="empty">该实体暂无关系</div>';
  document.getElementById('panel').innerHTML = h;
}

function showLink(l){
  document.getElementById('panel').innerHTML =
    '<h3>关系详情</h3>'+
    '<div class="kv"><div class="k">关系类型</div><span class="tag">'+esc(l.type)+'</span></div>'+
    '<div class="kv"><div class="k">头实体</div><b>'+esc(l.source)+'</b></div>'+
    '<div class="kv"><div class="k">尾实体</div><b>'+esc(l.target)+'</b></div>'+
    '<div class="kv"><div class="k">合并权重（同向关系出现次数）</div>'+l.weight+'</div>'+
    '<div class="kv"><div class="k">原文依据</div>'+(esc(l.desc)||'<span class="empty">无</span>')+'</div>';
}

// ---------------------------------------------------------------------
// 公共：图例 / 过滤
// ---------------------------------------------------------------------
const hidden = new Set();
let showLabels = true;
function buildLegend(){
  const box = document.getElementById('legend');
  const cnt = {};
  GRAPH.nodes.forEach(n=>cnt[n.type]=(cnt[n.type]||0)+1);
  box.innerHTML='';
  Object.keys(TYPE_COLORS).forEach(t=>{
    if(!cnt[t]) return;
    const d=document.createElement('div');
    d.className='legend'; d.dataset.type=t;
    d.innerHTML='<span class="dot" style="background:'+TYPE_COLORS[t]+'"></span>'+
                esc(t)+'<span class="cnt">'+cnt[t]+'</span>';
    d.onclick=()=>{ if(hidden.has(t)) hidden.delete(t); else hidden.add(t);
                    d.classList.toggle('off'); applyFilterAll(); };
    box.appendChild(d);
  });
}
document.getElementById('btn-labels').onclick = function(){
  showLabels = !showLabels; this.classList.toggle('off'); applyFilterAll();
};

function visibleNodes(){ return GRAPH.nodes.filter(n=>!hidden.has(n.type)); }
function visibleLinks(){ const ok=new Set(visibleNodes().map(n=>n.id));
  return GRAPH.links.filter(l=>ok.has(l.source)&&ok.has(l.target)); }
function updateCount(){ document.getElementById('s-show').textContent = visibleNodes().length; }
// 类型过滤统一入口：两种渲染器共用同一份过滤状态
function applyFilterAll(){
  if(useECharts){ drawECharts(); }
  else if(sg){ sg.applyFilter(new Set(visibleNodes().map(n=>n.id))); updateCount(); }
  else { updateCount(); }
}

// ---------------------------------------------------------------------
// 渲染器一：ECharts（首选，交互最完整）
// ---------------------------------------------------------------------
let chart=null, useECharts=false;
function initECharts(){
  chart = echarts.init(document.getElementById('ec'), null, {renderer:'canvas'});
  chart.on('click', p=>{
    if(p.dataType==='edge'){
      const l=GRAPH.links.find(x=>x.source===p.data.source&&x.target===p.data.target);
      if(l) showLink(l);
    } else if(p.dataType==='node'){
      const n=GRAPH.nodes.find(x=>x.id===p.data.id); if(n) showNode(n);
    }
  });
  return chart;
}
function drawECharts(){
  const nodes=visibleNodes(), links=visibleLinks();
  chart.setOption({
    backgroundColor:'transparent',
    tooltip:{show:true,backgroundColor:'#1e293b',borderColor:'#334155',
             textStyle:{color:'#e2e8f0',fontSize:12},
             formatter:p=>{
               if(p.dataType==='edge') return '关系：'+p.data.value;
               const d=p.data; return '<b>'+d.name+'</b><br/>类型：'+d.type+
                 (d.desc?'<br/>'+String(d.desc).slice(0,80):'');
             }},
    series:[{
      type:'graph', layout:'force', roam:true, draggable:true,
      animationDurationUpdate:300, focusNodeAdjacency:true,
      data:nodes.map(n=>({id:n.id,name:n.name,value:n.desc,
        symbolSize:n.size, category:GRAPH.categories.indexOf(n.type),
        itemStyle:{color:TYPE_COLORS[n.type]||'#9ca3af'},
        label:{show:showLabels,position:'right',fontSize:10,color:'#cbd5e1'}})),
      links:links.map(l=>({source:l.source,target:l.target,value:l.type,
        lineStyle:{color:'#475569',width:Math.min(1+l.weight,3),curveness:0.08}})),
      categories:GRAPH.categories.map(t=>({name:t,itemStyle:{color:TYPE_COLORS[t]||'#9ca3af'}})),
      edgeSymbol:['none','arrow'], edgeSymbolSize:6,
      force:{repulsion:420,edgeLength:[70,190],gravity:0.09,friction:0.12},
      labelLayout:{hideOverlap:true},
      emphasis:{focus:'adjacency',lineStyle:{color:'#38bdf8',width:3}},
    }],
  }, true);
  updateCount();
}
window.addEventListener('resize',()=>{ if(useECharts&&chart) chart.resize(); });

// ---------------------------------------------------------------------
// 渲染器二：手写 Canvas 力导向（离线兜底，零依赖）
// ---------------------------------------------------------------------
let sg=null;
function SimpleGraph(canvas, data, onNode, onLink){
  const ctx=canvas.getContext('2d');
  const W=()=>canvas.clientWidth, H=()=>canvas.clientHeight;
  const nodes=data.nodes.map(n=>({...n,x:W()/2+(Math.random()-.5)*W()*.7,
                                  y:H()/2+(Math.random()-.5)*H()*.7,vx:0,vy:0}));
  const idx={}; nodes.forEach((n,i)=>idx[n.id]=i);
  const links=data.links.map(l=>({...l,s:idx[l.source],t:idx[l.target]}))
                        .filter(l=>l.s!=null&&l.t!=null);
  let scale=1, ox=0, oy=0, drag=null, panning=null, dragMoved=false;

  function resize(){ canvas.width=W()*devicePixelRatio; canvas.height=H()*devicePixelRatio; }
  function step(){
    for(let i=0;i<nodes.length;i++){
      const a=nodes[i];
      for(let j=i+1;j<nodes.length;j++){
        const b=nodes[j]; let dx=b.x-a.x, dy=b.y-a.y;
        let d2=dx*dx+dy*dy||.01, d=Math.sqrt(d2);
        if(d>420) continue;
        const f=2600/d2, ux=dx/d, uy=dy/d;
        a.vx-=f*ux; a.vy-=f*uy; b.vx+=f*ux; b.vy+=f*uy;
      }
    }
    links.forEach(l=>{
      const a=nodes[l.s], b=nodes[l.t];
      let dx=b.x-a.x, dy=b.y-a.y, d=Math.sqrt(dx*dx+dy*dy)||.01;
      const f=(d-130)*0.0055, ux=dx/d, uy=dy/d;
      a.vx+=f*ux; a.vy+=f*uy; b.vx-=f*ux; b.vy-=f*uy;
    });
    nodes.forEach(n=>{
      n.vx+=(W()/2-n.x)*0.0016; n.vy+=(H()/2-n.y)*0.0016;
      n.vx*=0.82; n.vy*=0.82;
      if(n!==drag){ n.x+=n.vx; n.y+=n.vy; }
    });
  }
  function draw(){
    const dpr=devicePixelRatio||1;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W(),H());
    ctx.save(); ctx.translate(ox,oy); ctx.scale(scale,scale);
    ctx.lineWidth=Math.max(.6/scale,.4); ctx.strokeStyle='rgba(148,163,184,.42)';
    links.forEach(l=>{
      const a=nodes[l.s], b=nodes[l.t];
      if(!a||!b || a.hidden || b.hidden) return;      // 被过滤的节点不画
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
      const mx=(a.x+b.x)/2,my=(a.y+b.y)/2;
      if(scale>0.75){
        ctx.save(); ctx.fillStyle='rgba(148,163,184,.75)'; ctx.font=(9/scale)+'px sans-serif';
        ctx.textAlign='center'; ctx.fillText(l.type,mx,my-2); ctx.restore();
      }
    });
    nodes.forEach(n=>{
      if(n.hidden) return;
      if(n===drag){ ctx.beginPath(); ctx.arc(n.x,n.y,n.size*1.6,0,7);
        ctx.strokeStyle='#38bdf8'; ctx.lineWidth=2/scale; ctx.stroke(); }
      ctx.beginPath(); ctx.arc(n.x,n.y,n.size*.62,0,7);
      ctx.fillStyle=n.color||'#9ca3af'; ctx.fill();
      ctx.strokeStyle='rgba(255,255,255,.55)'; ctx.lineWidth=.8/scale; ctx.stroke();
      if(showLabels && scale>0.55){
        ctx.font=(10.5/scale)+'px "Microsoft YaHei",sans-serif';
        ctx.fillStyle='#cbd5e1'; ctx.textAlign='center';
        ctx.fillText(n.name,n.x,n.y-n.size*.62-4/scale);
      }
    });
    ctx.restore();
  }
  function loop(){ step(); draw(); requestAnimationFrame(loop); }
  function toWorld(e){
    const r=canvas.getBoundingClientRect();
    return {x:(e.clientX-r.left-ox)/scale, y:(e.clientY-r.top-oy)/scale};
  }
  function pick(p){
    let best=null,bd=1e9;
    nodes.forEach(n=>{ if(n.hidden) return;
      const d=Math.hypot(n.x-p.x,n.y-p.y);
      if(d<n.size*.75&&d<bd){bd=d;best=n;} });
    return best;
  }
  canvas.addEventListener('mousedown',e=>{
    const p=toWorld(e); const n=pick(p); dragMoved=false;
    if(n){ drag=n; canvas.classList.add('dragging'); }
    else { panning={x:e.clientX-ox,y:e.clientY-oy}; canvas.classList.add('dragging'); }
  });
  window.addEventListener('mousemove',e=>{
    if(dragMoved) return;
    if(drag){ const p=toWorld(e); drag.x=p.x; drag.y=p.y; drag.vx=drag.vy=0; dragMoved=true; }
    else if(panning){ ox=e.clientX-panning.x; oy=e.clientY-panning.y; dragMoved=true; }
  });
  window.addEventListener('mouseup',e=>{
    if(drag&&!dragMoved){ const n=drag; onNode&&onNode(n); }
    else if(!drag&&!dragMoved){ const p=toWorld(e); const n=pick(p);
      if(n) onNode&&onNode(n); else onLink&&onLink(); }
    drag=null; panning=null; canvas.classList.remove('dragging');
    setTimeout(()=>dragMoved=false,0);
  });
  canvas.addEventListener('wheel',e=>{ e.preventDefault();
    const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
    const k=e.deltaY<0?1.12:1/1.12;
    ox=mx-(mx-ox)*k; oy=my-(my-oy)*k; scale*=k;
  },{passive:false});
  window.addEventListener('resize',resize);
  resize(); loop();
  return {draw, nodes, applyFilter(keep){
      nodes.forEach(n=>{ n.hidden=!keep.has(n.id); }); }};
}
function initSimple(reason){
  document.getElementById('err').style.display='block';
  document.getElementById('renderer').textContent='渲染器：内置 Canvas 力导向（'+reason+'）';
  document.getElementById('canvasWrap').innerHTML=
    '<canvas id="cv"></canvas>'+
    '<div id="hint">拖拽节点 ｜ 空白处拖动平移 ｜ 滚轮缩放 ｜ 点击节点查看属性与关系</div>';
  const cv=document.getElementById('cv');
  sg=new SimpleGraph(cv,{nodes:GRAPH.nodes,links:GRAPH.links},showNode,()=>{});
  buildLegend();           // 图例点击会调用 applyFilterAll()，进而调用 sg.applyFilter
  updateCount();
}

// ---------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------
document.getElementById('search').addEventListener('keydown',e=>{
  if(e.key!=='Enter') return;
  const q=e.target.value.trim(); if(!q) return;
  const hit=GRAPH.nodes.find(n=>n.name.includes(q))||
            GRAPH.nodes.find(n=>(n.desc||'').includes(q));
  if(!hit){ alert('未找到包含「'+q+'」的实体'); return; }
  showNode(hit);
  if(useECharts){
    chart.dispatchAction({type:'highlight',seriesIndex:0,name:hit.id});
    chart.dispatchAction({type:'showTip',seriesIndex:0,name:hit.id});
  } else if(sg){
    const n=sg.nodes.find(x=>x.id===hit.id); if(n){ n.vx=n.vy=0; }
    alert('已定位到实体：'+hit.name+'（内置渲染器下请自行拖动画布查看，详情见右侧面板）');
  }
});

document.getElementById('s-ent').textContent = GRAPH.stats.n_entities;
document.getElementById('s-rel').textContent = GRAPH.stats.n_relations;
document.getElementById('s-com').textContent = (GRAPH.communities||[]).length;

(function boot(){
  const s=document.createElement('script');
  s.src='https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js';
  s.onload=function(){
    if(typeof echarts==='undefined'){ initSimple('CDN 无响应'); return; }
    useECharts=true;
    document.getElementById('renderer').textContent='渲染器：ECharts 5（CDN）';
    document.getElementById('canvasWrap').innerHTML=
      '<div id="ec" style="width:100%;height:100%"></div>'+
      '<div id="hint">拖拽节点 ｜ 滚轮缩放 ｜ 点击节点看属性 ｜ 点击连线看关系描述</div>';
    initECharts(); buildLegend(); drawECharts();
  };
  s.onerror=function(){ initSimple('离线环境'); };
  document.head.appendChild(s);
  // 8 秒兜底：CDN 卡住时也要能看
  setTimeout(()=>{ if(!useECharts && !sg) initSimple('CDN 超时'); }, 8000);
})();
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    """把数据载荷注入 HTML 模板（供本脚本与 serve.py 共用）。"""
    data = json.dumps(payload, ensure_ascii=False)
    # 防止 JSON 里的 </script> 提前闭合脚本标签
    data = data.replace("</", "<\\/")
    return (_HTML_TEMPLATE
            .replace("__GRAPH_DATA__", data)
            .replace("__TYPE_COLORS__", json.dumps(TYPE_COLORS, ensure_ascii=False)))


# ---------------------------------------------------------------------------
# 三、「实体/关系创建思路」说明材料
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])")
_SECTION_PREFIX = re.compile(r"^【[^】]*】")


def _sentences(text: str) -> list[str]:
    text = _SECTION_PREFIX.sub("", (text or "").strip())
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 8]


def _find_evidence(text: str, names: list[str], max_len: int = 220) -> str:
    """在原文块中找包含指定实体名的句子，作为「原文依据」。"""
    for sent in _sentences(text):
        for nm in names:
            if nm and nm in sent:
                return sent[:max_len]
    for nm in names:                      # 退化：名字在块里但被句子切分打断
        if nm and nm in (text or ""):
            i = text.index(nm)
            return text[max(0, i - 60):i + 120].strip()
    return (text or "").strip()[:max_len]


def _load_extractions(cache_path: Path) -> list[dict]:
    out = []
    if not Path(cache_path).exists():
        return out
    for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def build_extraction_examples(kg, chunks: list, cache_path: Path = EXTRACT_CACHE,
                              per_type: int = 3) -> str:
    """
    生成「实体关系创建思路」说明：
    逐类型给出「原文 → 抽取结果」对照，外加归一化与社区摘要实例。
    """
    chunk_text = {c.chunk_id: c.text for c in chunks}
    chunk_doc = {c.chunk_id: c.doc for c in chunks}
    extractions = _load_extractions(cache_path)

    # 实体类型 -> 示例；关系类型 -> 示例
    ent_examples: dict[str, list[dict]] = defaultdict(list)
    rel_examples: dict[str, list[dict]] = defaultdict(list)

    for rec in extractions:
        cid = rec.get("chunk_id", "")
        text = chunk_text.get(cid, "")
        data = rec.get("data") or {}
        for e in data.get("entities") or []:
            t = e.get("type", "其他")
            if len(ent_examples[t]) < per_type:
                nm = e.get("name", "")
                ent_examples[t].append({
                    "entity": e, "evidence": _find_evidence(text, [nm]),
                    "doc": chunk_doc.get(cid, ""), "chunk_id": cid,
                })
        for r in data.get("relations") or []:
            t = r.get("type", "相关")
            if len(rel_examples[t]) < per_type:
                rel_examples[t].append({
                    "relation": r,
                    "evidence": _find_evidence(
                        text, [r.get("source", ""), r.get("target", "")]),
                    "doc": chunk_doc.get(cid, ""), "chunk_id": cid,
                })

    lines: list[str] = []
    lines.append("# 知识图谱「实体 / 关系创建思路」说明材料\n")
    lines.append("> 工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答  \n"
                 "> 本文件由 `src/visualize_graph.py` 自动生成，是演示视频中"
                 "「实体、关系创建思路讲解」环节的取材依据。\n")

    # ---- 1. 总体思路 ----
    lines.append("## 一、总体思路：三步把 PDF 变成图谱\n")
    lines.append("```")
    lines.append("年报 PDF ──解析/分块──▶ 文本块 ──LLM 抽取──▶ 三元组 ──归一化/合并──▶ 知识图谱")
    lines.append("  ① rag_core.pdf_parse     ② GraphExtractor       ③ KnowledgeGraph.add_extraction")
    lines.append("     PyMuPDF + pdfplumber     optimized prompt       别名消解 + 同向合并加权")
    lines.append("```\n")
    lines.append("关键点：**实体和关系不是预先定义的字典，而是由 LLM 逐块从原文中读出来的**；"
                 "代码提供的是「类型体系（Schema）」，用来约束抽取结果的形状，"
                 "使不同块、不同文档抽出来的实体能够落到同一套类型上，从而可以被合并、检索和统计。\n")

    # ---- 2. 类型体系 ----
    lines.append("## 二、为什么用这套类型体系\n")
    lines.append("### 实体类型（抽取时 prompt 里给出的封闭枚举）\n")
    lines.append("| 类型 | 定义 | 设计理由 |")
    lines.append("| --- | --- | --- |")
    _why_ent = {
        "公司": "金融问答的主体，绝大多数问题都要先定位到某家公司",
        "机构": "监管机构/交易所是「受监管于」关系的必备端点，也是行业分析的关键",
        "人物": "董事长致辞、高管变动类问题的答案来源",
        "金融产品": "贷款/存款/理财/保险/债券等是业务分析的抓手",
        "业务板块": "零售/对公/资金同业等条线，支撑「业务结构优化」类分析题",
        "财务指标": "数值型问题的答案载体，要求带报告期（如「营业收入(2019年)」）",
        "风险类型": "信用/市场/操作/流动性风险，支撑风险类与周期类问题",
        "行业": "信贷投向分析的维度（制造业、房地产业…）",
        "地区": "区域集中度分析的维度（回答「潜在压力-区域集中度」）",
        "监管政策": "监管口径变化是金融分析的重要背景",
        "时间": "锚定报告期，避免不同年份数值串台",
    }
    for t, desc in FIN_ENTITY_TYPES.items():
        lines.append(f"| {t} | {desc} | {_why_ent.get(t, '—')} |")
    lines.append("")
    lines.append("### 关系类型\n")
    lines.append("| 关系 | 定义 | 设计理由 |")
    lines.append("| --- | --- | --- |")
    _why_rel = {
        "子公司": "集团型年报的基本结构关系",
        "控股股东": "股权结构类问题的直接答案",
        "参股": "区分控制与非控制关联方（工单07 同类考点）",
        "投资": "金融资产投向分析",
        "任职": "「法定代表人/董事长是谁」类问题的答案路径",
        "提供产品服务": "业务关系，支撑业务结构分析",
        "发行": "金融产品发行关系",
        "披露指标": "把「文档」与「指标」连起来，回答「哪份年报披露了某指标」",
        "指标数值": "**数值型问题的核心关系**，指标 → 取值",
        "面临风险": "风险类问题的答案路径",
        "受监管于": "监管关系",
        "属于行业": "行业分析维度",
        "位于地区": "区域分析维度",
        "同比变化": "**增长率类问题的核心关系**",
        "提及": "兜底关系，保证弱相关信息不丢",
    }
    for t, desc in FIN_RELATION_TYPES.items():
        lines.append(f"| {t} | {desc} | {_why_rel.get(t, '—')} |")
    lines.append("")

    # ---- 3. 抽取 Prompt 设计 ----
    lines.append("## 三、抽取 Prompt 的设计（optimized 档位）\n")
    lines.append("`rag_core/graph_rag.py` 的 `PromptProfile(optimized)` 与本工单使用的要点：\n")
    lines.append("1. **封闭类型枚举**：prompt 中直接列出上表全部实体/关系类型，"
                 "要求 `type` 必须从列表中选择——这是后续能做类型统计与类型过滤可视化的前提。")
    lines.append("2. **指代还原**：明确要求「本公司/该行/本行」要还原成文本中出现过的具体全称，"
                 "否则 9 份年报里的「本行」会全部塌缩成一个错误实体。")
    lines.append("3. **指标带报告期**：要求财务指标实体写成「营业收入(2019年)」，"
                 "并把具体数值写进 description，解决不同年份数值混淆的问题。")
    lines.append("4. **只抽明确陈述**：关系必须能在原文找到依据，description 要引用原文关键短语，"
                 "从源头抑制幻觉，也便于人工核对（见下方每一类示例的「原文依据」）。")
    lines.append("5. **二次补漏（gleaning, 1 轮）**：针对只出现一次的数字、顿号并列实体、"
                 "表格行列关系再做一次「查漏」，缓解长文档漏抽。")
    lines.append("6. **上限保护**：单块最多 30 个实体 / 45 条关系，避免个别表格块撑爆输出长度。\n")

    # ---- 4. 实体抽取示例 ----
    lines.append("## 四、实体抽取实例（原文 → 抽取结果）\n")
    lines.append("> 每一类实体都是从年报原文里「读」出来的，下面给出对照。"
                 "原文列取自 chunk 正文，抽取结果列为 LLM 返回的 JSON 片段。\n")
    lines.append("| 实体类型 | 原文例句（截取） | 抽取出的实体 |")
    lines.append("| --- | --- | --- |")
    for t in list(FIN_ENTITY_TYPES) + ["其他"]:
        for ex in ent_examples.get(t, [])[:2]:
            e = ex["entity"]
            lines.append("| {} | {} | **{}**（type={}）<br/>description：{} |".format(
                t,
                _md_cell(ex["evidence"], 150),
                _md_cell(e.get("name", ""), 40),
                _md_cell(e.get("type", ""), 12),
                _md_cell(e.get("description", ""), 60) or "—"))
    lines.append("")

    # ---- 5. 关系抽取示例 ----
    lines.append("## 五、关系抽取实例（原文 → 三元组）\n")
    lines.append("| 关系类型 | 原文依据（截取） | 抽取出的三元组 |")
    lines.append("| --- | --- | --- |")
    for t in list(FIN_RELATION_TYPES) + ["相关"]:
        for ex in rel_examples.get(t, [])[:2]:
            r = ex["relation"]
            lines.append("| {} | {} | **{}** —[{}]→ **{}**<br/>依据：{} |".format(
                t,
                _md_cell(ex["evidence"], 150),
                _md_cell(r.get("source", ""), 40),
                _md_cell(r.get("type", ""), 14),
                _md_cell(r.get("target", ""), 40),
                _md_cell(r.get("description", ""), 60) or "—"))
    lines.append("")

    # ---- 6. 归一化 ----
    lines.append("## 六、实体归一化 / 别名消解\n")
    lines.append("同一个对象在年报中会有多种写法（全称、简称、指代），"
                 "如果不去重，图谱会被拆成很多孤立节点，检索也会漏。"
                 "`KnowledgeGraph.add_extraction` 的处理链是：\n")
    lines.append("```")
    lines.append("原始实体名 ──_norm_name()──▶ 去空白/去首尾标点/全角括号转半角/剔除「本公司」等纯指代")
    lines.append("           ──_resolve_alias()──▶ 命中别名表 或 与已有规范名做包含匹配(≥6字) ──▶ 规范名")
    lines.append("           ──add_extraction()──▶ 描述累加、来源 chunk 记录、别名登记")
    lines.append("```\n")
    alias_ents = [e for e in kg.entities.values() if e.aliases]
    alias_ents.sort(key=lambda e: -e.degree)
    if alias_ents:
        lines.append("图谱中实际发生的别名消解（节选）：\n")
        lines.append("| 规范实体 | 被消解的别名 | 类型 | 度数 |")
        lines.append("| --- | --- | --- | --- |")
        for e in alias_ents[:12]:
            lines.append(f"| {_md_cell(e.name, 40)} | "
                         f"{_md_cell('、'.join(sorted(e.aliases)[:6]), 80)} | "
                         f"{e.type} | {e.degree} |")
    else:
        lines.append("（当前图谱中未出现别名合并——说明各文档的实体名写法已经足够规范；"
                     "这一层在实体名写法不统一的语料上作用更明显。）")
    lines.append("")
    lines.append("关系层面的去重：同一对实体之间的多条关系会按 `(source, target)` 合并，"
                 "`weight` 记录出现次数、`description` 累加原文依据，"
                 "因此**边的粗细本身就是一个「该关系在年报中被反复强调」的置信度信号**。\n")

    # ---- 7. 社区 ----
    lines.append("## 七、社区检测与社区摘要（Graph RAG 的全局视角）\n")
    lines.append("图谱建好后，用 Louvain 算法按连接稠密度把实体切成若干主题簇，"
                 "再用 LLM 为每个簇写一段 150 字摘要。这一步是传统 RAG 没有的："
                 "它把「散落在 9 份年报里、彼此相关的实体」聚成一个可被整体检索的主题。\n")
    lines.append(f"当前图谱共 **{len(kg.communities)}** 个社区，摘要示例：\n")
    summaries = getattr(kg, "community_summaries", {}) or {}
    for cid, members in sorted(kg.communities.items(),
                               key=lambda x: -len(x[1]))[:8]:
        s = summaries.get(cid, "（未生成摘要）")
        lines.append(f"**社区 {cid}**（{len(members)} 个实体）："
                     f"{'、'.join(members[:8])} …\n")
        lines.append(f"> {s}\n")

    # ---- 8. 检索示例 ----
    lines.append("## 八、这些实体关系如何被检索用到\n")
    lines.append("```")
    lines.append("用户问题 ──link_entities()──▶ 命中种子实体（含别名匹配）")
    lines.append("        ──local_search()──▶ 沿关系扩展 2 跳，收集子图（实体属性 + 关系描述）→ 局部事实题")
    lines.append("        ──global_search()─▶ 在社区摘要上做相关性排序 → 全局归纳题")
    lines.append("        ──两者拼接──────▶ 送入 LLM 生成答案（mode=hybrid）")
    lines.append("```\n")
    lines.append("例如问「平安银行的拨备覆盖率是多少」，`link_entities` 会命中"
                 "「平安银行股份有限公司」，再沿 `指标数值` / `披露指标` 关系"
                 "两跳内就能拿到「拨备覆盖率(2019年)」这个实体及其 description 中的数值，"
                 "并附带原文依据——这正是数值型问题答得准的原因。\n")

    return "\n".join(lines)


def _md_cell(text: str, limit: int) -> str:
    """表格单元格清洗：截断 + 转义竖线与换行。"""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = s.replace("|", "\\|")
    return s[:limit] + ("…" if len(s) > limit else "")


# ---------------------------------------------------------------------------
# 四、主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="工单08 知识图谱可视化")
    ap.add_argument("--graph", type=Path, default=GRAPH_PATH, help="kg.json 路径")
    ap.add_argument("--max-nodes", type=int, default=300, help="交互图最多显示多少节点")
    ap.add_argument("--static-nodes", type=int, default=120, help="静态图最多显示多少节点")
    ap.add_argument("--no-static", action="store_true", help="跳过 matplotlib 静态图")
    ap.add_argument("--no-examples", action="store_true", help="跳过抽取示例文档")
    args = ap.parse_args()

    kg = load_graph(args.graph)
    print(f"载入图谱：{len(kg.entities)} 实体 / "
          f"{len(getattr(kg, 'relations_merged', []))} 关系 / "
          f"{len(kg.communities)} 社区")

    # ---- 1. 静态图 ----
    if not args.no_static:
        try:
            p1 = kg.visualize(PNG_PATH, max_nodes=args.static_nodes, layout="spring")
            print(f"静态图（spring）：{p1}")
            p2 = kg.visualize(PNG_CIRCULAR, max_nodes=min(args.static_nodes, 80),
                              layout="circular")
            print(f"静态图（circular）：{p2}")
        except Exception as e:                       # matplotlib / 中文字体缺失时不影响主产物
            print(f"[warn] 静态图生成失败（不影响交互式 HTML）：{e}")

    # ---- 2. 交互式 HTML ----
    payload = build_payload(kg, max_nodes=args.max_nodes)
    HTML_PATH.write_text(build_html(payload), encoding="utf-8")
    DATA_JSON.write_text(json.dumps(payload, ensure_ascii=False),
                         encoding="utf-8")
    print(f"交互式 HTML：{HTML_PATH}")
    print(f"前端数据快照：{DATA_JSON}")

    # ---- 3. 抽取思路说明 ----
    if not args.no_examples:
        try:
            chunks = load_chunks(CORPUS_CHUNKS)
        except FileNotFoundError:
            chunks = []
            print("[warn] 未找到分块结果，抽取示例将缺少原文对照")
        md = build_extraction_examples(kg, chunks)
        EXAMPLES_MD.write_text(md, encoding="utf-8")
        print(f"抽取思路说明：{EXAMPLES_MD}")

    print("\n完成。用浏览器打开 knowledge_graph.html 即可交互查看（支持拖拽/过滤/点击详情）。")


if __name__ == "__main__":
    main()
