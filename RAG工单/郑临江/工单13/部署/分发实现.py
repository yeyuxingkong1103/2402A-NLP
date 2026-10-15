"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。仅分发源码快照。"""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for i in range(2,14):
    p=ROOT/f'工单{i}'
    if i!=13:
        shutil.copy2(ROOT/'工单13/研发/app/core.py',p/'研发/app/core.py')
    config=json.loads((p/'研发/project.json').read_text(encoding='utf-8'))
    requirements=p/'研发/requirements.txt'
    requirements.write_text(requirements.read_text(encoding='utf-8')+'\nrank-bm25>=0.2.2\nnetworkx>=3\n',encoding='utf-8')
    for file in (p/'研发/app').glob('*.py'):
        text=file.read_text(encoding='utf-8')
        file.write_text('# 工单编号：'+config['ticket']+'\n'+text,encoding='utf-8')
    server=p/'研发/app/server.py'
    text=server.read_text(encoding='utf-8')
    text=text.replace('data.get("mode", "rag"))','data.get("mode", "rag"), strategy=data.get("strategy", "hybrid"), session_id=data.get("session_id", "browser"))')
    text=text.replace('if path == "/api/documents":','if path == "/api/graph":\n                return self.send_json(state.kb.graph)\n            if path == "/api/documents":')
    text=text.replace('if path == "/api/ask":','if path == "/api/settings":\n                    state.kb.query_settings.update({k:v for k,v in data.items() if k in {"strategy","reranker","weight"}})\n                    state.kb.cache.clear()\n                    return self.send_json(state.kb.query_settings)\n                if path == "/api/ask":')
    text=text.replace('return self.send_json({"message": "反馈已记录"})','for h in state.kb.chunks:\n                        if str(data.get("source_id", "")) == h["id"]:\n                            state.kb.rerank_feedback[h["id"]] += 1 if data["rating"] == "up" else -1\n                    return self.send_json({"message": "反馈已记录"})')
    server.write_text(text,encoding='utf-8')
    html=p/'研发/static/index.html';text=html.read_text(encoding='utf-8')
    text=text.replace('招股文件 · 问答工作台',f'工单{i} · '+config['title'])
    text=text.replace('向文档提问','向本项目文档提问')
    text=text.replace('<button id="ask" class="primary">','<select id="strategy"><option value="hybrid">混合检索</option><option value="dense">语义向量</option><option value="fulltext">全文 BM25</option>'+('<option value="graph">GraphRAG</option><option value="graph_optimized">图谱优化</option>' if config['graph'] else '')+('<option value="light">官方LightRAG</option>' if i==12 else '')+'</select><select id="reranker"><option value="tfidf">TF-IDF重排</option><option value="llm">LLM重排</option><option value="feedback">反馈重排</option></select><button id="ask" class="primary">')
    text=text.replace("question:q,mode:$('mode').value","question:q,mode:$('mode').value,strategy:$('strategy').value,session_id:sessionStorage.getItem('sid')||(sessionStorage.setItem('sid',crypto.randomUUID()),sessionStorage.getItem('sid'))")
    text=text.replace("const r=await api('/api/ask'","await api('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reranker:$('reranker').value})});const r=await api('/api/ask'")
    text=text.replace("mode:$('mode').value})", "mode:$('mode').value})")
    if config['graph']:
        text=text.replace('<footer>','<section class="panel" style="max-width:1195px;margin:20px auto"><h2>本项目知识图谱</h2><button id="graphButton">查看实体与关系</button><svg id="graphSvg" viewBox="0 0 1100 420" style="width:100%;height:420px"></svg><p id="graphSummary"></p></section><footer>')
        text=text.replace('refresh();\n</script>', '''refresh();
$('graphButton').onclick=async()=>{const g=await api('/api/graph');const svg=$('graphSvg');svg.replaceChildren();const nodes=g.nodes.slice(0,28),positions=new Map();nodes.forEach((n,i)=>positions.set(n.id,{x:100+(i%7)*150,y:60+Math.floor(i/7)*90}));const make=(tag,attrs)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e};g.edges.slice(0,150).forEach(e=>{const a=positions.get(e.source),b=positions.get(e.target);if(a&&b)svg.append(make('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#b9c9ce'}))});nodes.forEach(n=>{const p=positions.get(n.id);svg.append(make('circle',{cx:p.x,cy:p.y,r:13,fill:n.type==='Organization'?'#186d66':'#7298aa'}));const t=make('text',{x:p.x,y:p.y+30,'text-anchor':'middle','font-size':12});t.textContent=n.name.length>14?n.name.slice(0,14)+'…':n.name;svg.append(t)});$('graphSummary').textContent='实体 '+g.nodes.length+' 个，关系 '+g.edges.length+' 条；显示前28个实体。关系原文与来源见研发/data/index/graph.json';};
</script>''')
    html.write_text(text,encoding='utf-8')
    print('工单'+str(i)+'实现快照完成')
