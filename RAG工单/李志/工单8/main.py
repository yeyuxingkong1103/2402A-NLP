import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, graph_expand, graph_from_documents, load_documents, save

parser = argparse.ArgumentParser(description="工单8：Graph+RAG 金融问答")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="公司的销售组织和主营业务有什么关系？")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents(); graph = graph_from_documents(docs)
results = graph_expand(args.query, docs, graph)
save(Path(__file__).parent / "outputs/graph.json", graph)
nodes = graph["nodes"][:50]; edges = graph["edges"][:100]
html = """<!doctype html><meta charset='utf-8'><title>GraphRAG 知识图谱</title><style>body{font:14px sans-serif;background:#08111f;color:#e8f0ff}svg{width:100%;height:90vh}.edge{stroke:#49647e}.node{fill:#41d1a6}.label{fill:white;font-size:12px}</style><h1>金融知识图谱</h1><svg id='g'></svg><script>const nodes=__NODES__,edges=__EDGES__,svg=document.querySelector('#g'),W=innerWidth,H=innerHeight*.85;nodes.forEach((n,i)=>{n.x=W/2+Math.cos(i*2.4)*Math.min(W,H)*.38*(.2+i/nodes.length);n.y=H/2+Math.sin(i*2.4)*Math.min(W,H)*.38*(.2+i/nodes.length)});edges.forEach(e=>{let a=nodes.find(n=>n.id==e.source),b=nodes.find(n=>n.id==e.target);if(a&&b)svg.innerHTML+=`<line class=edge x1=${a.x} y1=${a.y} x2=${b.x} y2=${b.y}/>`});nodes.forEach(n=>svg.innerHTML+=`<circle class=node cx=${n.x} cy=${n.y} r=${4+Math.min(n.weight,8)}/><text class=label x=${n.x+8} y=${n.y}>${n.id}</text>`)</script>"""
html = html.replace("__NODES__", json.dumps(nodes, ensure_ascii=False)).replace("__EDGES__", json.dumps(edges, ensure_ascii=False))
(Path(__file__).parent / "outputs/graph.html").write_text(html, encoding="utf-8")
payload = {"question": args.query, "answer": answer(args.query, results), "results": results,
           "graph_stats": {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}, "visualization": "outputs/graph.html"}
save(Path(__file__).parent / "outputs/answer.json", payload); print(json.dumps(payload, ensure_ascii=False, indent=2))
