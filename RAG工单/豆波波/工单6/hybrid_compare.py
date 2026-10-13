from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.cli import load_documents
from rag_workbench.core import build_graph,graph_search,hybrid_search,lexical_search
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--query",required=True);p.add_argument("--top-k",type=int,default=3);p.add_argument("--output",type=Path,default=Path("comparison.json"));a=p.parse_args();d=load_documents([a.input]);r={"query":a.query,"lexical":lexical_search(a.query,d,a.top_k),"hybrid":hybrid_search(a.query,d,a.top_k),"graph":graph_search(a.query,build_graph(d),d,a.top_k)};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({k:len(v) for k,v in r.items() if isinstance(v,list)},ensure_ascii=False))
if __name__=="__main__":main()
