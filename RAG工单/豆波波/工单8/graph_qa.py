from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.cli import load_documents
from rag_workbench.core import build_graph,graph_search
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--query",required=True);p.add_argument("--output",type=Path,default=Path("graph_answer.json"));a=p.parse_args();d=load_documents([a.input]);g=build_graph(d);r={"query":a.query,"graph":{"entities":len(g["entities"]),"edges":len(g["edges"])},"evidence":graph_search(a.query,g,d)};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
