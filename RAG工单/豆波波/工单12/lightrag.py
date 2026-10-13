from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.core import build_graph,chunk_text,graph_search,hybrid_search,read_text
def load(path):
 data=json.loads(Path(path).read_text(encoding="utf-8"));from rag_workbench.core import Document;return [Document(**x) for x in data["documents"]],data["graph"]
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True);i=sub.add_parser("index");i.add_argument("--input",required=True);i.add_argument("--index",required=True);q=sub.add_parser("query");q.add_argument("--index",required=True);q.add_argument("--query",required=True);a=p.parse_args()
 if a.cmd=="index":
  d=chunk_text(read_text(a.input),a.input);g=build_graph(d);Path(a.index).write_text(json.dumps({"documents":[x.to_dict() for x in d],"graph":g},ensure_ascii=False),encoding="utf-8");print(f"indexed={len(d)}")
 else:
  d,g=load(a.index);local=hybrid_search(a.query,d,3);global_rows=graph_search(a.query,g,d,3);print(json.dumps({"query":a.query,"local":local,"global":global_rows},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
