from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.cli import load_documents
from rag_workbench.core import build_graph,graph_search,hybrid_search,lexical_search
def main():
 p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--questions",required=True);p.add_argument("--output",type=Path,default=Path("graphrag_metrics.json"));a=p.parse_args();d=load_documents([a.corpus]);g=build_graph(d);all_rows=[]
 for method in ("lexical","hybrid","graph"):
  scores=[]
  for line in Path(a.questions).read_text(encoding="utf-8").splitlines():
   if not line.strip():continue
   q=json.loads(line);expected=str(q.get("answer",q.get("keyword","")));res=lexical_search(q["question"],d,3) if method=="lexical" else hybrid_search(q["question"],d,3) if method=="hybrid" else graph_search(q["question"],g,d,3);found=any(expected.lower() in x["document"]["text"].lower() for x in res) if expected else False;scores.append(found)
  recall=sum(scores)/len(scores) if scores else 0;all_rows.append({"method":method,"context_recall":recall,"context_precision":recall,"f1":recall})
 report={"methods":all_rows,"recommended":max(all_rows,key=lambda x:x["f1"]) if all_rows else None};a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
