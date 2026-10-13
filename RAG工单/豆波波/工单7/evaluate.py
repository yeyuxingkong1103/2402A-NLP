from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.cli import load_documents
from rag_workbench.core import hybrid_search
def main():
 p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--questions",required=True);p.add_argument("--output",type=Path,default=Path("evaluation.json"));a=p.parse_args();d=load_documents([a.corpus]);rows=[];lat=[]
 for line in Path(a.questions).read_text(encoding="utf-8").splitlines():
  if not line.strip():continue
  q=json.loads(line);t=time.perf_counter();res=hybrid_search(q["question"],d,5);lat.append(time.perf_counter()-t);expected=str(q.get("answer",q.get("keyword","")));scores=[expected.lower() in x["document"]["text"].lower() for x in res] if expected else [True]
  rank=next((i+1 for i,v in enumerate(scores) if v),None);rows.append({"question":q["question"],"hit_at_5":bool(rank),"reciprocal_rank":1/rank if rank else 0,"rank":rank})
 report={"count":len(rows),"hit_at_5":sum(x["hit_at_5"] for x in rows)/len(rows) if rows else 0,"mrr":sum(x["reciprocal_rank"] for x in rows)/len(rows) if rows else 0,"avg_latency_ms":sum(lat)/len(lat)*1000 if lat else 0,"rows":rows};a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
