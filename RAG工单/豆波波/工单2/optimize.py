from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "共享"))
from rag_workbench.core import chunk_text, hybrid_search, read_text

def main():
    p=argparse.ArgumentParser(); p.add_argument("--corpus",required=True); p.add_argument("--questions",required=True); p.add_argument("--output",type=Path,default=Path("optimization.json")); a=p.parse_args()
    text=read_text(a.corpus); questions=[json.loads(x) for x in Path(a.questions).read_text(encoding="utf-8").splitlines() if x.strip()]; rows=[]
    for size in (160,320,640):
      for overlap in (0,40,80):
       for weight in (.3,.6,.8):
        docs=chunk_text(text,a.corpus,size,overlap); hits=[]
        for q in questions:
         expected=str(q.get("answer") or q.get("expected") or q.get("keyword") or ""); result=hybrid_search(q["question"],docs,3,weight); joined=" ".join(r["document"]["text"] for r in result); hits.append(int(not expected or expected.lower() in joined.lower()))
        rows.append({"size":size,"overlap":overlap,"vector_weight":weight,"hit_at_3":sum(hits)/len(hits) if hits else 0})
    best=max(rows,key=lambda x:x["hit_at_3"]); report={"best":best,"trials":rows}; a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(best,ensure_ascii=False))
if __name__ == "__main__": main()
