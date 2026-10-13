from __future__ import annotations
import argparse,json,re
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",type=Path,default=Path("vlm_eval.json"));a=p.parse_args();rows=[]
 for i,line in enumerate(Path(a.input).read_text(encoding="utf-8").splitlines()):
  if not line.strip():continue
  x=json.loads(line);question=str(x.get("question",x.get("query","")));answer=str(x.get("answer",x.get("response","")));terms=set(re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]",answer.lower()));rows.append({"id":str(x.get("id",i)),"has_question":bool(question),"has_answer":bool(answer),"answer_terms":len(terms),"image":x.get("image") or x.get("image_path")})
 r={"count":len(rows),"valid_records":sum(x["has_question"] and x["has_answer"] for x in rows),"rows":rows};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
