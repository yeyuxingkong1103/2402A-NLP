from __future__ import annotations
import argparse,json,re
from pathlib import Path
def tokens(s):return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]",s.lower()))
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",type=Path,default=Path("embedding_eval.json"));a=p.parse_args();rows=[]
 for line in Path(a.data).read_text(encoding="utf-8").splitlines():
  if not line.strip():continue
  x=json.loads(line);q=tokens(x["query"]);pos=tokens(x["positive"]);neg=tokens(x["negative"]);ps=len(q&pos)/max(len(q|pos),1);ns=len(q&neg)/max(len(q|neg),1);rows.append({"query":x["query"],"positive_score":ps,"negative_score":ns,"correct_order":ps>ns})
 r={"count":len(rows),"pairwise_accuracy":sum(x["correct_order"] for x in rows)/len(rows) if rows else 0,"rows":rows};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
