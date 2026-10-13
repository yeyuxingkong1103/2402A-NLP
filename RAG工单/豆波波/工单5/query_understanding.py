from __future__ import annotations
import argparse,json,re

SYNONYMS={"维护":"检修 保养 维修","故障":"异常 问题","费用":"成本 价格","风险":"隐患 风险"}
def understand(query,history=""):
    q=" ".join((history+" "+query).split()); normalized=re.sub(r"[？?！!。]","",q).strip(); expanded=normalized
    for key,value in SYNONYMS.items():
        if key in normalized: expanded += " " + value
    parts=[p.strip() for p in re.split(r"(?:并且|以及|和|，|,|;|；)",query) if p.strip()]
    route="graph" if any(x in normalized for x in ("关系","影响","供应链","风险")) else "hybrid"
    return {"original":query,"normalized":normalized,"expanded":expanded,"sub_queries":parts or [normalized],"route":route,"history_used":bool(history)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--query",required=True);p.add_argument("--history",default="");a=p.parse_args();print(json.dumps(understand(a.query,a.history),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
