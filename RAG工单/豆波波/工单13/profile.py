from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.cli import load_documents
from rag_workbench.core import hybrid_search
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--query",required=True);p.add_argument("--iterations",type=int,default=10);p.add_argument("--output",type=Path,default=Path("profile.json"));a=p.parse_args();durs=[]
 d=load_documents([a.input]);durs=[]
 for _ in range(max(a.iterations,1)):
  t=time.perf_counter();hybrid_search(a.query,d,3);durs.append((time.perf_counter()-t)*1000)
 s=sorted(durs);p50=s[len(s)//2];p95=s[min(len(s)-1,int(len(s)*.95))];r={"iterations":len(durs),"p50_ms":p50,"p95_ms":p95,"max_ms":max(durs),"recommendation":"检查分块/向量服务" if p95>100 else "当前离线检索无明显瓶颈"};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
