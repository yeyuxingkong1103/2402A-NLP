from __future__ import annotations
import argparse,asyncio,json,time
from pathlib import Path
async def run_one(sem,delay,stats):
 async with sem:
  stats["active"]+=1;stats["peak"]=max(stats["peak"],stats["active"]);t=time.perf_counter();await asyncio.sleep(delay);stats["active"]-=1;stats["latencies"].append((time.perf_counter()-t)*1000)
async def run(n,limit,delay):
 stats={"active":0,"peak":0,"latencies":[]};sem=asyncio.Semaphore(limit);await asyncio.gather(*(run_one(sem,delay,stats) for _ in range(n)));return stats
def main():
 p=argparse.ArgumentParser();p.add_argument("--requests",type=int,default=20);p.add_argument("--limit",type=int,default=4);p.add_argument("--output",type=Path,default=Path("concurrency.json"));a=p.parse_args();s=asyncio.run(run(a.requests,a.limit,.005));r={"requests":a.requests,"limit":a.limit,"peak_active":s["peak"],"completed":len(s["latencies"]),"avg_latency_ms":sum(s["latencies"])/len(s["latencies"])};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
