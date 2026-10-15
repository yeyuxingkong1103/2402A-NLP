"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏。"""
import asyncio,json,statistics,time,urllib.request
from pathlib import Path
import psutil
BASE=Path(__file__).resolve().parents[1]
async def request(q):
    start=time.perf_counter()
    def call():
        req=urllib.request.Request('http://127.0.0.1:8817/ask',json.dumps({'question':q}).encode(),{'Content-Type':'application/json'})
        try:return urllib.request.urlopen(req,timeout=10).status
        except Exception:return 0
    status=await asyncio.to_thread(call);return status,(time.perf_counter()-start)*1000
async def main():
    results={}
    for name,n in [('A',20),('B',10)]:
        started=psutil.Process().memory_info().rss;start=time.perf_counter();rows=[]
        for _ in range(20):rows+=await asyncio.gather(*[request(name+' question') for i in range(n)])
        times=sorted(x[1] for x in rows);results[name]={'requests':len(rows),'errors':sum(x[0]!=200 for x in rows),'p50_ms':statistics.median(times),'p95_ms':times[int(len(times)*.95)-1],'max_ms':max(times),'rss_delta_mb':(psutil.Process().memory_info().rss-started)/1048576,'duration_s':time.perf_counter()-start}
    (BASE/'测试/压测结果.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print(results)
if __name__=='__main__':asyncio.run(main())
