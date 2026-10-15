"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏。12小时真实观测，不提前写通过。"""
import argparse,json,time,urllib.request,datetime
from pathlib import Path
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=12);parser.add_argument('--port',type=int,default=8817);parser.add_argument('--output',default='12小时资源观测.json');args=parser.parse_args()
start=time.monotonic();baseline=None;rows=[];duration=args.hours*3600;started_local=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
while time.monotonic()-start<duration:
    try:
        probe_start=time.monotonic()
        probe=urllib.request.Request(f'http://127.0.0.1:{args.port}/ask',json.dumps({'question':'根据文本信息，该静电除尘器的发明人是：','mode':'retrieval','use_cache':False}).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(probe,timeout=10) as response:probe_body=json.load(response)
        probe_ms=(time.monotonic()-probe_start)*1000
        result=json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=5));rss=result['rss_mb'];baseline=baseline or rss
        rows.append({'elapsed_s':time.monotonic()-start,'rss_mb':rss,'growth':(rss-baseline)/baseline,'requests':result['metrics']['requests'],'probe_ms':probe_ms,'probe_has_evidence':bool(probe_body.get('answer'))})
    except Exception as exc:rows.append({'elapsed_s':time.monotonic()-start,'error':str(exc)})
    value={'status':'running','scope':'每30秒真实检索一次的低负载存活/RSS观察；不是完整RAGFlow容器12小时或持续高并发','target_hours':args.hours,'started_local':started_local,'samples':rows}
    (BASE/args.output).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    time.sleep(min(30,max(0,duration-(time.monotonic()-start))))
value['status']='complete';value['passed_rss_growth_under_20pct']=all(r.get('growth',1)<=.2 for r in rows) and not any('error'in r for r in rows)
(BASE/args.output).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
