"""工单13：分阶段计时、缓存优化、HTTP压测与性能页面。"""
import argparse
import asyncio
import copy
import cProfile
import hashlib
import io
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'研发'
TEST=ROOT/'测试'
URL='http://127.0.0.1:8513'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
RESULTS=OrderedDict()
VECTORS=OrderedDict()
EMBED=[]
RECENT=[]

def save(name,value):
    (TEST/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read(name):
    return json.loads((TEST/name).read_text(encoding='utf-8'))

def http(path,body=None):
    req=urllib.request.Request(URL+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=240) as r:return json.load(r)

def keywords(q):
    high=[w for w in ['发行','关联','股东','收入','资本','组织','销售','行业','标准','供应商','工程','流动资金','增长'] if w in q] or ['公司业务']
    low=re.findall(r'武汉[^，。？\s]{2,25}|销售部|军用|电子信息|中国IC',q) or [q]
    return high,low

async def engines():
    import torch
    from sentence_transformers import SentenceTransformer
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    torch.set_num_threads(4)
    cache=Path.home()/'.cache/huggingface/hub/models--moka-ai--m3e-small/snapshots'
    revision='44c696631b2a8c200220aaaad5f987f096e986df'
    local=cache/revision
    model=SentenceTransformer(str(local) if (local/'modules.json').exists() else 'moka-ai/m3e-small',revision=revision,device='cpu')
    model.max_seq_length=256
    model.encode(['启动预热'],normalize_embeddings=True)
    lock=asyncio.Lock()
    async def baseline(texts,**kwargs):
        start=time.perf_counter()
        value=model.encode(texts,batch_size=16,normalize_embeddings=True)
        EMBED.append({'optimized':False,'texts':len(texts),'ms':(time.perf_counter()-start)*1000})
        return value
    async def optimized(texts,**kwargs):
        async with lock:
            missing=list(dict.fromkeys(t for t in texts if t not in VECTORS))
            start=time.perf_counter()
            if missing:
                values=await asyncio.to_thread(model.encode,missing,batch_size=16,normalize_embeddings=True)
                for t,v in zip(missing,values):VECTORS[t]=v
            value=np.array([VECTORS[t] for t in texts])
            while len(VECTORS)>128:VECTORS.popitem(last=False)
            EMBED.append({'optimized':True,'texts':len(texts),'computed':len(missing),'ms':(time.perf_counter()-start)*1000})
            return value
    rags=[]
    async def complete(*args,**kwargs):
        raise RuntimeError('检索已传入显式关键词，不应调用LLM；完整生成使用answer接口')
    for func in (baseline,optimized):
        rag=LightRAG(working_dir=str(DATA/'storage'),embedding_func=EmbeddingFunc(embedding_dim=512,max_token_size=256,func=func),
                     llm_model_func=complete,embedding_func_max_async=1,embedding_batch_num=16,default_embedding_timeout=180,cosine_threshold=0.1)
        await rag.initialize_storages()
        rags.append(rag)
    return rags

def generate(question,context,optimized):
    prompt='只根据原文简短回答，标注文件和PDF页码；证据不足就说不足。<｜User｜>问题：'+question+'\n原文：\n'+context+'<｜Assistant｜><think>\n</think>\n'
    body={'model':'deepseek-r1:7b','prompt':'<｜begin▁of▁sentence｜>'+prompt,'raw':True,'stream':False,'keep_alive':'10m',
          'options':{'temperature':0,'num_ctx':8192,'num_predict':192 if optimized else 1200}}
    req=urllib.request.Request('http://127.0.0.1:11434/api/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=240) as r:return json.load(r)

async def run(rags,body,answer=False):
    from lightrag import QueryParam
    q=body.get('question')
    if not isinstance(q,str) or not 3<=len(q.strip())<=500:raise ValueError('问题需为3到500个字符')
    q=q.strip();mode=body.get('mode','LightRAG')
    if mode not in ('RAG','LightRAG'):raise ValueError('检索方式必须为RAG或LightRAG')
    optimized=body.get('optimized',True);key=(mode,q)
    start=time.perf_counter();started_at=datetime.now(timezone.utc).isoformat();stages=[]
    def mark(name,t):
        end=time.perf_counter()
        stages.append({'stage':name,'start_ms':(t-start)*1000,'end_ms':(end-start)*1000,'ms':(end-t)*1000})
    t=time.perf_counter();high,low=keywords(q);mark('query',t)
    hit=bool(optimized and body.get('use_cache',True) and key in RESULTS)
    t=time.perf_counter()
    if hit:
        result=copy.deepcopy(RESULTS[key]);RESULTS.move_to_end(key)
    else:
        result=await rags[int(bool(optimized))].aquery_data(q,QueryParam(mode='mix' if mode=='LightRAG' else 'naive',top_k=10,chunk_top_k=3,
                       max_total_tokens=6000,max_entity_tokens=800,max_relation_tokens=800,hl_keywords=high,ll_keywords=low,enable_rerank=False))
        if optimized and body.get('use_cache',True):
            RESULTS[key]=copy.deepcopy(result)
            while len(RESULTS)>128:RESULTS.popitem(last=False)
    mark('retrieval',t)
    t=time.perf_counter();chunks=result.get('data',{}).get('chunks',[])
    context='\n'.join(c['file_path']+'：'+c['content'] for c in chunks);mark('context',t)
    retrieval_ms=(time.perf_counter()-start)*1000
    t=time.perf_counter();generated=await asyncio.to_thread(generate,q,context,optimized) if answer else {};mark('generation',t)
    t=time.perf_counter()
    payload={'request_id':uuid.uuid4().hex,'question':q,'mode':mode,'optimized':optimized,'cache_hit':hit,'chunks':chunks,
             'answer':generated.get('response'),'output_tokens':generated.get('eval_count',0),'input_tokens':generated.get('prompt_eval_count',0),
             'done_reason':generated.get('done_reason'),'prefill_ms':generated.get('prompt_eval_duration',0)/1e6,'decode_ms':generated.get('eval_duration',0)/1e6,
             'load_ms':generated.get('load_duration',0)/1e6,'retrieval_ms':retrieval_ms,'context_chars':len(context)}
    json.dumps(payload,ensure_ascii=False);mark('format',t)
    payload.update(stages=stages,total_ms=(time.perf_counter()-start)*1000,started_at=started_at,ended_at=datetime.now(timezone.utc).isoformat())
    RECENT.append({k:v for k,v in payload.items() if k not in ('chunks','answer')})
    if len(RECENT)>128:RECENT.pop(0)
    with (TEST/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(RECENT[-1],ensure_ascii=False)+'\n')
    return payload

def resources():
    p=psutil.Process();net=psutil.net_io_counters();disk=psutil.disk_io_counters()
    return {'time':datetime.now(timezone.utc).isoformat(),'cpu_percent':psutil.cpu_percent(),'rss_mb':p.memory_info().rss/1e6,
            'net_bytes':net.bytes_sent+net.bytes_recv,'disk_bytes':disk.read_bytes+disk.write_bytes,'embedding_calls':len(EMBED)}

async def serve():
    from fastapi import FastAPI,HTTPException
    from fastapi.responses import HTMLResponse
    import uvicorn
    start=time.perf_counter();rags=await engines();boot=time.perf_counter()-start
    app=FastAPI()
    @app.get('/',response_class=HTMLResponse)
    async def home():return (DATA/'index.html').read_text(encoding='utf-8')
    @app.get('/health')
    async def health():return {'ready':True,'startup_seconds':boot,'vectors':2365,'index_sha256':hashlib.sha256((DATA/'storage/vdb_chunks.json').read_bytes()).hexdigest()}
    @app.get('/stats')
    async def stats():return {'recent':RECENT,'resources':resources(),'embedding':EMBED[-128:],'response_cache':len(RESULTS),'vector_cache':len(VECTORS)}
    @app.get('/report')
    async def report():return read('benchmark.json') if (TEST/'benchmark.json').exists() else {'groups':[]}
    @app.post('/reset')
    async def reset():RESULTS.clear();VECTORS.clear();return {'cleared':True}
    @app.post('/retrieve')
    async def retrieve(body:dict):
        try:return await run(rags,body)
        except ValueError as e:raise HTTPException(400,str(e))
    @app.post('/answer')
    async def answer(body:dict):return await run(rags,body,True)
    @app.post('/profile')
    async def profile(body:dict):
        profiler=cProfile.Profile();profiler.enable();result=await run(rags,body);profiler.disable()
        profiler.dump_stats(str(TEST/'baseline.prof'))
        import pstats
        s=io.StringIO();pstats.Stats(profiler,stream=s).sort_stats('cumulative').print_stats(35)
        (TEST/'profile.txt').write_text(s.getvalue(),encoding='utf-8');return result
    await uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8513)).serve()
    for rag in rags:await rag.finalize_storages()

def benchmark():
    cases=json.loads((DATA/'questions.json').read_text(encoding='utf-8'))
    report={'health':http('/health'),'groups':[],'quality':[],'answers':[]}
    def request(body):
        start=time.perf_counter()
        try:return {'response':http('/retrieve',body)}
        except (urllib.error.URLError,TimeoutError) as e:return {'error':str(e)}
    # 时间在收到HTTP结果以后记录，包括服务器排队和响应传输。
    def timed(body):
        start=time.perf_counter();row=request(body);row['client_ms']=(time.perf_counter()-start)*1000;return row
    report['first_query']=timed({'question':cases[0]['question'],'optimized':False,'use_cache':False})
    http('/profile',{'question':cases[4]['question'],'optimized':False,'use_cache':False})
    for concurrency in (1,4,8):
        for optimized in (False,True):
            http('/reset',{})
            jobs=[{'question':q['question'],'optimized':optimized,'use_cache':False} for q in cases]*2
            before=http('/stats')['resources'];start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(timed,jobs))
            wall=time.perf_counter()-start;times=[r['client_ms'] for r in rows]
            group={'concurrency':concurrency,'optimized':optimized,'response_cache_enabled':False,'requests':len(rows),'errors':sum('error' in r for r in rows),
                   'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),
                   'p99_ms':float(np.percentile(times,99)),'max_ms':max(times),'rps':len(rows)/wall,'resources_before':before,'resources_after':http('/stats')['resources'],'rows':rows}
            report['groups'].append(group);save('benchmark.json',report);print('load',concurrency,optimized,round(group['p95_ms'],2),flush=True)
    for mode in ('RAG','LightRAG'):
        for q in cases:
            bodies=[{'question':q['question'],'mode':mode,'optimized':o,'use_cache':False} for o in (False,True)]
            rows=[timed(b) for b in bodies]
            def signature(row):return [(h['file_path'],h['content']) for h in row['response']['chunks']]
            report['quality'].append({'id':q['id'],'mode':mode,'same_top3':signature(rows[0])==signature(rows[1]),'rows':rows})
    http('/reset',{})
    for q in cases:timed({'question':q['question']})
    with ThreadPoolExecutor(max_workers=8) as pool:report['cache_hits']=list(pool.map(timed,[{'question':q['question']} for q in cases]*2))
    for q in cases[:3]:
        for optimized in (False,True):
            t=time.perf_counter();row=http('/answer',{'question':q['question'],'optimized':optimized,'use_cache':False})
            report['answers'].append({'id':q['id'],'reference':q['reference'],'client_ms':(time.perf_counter()-t)*1000,'response':row})
            save('benchmark.json',report);print('answer',q['id'],optimized,round(row['total_ms']),flush=True)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,creationflags=0x08000000)
    report['gpu_after']=gpu.stdout.strip();save('benchmark.json',report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['serve','benchmark'])
    args=parser.parse_args()
    asyncio.run(serve()) if args.action=='serve' else benchmark()
