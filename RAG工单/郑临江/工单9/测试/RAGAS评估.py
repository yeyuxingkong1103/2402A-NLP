"""工单编号：人工智能NLP-RAG项目-LightRAG优化。真实RAGAS，不用关键词代理冒充指标。"""
import asyncio
import json
import os
import urllib.request
import httpx
from pathlib import Path
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision,ContextRecall,Faithfulness
BASE=Path(__file__).resolve().parents[1]

async def main():
    # 协议适配：仍真实调用本机Ollama，关闭思考并传递结构化输出schema；不返回测试桩答案。
    async def adapt(request):
        data=json.loads(request.content)
        async def invoke():
            def native_call():
                payload={'model':'Qwen3.5:4B','messages':data['messages'],'stream':False,'think':False,
                    'options':{'num_ctx':8192,'num_predict':1400,'temperature':0}}
                fmt=data.get('response_format',{})
                payload['format']=fmt.get('json_schema',{}).get('schema','json')
                req=urllib.request.Request('http://127.0.0.1:11434/api/chat',json.dumps(payload).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
            return await asyncio.to_thread(native_call)
        result=await invoke()
        return httpx.Response(200,json={'id':'local-real-judge','object':'chat.completion','created':0,'model':'Qwen3.5:4B',
            'choices':[{'index':0,'message':{'role':'assistant','content':result['message']['content']},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':result.get('prompt_eval_count',0),'completion_tokens':result.get('eval_count',0),'total_tokens':result.get('prompt_eval_count',0)+result.get('eval_count',0)}},request=request)
    client=AsyncOpenAI(base_url='http://127.0.0.1:11434/v1',api_key='local',timeout=120,
                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(adapt)))
    llm=llm_factory('Qwen3.5:4B',client=client,max_tokens=1400,temperature=0)
    metrics=[ContextPrecision(llm=llm),ContextRecall(llm=llm),Faithfulness(llm=llm)]
    report=json.loads((BASE/'测试/实测结果.json').read_text(encoding='utf-8'))
    selected=os.getenv('RAGAS_STRATEGIES','').split(',') if os.getenv('RAGAS_STRATEGIES') else list(report['strategies'])
    destination=BASE/'测试/RAGAS实测结果.json'
    previous_records=json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else []
    records=[r for r in previous_records if r['strategy'] not in selected]
    snapshot_file=BASE/'测试'/os.getenv('RAGAS_REUSE_SNAPSHOT','__no_snapshot__.json')
    snapshot=json.loads(snapshot_file.read_text(encoding='utf-8')) if snapshot_file.exists() else {}
    for strategy,details in report['strategies'].items():
        if strategy not in selected:continue
        for sample in details['samples']:
            contexts=[h['text'][:1400] for h in sample['hits']]
            old_sample=next((s for s in snapshot.get('strategies',{}).get(strategy,{}).get('samples',[]) if s['id']==sample['id']),None)
            old_result=next((r for r in previous_records if r['strategy']==strategy and r['id']==sample['id']),None)
            if old_sample and old_result and contexts==[h['text'][:1400] for h in old_sample['hits']] and old_sample['reference']==sample['reference']:
                records.append({**old_result,'reuse_note':'评审问题、标准答案及全部上下文完全一致，保留原模型实测；不是补写新分数'})
                destination.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
                print(strategy,sample['id'],'unchanged judge inputs; retained',flush=True)
                continue
            answer=next(a['result']['answer'] for a in report['answers'] if a['id']==sample['id'])
            values={}
            for metric in metrics[:2]:
                try:
                    res=await metric.ascore(user_input=sample['question'],reference=sample['reference'],retrieved_contexts=contexts)
                    values[metric.name]=res.value
                except Exception as exc:values[metric.name]={'error':type(exc).__name__,'detail':str(exc)[:200]}
            records.append({'strategy':strategy,'id':sample['id'],'metrics':values,'judge':'Qwen3.5:4B','framework':'ragas 0.4.3'})
            (BASE/'测试/RAGAS实测结果.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
            print(strategy,sample['id'],values,flush=True)
    await client.close()

if __name__=='__main__':asyncio.run(main())
