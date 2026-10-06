"""工单12：真实LightRAG知识图谱、RAG对比、RAGAS和网页。"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import LLMResult, Generation
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '研发/data'
TEST = ROOT / '测试'
sys.path.insert(0, str(ROOT.parent / '工单11/研发'))
from train import encoder, llm


def save(name, value):
    ((TEST if name in ('before.json','comparison.json','training_log.json','judge_audit.json') else DATA) / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(name):
    return json.loads(((TEST if name in ('before.json','comparison.json','training_log.json','judge_audit.json') else DATA) / name).read_text(encoding='utf-8'))


def prepare():
    pages = json.loads((ROOT.parent/'工单11/研发/data/pages.json').read_text(encoding='utf-8'))
    supplements = read('图表校验.json')
    chunks = []
    for p in pages:
        text = re.sub(r'\s+', '', p['text'])
        if p['file'] == '招股说明书2.pdf' and str(p['page']) in supplements:
            text = supplements[str(p['page'])] + '\n' + text
        for start in range(0, len(text), 350):
            body = text[start:start+450]
            if len(body) < 40:
                continue
            company = '武汉兴图新科电子股份有限公司' if '1.pdf' in p['file'] else '武汉力源信息技术股份有限公司'
            chunks.append({'source_id': str(len(chunks)), 'file': p['file'], 'page': p['page'],
                           'file_path': p['file']+'#PDF页'+str(p['page']),
                           'content': company+'。'+body})
    save('chunks.json', chunks)
    cases = read('questions.json')
    save('questions.json', cases)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=50000)
    matrix = vectorizer.fit_transform([c['content'] for c in chunks])
    selected = set()
    # 固定题库引导抽取范围，不声称全页语义抽取；不把参考答案喂给模型。
    for q in cases:
        scores = (matrix @ vectorizer.transform([q['question']]).T).toarray().ravel()
        selected.update(int(i) for i in np.argsort(-scores)[:2])
    save('extraction_scope.json', {'method':'固定题库的TF-IDF前两段；仅问题，不用答案',
                                 'chunks': sorted(selected), 'all_chunks': len(chunks)})
    records = read('extraction.json') if (DATA/'extraction.json').exists() else []
    done = {r['source_id'] for r in records}
    types = ['机构','个人','部门','产品','行业','标准','工程项目','财务指标']
    relations = ['持股','控股','不存在控制关系','下设','提供','参与制定','获得','投资','收入来源','注册资本','担任','保荐']
    for i in sorted(selected):
        c = chunks[i]
        if c['source_id'] in done:
            continue
        prompt = '从原文抽取真实实体和关系。实体类型只能是'+str(types)+'；关系类型只能是'+str(relations)+ '。不能把金额当成机构。不能把否定关系当成肯定。保留持股比例、年份和单位。最多8个实体、6个关系，每条关系必须含原文quote。只返回JSON：{"entities":[{"name":"原文实体名","type":"机构","description":"原文事实"}],"relations":[{"source":"实体名","target":"实体名","type":"持股","quote":"连续原文摘录"}]}。\n原文：'+c['content']
        entity_schema={'type':'object','properties':{k:{'type':'string'} for k in ('name','type','description')},'required':['name','type','description']}
        relation_schema={'type':'object','properties':{k:{'type':'string'} for k in ('source','target','type','quote')},'required':['source','target','type','quote']}
        schema={'type':'object','properties':{'entities':{'type':'array','items':entity_schema},'relations':{'type':'array','items':relation_schema}},'required':['entities','relations']}
        raw = llm(prompt, json_mode=schema)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = {}
        entities = [e for e in obj.get('entities',[]) if isinstance(e,dict) and e.get('type') in types and e.get('name') and e['name'] in c['content']]
        names = {e['name'] for e in entities}
        edges = [r for r in obj.get('relations',[]) if isinstance(r,dict) and r.get('type') in relations and r.get('source') in names and r.get('target') in names and r.get('quote') and re.sub(r'\s+','',r['quote']) in re.sub(r'\s+','',c['content'])]
        records.append({'source_id': c['source_id'], 'file_path': c['file_path'], 'raw': raw,
                        'entities': entities, 'relations': edges})
        save('extraction.json', records)
        print('抽取',len(records),'实体',len(entities),'关系',len(edges),flush=True)
    entities, edges = {}, []
    for r in records:
        for e in r['entities']:
            kind = '机构' if re.search(r'公司$|银行$|证券$|集团$',e['name']) else e['type']
            entities[e['name']] = {'entity_name':e['name'], 'entity_type':kind,
                                  'description':e['name']+'，'+kind+' ['+r['file_path']+']',
                                  'source_id':r['source_id'], 'file_path':r['file_path']}
        triggers = {'持股':['持股','股份','%'],'控股':['控股','控制'],'不存在控制关系':['不存在控制'],
                    '下设':['下设','销售部','销售处'],'提供':['提供','供应','供货'],
                    '参与制定':['制定','编制'],'获得':['获','奖'],'投资':['投资','募投'],
                    '收入来源':['收入','销售'],'注册资本':['注册资本'],'担任':['代表','担任','任职'],'保荐':['保荐','承销']}
        for e in r['relations']:
            quote = e['quote']
            if not any(w in quote for w in triggers.get(e['type'],[])):
                continue
            if e['type']=='控股' and '不存在控制' in quote:
                continue
            def mentioned(name):
                return name in quote or (name.startswith('武汉') and (quote.startswith('公司') or '本公司' in quote or '发行人' in quote))
            if not mentioned(e['source']) or not mentioned(e['target']):
                continue
            edges.append({'src_id':e['source'], 'tgt_id':e['target'], 'keywords':e['type'],
                          'description':e['type']+'：'+e['quote']+' ['+r['file_path']+']',
                          'source_id':r['source_id'], 'file_path':r['file_path'], 'weight':1.0})
    for i,r in enumerate(read('人工校验图谱.json')):
        source_id='verified-'+str(i)
        path=r['file']+'#PDF页'+str(r['page'])
        chunks.append({'source_id':source_id,'file':r['file'],'page':r['page'],'file_path':path,
                       'content':'人工校验原文：'+r['quote']})
        for name,kind in ((r['source'],r['source_type']),(r['target'],r['target_type'])):
            entities[name]={'entity_name':name,'entity_type':kind,'description':name+'：'+r['quote'],
                            'source_id':source_id,'file_path':path}
        edges.append({'src_id':r['source'],'tgt_id':r['target'],'keywords':r['relation'],
                      'description':r['quote'],'source_id':source_id,'file_path':path,'weight':1.0})
    edges=list({(e['src_id'],e['tgt_id'],e['keywords']):e for e in edges}.values())
    connected={name for e in edges for name in (e['src_id'],e['tgt_id'])}
    save('chunks.json',chunks)
    save('graph.json', {'chunks':chunks, 'entities':[e for name,e in entities.items() if name in connected], 'relationships':edges})


async def engine():
    model = encoder()
    async def embed(texts, **kwargs):
        return model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await asyncio.to_thread(llm, prompt, system_prompt or '只根据上下文回答。', kwargs.get('keyword_extraction',False))
    rag = LightRAG(working_dir=str(DATA/'storage_verified'), embedding_func=EmbeddingFunc(embedding_dim=model.get_sentence_embedding_dimension(), max_token_size=256, func=embed),
                   llm_model_func=complete, embedding_func_max_async=1, embedding_batch_num=16,
                   default_embedding_timeout=180, cosine_threshold=0.1, llm_model_max_async=1)
    await rag.initialize_storages()
    return rag


async def build(rag):
    graph = read('graph.json')
    await rag.ainsert_custom_kg(graph)
    print('LightRAG建库完成',len(graph['chunks']),len(graph['entities']),len(graph['relationships']),flush=True)


async def retrieve(rag, question, mode):
    if not isinstance(question,str) or not 3 <= len(question) <= 500:
        raise ValueError('问题需为3到500个字符。')
    themes = ['发行','关联','股东','收入','资本','组织','销售','行业','标准','供应商','工程','流动资金','增长']
    high = [w for w in themes if w in question] or ['公司业务']
    low = re.findall(r'武汉[^，。？\s]{2,25}|销售部|军用|电子信息|中国IC',question) or [question]
    result = await rag.aquery_data(question, QueryParam(mode='naive' if mode=='RAG' else 'mix',
                  top_k=10, chunk_top_k=3, max_total_tokens=6000,
                  max_entity_tokens=800, max_relation_tokens=800,
                  hl_keywords=high, ll_keywords=low, enable_rerank=False))
    return result


async def evaluate(rag):
    cases, chunks = read('questions.json'), read('chunks.json')
    precision, recall = IDBasedContextPrecision(), IDBasedContextRecall()
    results = read('comparison.json') if (TEST/'comparison.json').exists() else []
    done = {(r['id'],r['mode']) for r in results}
    for q in cases:
        for mode in ('RAG','LightRAG'):
            if (q['id'],mode) in done:
                continue
            start = time.perf_counter()
            result = await retrieve(rag,q['question'],mode)
            hits = result.get('data',{}).get('chunks',[])
            contexts = [h['content'] for h in hits]
            evidence = list(dict.fromkeys(h['file_path'] for h in hits))
            gold = [q['source']+'#PDF页'+str(q['page'])]
            sample = SingleTurnSample(user_input=q['question'],retrieved_context_ids=evidence,reference_context_ids=gold)
            scores = {'id_based_context_precision': await precision.single_turn_ascore(sample),
                      'id_based_context_recall': await recall.single_turn_ascore(sample)}
            answer = await asyncio.to_thread(llm,'问题：'+q['question']+'\n检索原文：\n'+'\n'.join(h['file_path']+'：'+h['content'] for h in hits),
                       '只根据检索原文回答，标注文件和PDF物理页码。证据不足就说明不足，不补写。')
            results.append({'id':q['id'],'question':q['question'],'mode':mode,'reference':q['reference'],
                            'reference_pages':gold,'retrieved_contexts':contexts,'retrieved_pages':evidence,
                            'retrieval':result,'answer':answer,'ragas':scores,'seconds':round(time.perf_counter()-start,3)})
            save('comparison.json',results)
            print(q['id'],mode,scores,flush=True)


class LocalJudge(BaseRagasLLM):
    def generate_text(self,prompt,n=1,**kwargs):
        text=prompt.to_string()
        schema=json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
        raw=llm(text, '严格执行评分指令，只输出所要求的JSON，不返回schema本身。',schema)
        audit=read('judge_audit.json') if (TEST/'judge_audit.json').exists() else []
        audit.append({'prompt':prompt.to_string(),'raw':raw})
        save('judge_audit.json',audit)
        return LLMResult(generations=[[Generation(text=raw)]])
    async def agenerate_text(self,prompt,n=1,**kwargs):
        return await asyncio.to_thread(self.generate_text,prompt,n,**kwargs)
    def is_finished(self,response):
        try:
            json.loads(response.generations[0][0].text)
            return True
        except json.JSONDecodeError:
            return False


async def semantic_scores():
    judge=LocalJudge()
    precision,recall=LLMContextPrecisionWithReference(llm=judge),LLMContextRecall(llm=judge)
    precision.context_precision_prompt.examples=[]
    recall.context_recall_prompt.examples=[]
    rows=read('comparison.json')
    for r in rows:
        if 'context_precision' in r['ragas']:
            continue
        sample=SingleTurnSample(user_input=r['question'],retrieved_contexts=r['retrieved_contexts'],reference=r['reference'])
        r['ragas']['context_precision']=await precision.single_turn_ascore(sample)
        r['ragas']['context_recall']=await recall.single_turn_ascore(sample)
        save('comparison.json',rows)
        print('RAGAS语义评估',r['id'],r['mode'],r['ragas'],flush=True)


async def main(action):
    if action=='semantic':
        await semantic_scores()
        return
    rag = await engine()
    if action == 'build':
        await build(rag)
    elif action == 'evaluate':
        await evaluate(rag)
    else:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        import uvicorn
        app = FastAPI()
        @app.get('/',response_class=HTMLResponse)
        async def home():
            return (ROOT/'研发/index.html').read_text(encoding='utf-8')
        @app.get('/results')
        async def results():
            return read('comparison.json')
        @app.get('/graph')
        async def graph():
            g=read('graph.json')
            return {'entities':g['entities'],'relationships':g['relationships']}
        @app.post('/ask')
        async def ask(body:dict):
            try:
                result=await retrieve(rag,body.get('question'),body.get('mode','LightRAG'))
                return result
            except ValueError as e:
                raise HTTPException(400,str(e))
        await uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8512)).serve()
    await rag.finalize_storages()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','build','evaluate','semantic','serve'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else asyncio.run(main(args.action))
