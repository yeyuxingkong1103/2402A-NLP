"""工单编号：人工智能NLP-RAG项目-LightRAG优化。修正上下文预算后真实复测。"""
import json
import os
import re
import sys
import time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'研发'))
for k,v in {'LLM_PROVIDER':'ollama','LLM_BASE_URL':'http://127.0.0.1:11434','LLM_MODEL':'Qwen3.5:4B','LLM_TIMEOUT':'120'}.items():os.environ.setdefault(k,v)
from app.core import KnowledgeBase
from app.light import query_sync
def norm(s):return re.sub(r'[\s,，*]','',s).lower()
kb=KnowledgeBase();gold=json.loads((BASE/'测试/gold.json').read_text(encoding='utf-8'))
report=json.loads((BASE/'测试/实测结果.json').read_text(encoding='utf-8'))
samples=[];answers=[]
previous_answers=json.loads((BASE/'测试/LightRAG生成对比.json').read_text(encoding='utf-8')) if (BASE/'测试/LightRAG生成对比.json').exists() else []
selected=[int(i) for i in os.getenv('LIGHT_TEST_IDS','').split(',') if i] if os.getenv('LIGHT_TEST_IDS') else None
for g in gold:
    if selected and g['id'] not in selected:continue
    started=time.perf_counter();raw=query_sync(g['question']);hits=raw['mapped_sources']
    ranks=[i+1 for i,h in enumerate(hits) if h['document']==g['document'] and h['pdf_page'] in g['pages']]
    samples.append({'id':g['id'],'question':g['question'],'reference':g['answer'],'hits':hits,'official_response':raw['official_result'],
        'hit_at_5':bool(ranks),'mrr_at_5':1/min(ranks) if ranks else 0,'labelled_page_precision_at_5':len(ranks)/len(hits) if hits else 0,
        'fact_coverage_proxy':sum(norm(f) in norm(' '.join(h['text'] for h in hits)) for f in g['facts'])/len(g['facts']),
        'retrieval_ms':(time.perf_counter()-started)*1000})
    result=kb.ask(g['question'],strategy='light',use_cache=False)
    answers.append({'id':g['id'],'question':g['question'],'reference':g['answer'],'result':result})
    visible=[a for a in previous_answers if a['id'] not in {x['id'] for x in answers}]+answers if selected else answers
    (BASE/'测试/LightRAG生成对比.json').write_text(json.dumps(visible,ensure_ascii=False,indent=2),encoding='utf-8')
    print(g['id'],'hits',len(hits),'generation',result['generation'],flush=True)
if selected:
    samples=[s for s in report['strategies']['light']['samples'] if s['id'] not in selected]+samples
    answers=[a for a in previous_answers if a['id'] not in selected]+answers
    (BASE/'测试/LightRAG生成对比.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
report['strategies']['light']={'hit_at_5':sum(s['hit_at_5'] for s in samples)/len(samples),
    'mrr_at_5':sum(s['mrr_at_5'] for s in samples)/len(samples),'fact_coverage_proxy':sum(s['fact_coverage_proxy'] for s in samples)/len(samples),'samples':samples}
report['light_budget_fix']='实体与关系最多1000/1200 token，总上下文预算10000；此前3500预算让图谱占满上下文，丢失文本块。此文件保留最终实际复测。'
(BASE/'测试/实测结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
