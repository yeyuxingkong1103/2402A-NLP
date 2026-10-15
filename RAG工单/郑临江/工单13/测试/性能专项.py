"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。分阶段追踪和真实优化对比。"""
import cProfile
import json
import os
import statistics
import sys
import time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'研发'))
for k,v in {'LLM_PROVIDER':'ollama','LLM_BASE_URL':'http://127.0.0.1:11434','LLM_MODEL':'Qwen3.5:4B','LLM_TIMEOUT':'120'}.items():os.environ.setdefault(k,v)
from app.core import KnowledgeBase,understand,rewrite
from app import base
kb=KnowledgeBase();gold=json.loads((BASE/'测试/gold.json').read_text(encoding='utf-8'))[:10]
records=[]
for policy in ['before','after']:
    for g in gold:
        timings={};start=time.perf_counter();info=understand(g['question']);timings['query_analysis_ms']=(time.perf_counter()-start)*1000
        phase=time.perf_counter();_,hits=kb.retrieve(g['question'],k=10 if policy=='before' else 3);timings['retrieval_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        if policy=='before':
            contexts=hits
        else:
            contexts=[];seen=set()
            for h in hits:
                key=(h['document'],h['pdf_page'])
                if key not in seen:contexts.append(h);seen.add(key)
        timings['context_assembly_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();answer=base.call_llm(g['question'],contexts);timings['llm_generation_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();answer=answer.strip();timings['postprocess_ms']=(time.perf_counter()-phase)*1000
        timings['total_ms']=(time.perf_counter()-start)*1000
        records.append({'policy':policy,'id':g['id'],'contexts':len(contexts),'context_chars':sum(len(h['text']) for h in contexts),
            'answer':answer,'timings':timings,'cache':False})
        (BASE/'测试/性能专项结果.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(policy,g['id'],round(timings['total_ms']),flush=True)
summary={p:{'mean_ms':statistics.mean(r['timings']['total_ms'] for r in records if r['policy']==p),
            'max_ms':max(r['timings']['total_ms'] for r in records if r['policy']==p)} for p in ['before','after']}
(BASE/'测试/性能汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
