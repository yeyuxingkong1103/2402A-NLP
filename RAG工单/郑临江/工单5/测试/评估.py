"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。真实逐题评估，不读取答案生成。"""
import cProfile
import concurrent.futures
import json
import os
import pstats
import re
import statistics
import sys
import time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'研发'))
for k,v in {'LLM_PROVIDER':'ollama','LLM_BASE_URL':'http://127.0.0.1:11434','LLM_MODEL':'Qwen3.5:4B','LLM_TIMEOUT':'120'}.items():os.environ.setdefault(k,v)
from app.core import KnowledgeBase,dump_json,CONFIG

def norm(s):return re.sub(r'[\s,，*]', '',s).lower()
def run():
    kb=KnowledgeBase();gold=json.loads((BASE/'测试/gold.json').read_text(encoding='utf-8'))
    strategies=['hybrid']
    if CONFIG['number'] in {2,6,13}:strategies=['baseline','dense','fulltext','hybrid']
    if CONFIG['number'] in {8,9,10}:strategies=['hybrid','graph','graph_optimized']
    if CONFIG['number']==12:strategies=['hybrid','light']
    results={}
    for strategy in strategies:
        samples=[]
        for item in gold:
            started=time.perf_counter()
            if strategy=='light':
                from app.light import query_sync
                context=query_sync(item['question'])
                # 后续正式LightRAG对比独立保存，避免假定返回结构。
                hits=context.get('mapped_sources',[])
                info={'backend':'official_lightRAG'}
            else:info,hits=kb.retrieve(item['question'],strategy=strategy)
            evidence=norm('\n'.join(h['text'] for h in hits))
            ranks=[i+1 for i,h in enumerate(hits) if h['document']==item['document'] and h['pdf_page'] in item['pages']]
            relevant=sum(h['document']==item['document'] and h['pdf_page'] in item['pages'] for h in hits)
            samples.append({'id':item['id'],'question':item['question'],'reference':item['answer'],'hits':hits,
                'hit_at_5':bool(ranks),'mrr_at_5':1/min(ranks) if ranks else 0,
                'labelled_page_precision_at_5':relevant/len(hits) if hits else 0,
                'fact_coverage_proxy':sum(norm(f) in evidence for f in item['facts'])/len(item['facts']),
                'retrieval_ms':(time.perf_counter()-started)*1000})
        results[strategy]={'hit_at_5':statistics.mean(x['hit_at_5'] for x in samples),
            'mrr_at_5':statistics.mean(x['mrr_at_5'] for x in samples),'fact_coverage_proxy':statistics.mean(x['fact_coverage_proxy'] for x in samples),
            'samples':samples}
        print(strategy,'Hit@5',results[strategy]['hit_at_5'],flush=True)
    answers=[]
    for item in gold:
        answer=kb.ask(item['question'],strategy='graph_optimized' if CONFIG['graph'] and CONFIG['number']!=12 else 'hybrid',use_cache=False)
        facts=sum(norm(f) in norm(answer['answer']) for f in item['facts'])/len(item['facts']) if answer['generation']=='llm' else None
        answers.append({'id':item['id'],'question':item['question'],'reference':item['answer'],'result':answer,'answer_fact_coverage_proxy':facts})
        print('工单',CONFIG['number'],'问题',item['id'],answer['generation'],round(answer['elapsed_ms']),flush=True)
    conversations=[]
    if CONFIG['number'] in {5,6}:
        for q in json.loads((BASE/'测试/多轮对话用例.json').read_text(encoding='utf-8')):
            conversations.append(kb.ask(q,session_id='acceptance-dialogue',use_cache=False))
    profiler=cProfile.Profile();profiler.enable()
    for item in gold:kb.retrieve(item['question'])
    profiler.disable();profiler.dump_stats(str(BASE/'测试/profile.prof'))
    with (BASE/'测试/profile.txt').open('w',encoding='utf-8') as out:pstats.Stats(profiler,stream=out).sort_stats('cumtime').print_stats(25)
    started=time.perf_counter()
    def load(i):
        s=time.perf_counter();kb.retrieve(gold[i%len(gold)]['question']);return (time.perf_counter()-s)*1000
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:times=sorted(pool.map(load,range(40)))
    report={'project':CONFIG,'questions':len(gold),'strategies':results,'answers':answers,'conversations':conversations,
        'load':{'requests':40,'workers':4,'p50_ms':statistics.median(times),'p95_ms':times[37],'max_ms':times[-1],
                'elapsed_s':time.perf_counter()-started,'scope':'检索引擎并发；不包含LLM生成，不是生产高并发验收'},
        'definitions':{'hit_at_5':'标注页命中率，不等于问答准确率','labelled_page_precision_at_5':'前5片段中标注来源页比例；未标注但有效的重复披露会被低估',
          'fact_coverage_proxy':'标准事实关键词覆盖率代理，不是RAGAS忠实度','answer_accuracy':'需人工核对所有数字、单位、口径、限定和额外事实'}}
    dump_json(BASE/'测试/实测结果.json',report)
    lines=['# 工单'+str(CONFIG['number'])+'实测报告','',f'样本 {len(gold)} 题；原始逐题答案与来源在实测结果.json。','',
           '|策略|Hit@5|MRR@5|事实关键词覆盖代理|','|---|---:|---:|---:|']
    for key,val in results.items():lines.append(f"|{key}|{val['hit_at_5']:.1%}|{val['mrr_at_5']:.3f}|{val['fact_coverage_proxy']:.1%}|")
    lines+=['','## 实际生成回答','']
    for row in answers:lines += [f"### ID {row['id']}：{row['question']}",'',row['result']['answer'],'',f"耗时 {row['result']['elapsed_ms']:.2f} ms；状态 {row['result']['generation']}。",'']
    lat=[x['result']['elapsed_ms'] for x in answers]
    lines+=['## 性能与边界','',f"完整生成耗时 {min(lat):.2f}-{max(lat):.2f} ms；3秒目标 {'本次样本通过' if max(lat)<=3000 else '本次未通过'}。",'缓存、建库和模型冷启动未纳入本次稳态统计。',
        '检索命中、关键词覆盖不等于问答正确。RAGAS指标如适用会另行实际运行，不杜撰缺失值。',
        '9份年报全部入库；sample_questions中的占位公司已按具体年报具名化。工单8原eval_question.md缺失，本项目附补建版本。' if CONFIG['dataset']=='annual_reports' else '图片摘要保留原多模态输出和人工原图校正；无OCR页需按清单人工复核。']
    (BASE/'测试/实测报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print('DONE',CONFIG['number'],flush=True)

if __name__=='__main__':run()
