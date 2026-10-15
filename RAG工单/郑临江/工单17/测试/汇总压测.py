"""工单17：汇总真实JMeter CSV，单独统计检索与上传；不把检索耗时写成LLM问答耗时。"""
import csv,json,statistics,math
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
report={'scope':'独立真实检索/缓存API，未包含每请求LLM生成或正式RAGFlow全栈；修改前后实验有部分并行负载，存在干扰因素','scenarios':{}}
for name in ['A','B','A-after','B-after','A-final']:
    path=BASE/'测试'/f'jmeter-{name}.jtl'
    if not path.exists():continue
    rows=[r for r in csv.DictReader(path.open(encoding='utf-8')) if r.get('success') in {'true','false'} and r.get('elapsed','').isdigit()]
    if not rows:continue
    samples={}
    for label in {r['label'] for r in rows}:
        selected=[r for r in rows if r['label']==label];times=sorted(int(r['elapsed']) for r in selected)
        samples[label]={'requests':len(times),'errors':sum(r['success']!='true' for r in selected),'p50_ms':statistics.median(times),
            'p95_ms':times[math.ceil(len(times)*.95)-1],'max_ms':max(times),'response_codes':{c:sum(r['responseCode']==c for r in selected) for c in {r['responseCode'] for r in selected}}}
    start=min(int(r['timeStamp']) for r in rows);end=max(int(r['timeStamp'])+int(r['elapsed']) for r in rows)
    report['scenarios'][name]={'duration_s':(end-start)/1000,'requests':len(rows),'errors':sum(r['success']!='true' for r in rows),'samples':samples,
        'file':path.name,'complete_600s':(end-start)>=595000}
(BASE/'测试/JMeter汇总.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    labels=[];latencies=[];errors=[]
    for name,s in report['scenarios'].items():
        labels.append(name);q=next(v for k,v in s['samples'].items() if '检索' in k);latencies.append(q['p95_ms']);errors.append(s['errors'])
    axes[0].bar(labels,latencies,color='#186d66');axes[0].set(ylabel='Retrieval P95 (ms)',title='Real HTTP retrieval requests')
    axes[1].bar(labels,errors,color='#416779');axes[1].set(ylabel='Failed HTTP requests',title='Includes upload backpressure')
    fig.tight_layout();fig.savefig(BASE/'测试/压测对比.png',dpi=180);plt.close(fig)
except ImportError:pass
