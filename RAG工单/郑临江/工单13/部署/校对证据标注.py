"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。对相同事实的重复披露补全证据页。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in [2,3,4,5,6,11,12,13]:
    folder=ROOT/f'工单{num}'
    p=folder/'测试/gold.json';gold=json.loads(p.read_text(encoding='utf-8'))
    for row in gold:
        if row['id'] in {260,33}:row['pages']=[4,36,129]
    p.write_text(json.dumps(gold,ensure_ascii=False,indent=2),encoding='utf-8')
    f=folder/'测试/实测结果.json'
    if not f.exists():continue
    report=json.loads(f.read_text(encoding='utf-8'));lookup={g['id']:g for g in gold}
    for detail in report['strategies'].values():
        for row in detail['samples']:
            g=lookup[row['id']]
            ranks=[i+1 for i,h in enumerate(row['hits']) if h['document']==g['document'] and h['pdf_page'] in g['pages']]
            row['hit_at_5']=bool(ranks);row['mrr_at_5']=1/min(ranks) if ranks else 0
            row['labelled_page_precision_at_5']=len(ranks)/len(row['hits']) if row['hits'] else 0
        detail['hit_at_5']=sum(r['hit_at_5'] for r in detail['samples'])/len(detail['samples'])
        detail['mrr_at_5']=sum(r['mrr_at_5'] for r in detail['samples'])/len(detail['samples'])
    report['annotation_note']='收入与占比在物理第4、36、129页重复披露，已核对并补全可接受证据页；保留原始答案与耗时，未重测/改写模型输出。'
    f.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('重复披露证据页已统一校对')
