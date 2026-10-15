"""工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
from app.crossmodal import query
ROOT=Path(__file__).resolve().parents[1];gold=json.loads((ROOT/'测试/gold.json').read_text(encoding='utf-8'));rows=[]
for optimized in [False,True]:
    for g in gold:
        r=query(g['question'],optimized);rows.append({'optimized':optimized,'reference':g['reference'],'result':r,'correct':g['reference'] in r.get('answer','')})
(ROOT/'测试/实测结果.json').write_text(json.dumps({'results':rows,'accuracy':sum(x['correct'] for x in rows if x['optimized'])/len(gold),'scope':'扩展跨模态验证层，非完整RAGFlow全栈'},ensure_ascii=False,indent=2),encoding='utf-8');print(rows)
