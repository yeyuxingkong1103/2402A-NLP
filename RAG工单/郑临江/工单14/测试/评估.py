"""工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失。真实基准与优化回答。"""
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'研发'))
from app.pipeline import Pipeline,CONFIG,save
pipeline=Pipeline();gold=json.loads((ROOT/'测试/gold.json').read_text(encoding='utf-8'))
results=[]
for optimized in [False,True]:
    for row in gold:
        result=pipeline.ask(row['question'],row['document'],optimized=optimized)
        results.append({'optimized':optimized,'question':row['question'],'reference':row['reference'],'group':row.get('group'),
                        'result':result,'evaluation_status':'等待逐题人工/独立评审核对，不能用关键词覆盖冒充准确率'})
        save(ROOT/'测试/实测结果.json',{'gold':gold,'results':results,'scope':'独立扩展流水线真实调用；RAGFlow正式链路验收另列'})
        print(CONFIG['number'],optimized,row['question'],result['answer'],result['elapsed_ms'],flush=True)
