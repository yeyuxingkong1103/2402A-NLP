"""工单编号：人工智能NLP-RAG-功能测试及评估。
全文全页解析；年报表格按含财务数据的前30页抽取，复杂图形页不跑无限制表格线段求交。
"""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pymupdf
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from app.core import Embedder,split_text,dump_json,build_graph
os.environ['EMBEDDING_DEVICE']='cuda'
def main():
    chunks=[];pages_dump={};documents=[]
    for path in sorted((BASE/'data/documents').glob('*.pdf')):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();pages=[];tables_count=0
        with pymupdf.open(path) as doc:
            for number,page in enumerate(doc,1):
                text=page.get_text();record={'pdf_page':number,'printed_page':'','text':text,'tables':[]}
                for part in split_text(text):
                    chunks.append({'id':digest[:12]+f'-p{number}-c{len(chunks)}','document':path.name,
                        'sha256':digest,'pdf_page':number,'printed_page':'','kind':'text','text':part})
                if number<=30 and any(t in text for t in ['营业收入','净利润','资产总额','资本充足率']):
                    try:
                        for table in page.find_tables().tables:
                            rows=[[str(x or '').replace('\n',' ') for x in row] for row in table.extract()]
                            if len(rows)<2:continue
                            record['tables'].append(rows);tables_count+=1
                            for part in split_text('表格\n'+'\n'.join(' | '.join(row) for row in rows),1000,180):
                                chunks.append({'id':digest[:12]+f'-p{number}-t{len(chunks)}','document':path.name,
                                    'sha256':digest,'pdf_page':number,'printed_page':'','kind':'table','text':part})
                    except Exception as exc:record['table_warning']=str(exc)
                else:record['table_extraction']='not_attempted;全文仍入库'
                pages.append(record)
        pages_dump[path.name]=pages
        documents.append({'name':path.name,'sha256':digest,'pages':len(pages),'tables':tables_count,
            'table_scope':'前30页含财务指标表格；正文全部页面','empty_pages':[p['pdf_page'] for p in pages if not p['text'].strip()]})
        print('全文提取',path.name,len(pages),flush=True)
    index=BASE/'data/index'
    dump_json(index/'chunks.json',chunks);dump_json(index/'pages.json',pages_dump)
    embedder=Embedder();vectors=embedder.encode([c['text'] for c in chunks])
    np.save(index/'vectors.npy',vectors)
    dump_json(index/'manifest.json',{'documents':documents,'chunks':len(chunks),'embedding':embedder.name,
        'embedding_status':'本地BGE','dimension':512,'table_scope':'财务摘要页表格，全文全页',
        'process_note':'第一次无限制find_tables在复杂年报页耗时过长，被取消；改为有界表格范围后重建'})
    build_graph(chunks);print('年报索引完成',len(chunks),flush=True)
if __name__=='__main__':main()
