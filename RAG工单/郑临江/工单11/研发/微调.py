"""工单编号：人工智能NLP-RAG项目-Embedding模型微调任务。
真实模型权重更新；按来源页分组划分，保留未见测试集与训练日志。
"""
import hashlib
import json
import random
import re
import time
import urllib.request
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE=Path(__file__).resolve().parent
DATA=BASE/'data/finetune';DATA.mkdir(parents=True,exist_ok=True)
MODEL=BASE/'models/bge-small-zh-v1.5'

def prepare():
    chunks=json.loads((BASE/'data/index/chunks.json').read_text(encoding='utf-8'))
    valid=[c for c in chunks if c['kind']=='text' and len(c['text'])>400 and re.search(r'财务|技术|募集|销售|客户|管理',c['text'])]
    random.Random(42).shuffle(valid)
    selected=[];seen=set()
    for c in valid:
        key=(c['document'],c['pdf_page'])
        if key in seen:continue
        seen.add(key);selected.append(c)
        if len(selected)==36:break
    records=[]
    for i,c in enumerate(selected):
        prompt='根据这段招股文件生成一个专业领域检索问答训练样本。问题应体现专业同义表达，不照抄段落标题；答案必须由该段明确支持。只返回JSON对象，键question与answer，不得编造。\n'+c['text']
        body=json.dumps({'model':'Qwen3.5:4B','messages':[{'role':'user','content':prompt}],
            'stream':False,'think':False,'format':'json','options':{'num_predict':250,'num_ctx':4096}}).encode()
        req=urllib.request.Request('http://127.0.0.1:11434/api/chat',body,{'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=120) as response:result=json.load(response)
            value=json.loads(result['message']['content'])
            if isinstance(value.get('question'),str) and isinstance(value.get('answer'),str):
                records.append({'question':value['question'],'answer':value['answer'],'positive':c['text'],
                    'document':c['document'],'pdf_page':c['pdf_page'],'source_chunk':c['id'],
                    'source_sha256':c['sha256'],'generator':'Qwen3.5:4B','validation':'原文配对、格式检查；合成问答需领域人工抽检'})
        except Exception as exc:print('dataset error',i,type(exc).__name__,flush=True)
        (DATA/'qa_all.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print('QA',i+1,len(records),flush=True)
    random.Random(17).shuffle(records)
    n=len(records)
    for name,rows in [('train',records[:int(n*.65)]),('validation',records[int(n*.65):int(n*.8)]),('test',records[int(n*.8):])]:
        (DATA/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

def evaluate(model,rows,corpus):
    model.eval()
    docs=model.encode(corpus,normalize_embeddings=True,show_progress_bar=False)
    queries=model.encode([r['question'] for r in rows],normalize_embeddings=True,show_progress_bar=False)
    scores=queries@docs.T
    ranks=[]
    for i,row in enumerate(rows):
        target=corpus.index(row['positive']);ranks.append(int(np.where(np.argsort(-scores[i])==target)[0][0])+1)
    return {'sample_count':len(rows),'recall_at_1':float(np.mean(np.array(ranks)<=1)),'recall_at_5':float(np.mean(np.array(ranks)<=5)),
            'mrr':float(np.mean(1/np.array(ranks))),'ranks':ranks,'candidate_count':len(corpus)}

def train():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    if not (DATA/'train.json').exists():prepare()
    train=json.loads((DATA/'train.json').read_text(encoding='utf-8'))
    val=json.loads((DATA/'validation.json').read_text(encoding='utf-8'))
    test=json.loads((DATA/'test.json').read_text(encoding='utf-8'))
    all_rows=json.loads((DATA/'qa_all.json').read_text(encoding='utf-8'))
    # 增加未标注真实片段作为干扰项，避免只在36个正例的小池上评估。
    chunks=json.loads((BASE/'data/index/chunks.json').read_text(encoding='utf-8'))
    corpus=list(dict.fromkeys([r['positive'] for r in all_rows]+[c['text'] for c in chunks if c['kind']=='text'][::4]))
    groups=[{(x['document'],x['pdf_page']) for x in rows} for rows in [train,val,test]]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    model=SentenceTransformer(str(MODEL),device='cuda',local_files_only=True)
    model.max_seq_length=256
    before=evaluate(model,test,corpus)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    logs=[];best=-1
    for epoch in range(1,6):
        model.train();rows=train.copy();random.Random(epoch).shuffle(rows)
        losses=[]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            if len(batch)<2:continue
            optimizer.zero_grad()
            q=model({k:(v.to('cuda') if isinstance(v,torch.Tensor) else v) for k,v in model.preprocess([r['question'] for r in batch]).items()})['sentence_embedding']
            d=model({k:(v.to('cuda') if isinstance(v,torch.Tensor) else v) for k,v in model.preprocess([r['positive'] for r in batch]).items()})['sentence_embedding']
            score=torch.nn.functional.normalize(q,dim=1)@torch.nn.functional.normalize(d,dim=1).T*20
            loss=torch.nn.functional.cross_entropy(score,torch.arange(len(batch),device='cuda'))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation=evaluate(model,val,corpus)
        logs.append({'epoch':epoch,'mean_loss':float(np.mean(losses)),'validation':validation})
        if validation['mrr']>best:
            best=validation['mrr'];model.save_pretrained(str(BASE/'models/bge-finance-finetuned'))
        print('epoch',epoch,'loss',logs[-1]['mean_loss'],'validation',validation,flush=True)
    tuned=SentenceTransformer(str(BASE/'models/bge-finance-finetuned'),device='cuda',local_files_only=True)
    after=evaluate(tuned,test,corpus)
    report={'base_model':'BAAI/bge-small-zh-v1.5','device':torch.cuda.get_device_name(0),'loss':'MultipleNegativesRankingLoss / in-batch InfoNCE',
        'lr':1e-5,'batch_size':4,'epochs':5,'max_seq_length':256,'splits':{'train':len(train),'validation':len(val),'test':len(test)},
        'page_group_leakage':False,'selection':'仅按验证集MRR选择checkpoint，测试集只做前后评估',
        'before':before,'after':after,'training':logs,'improved':after['mrr']>before['mrr'],
        'limits':'小规模合成问答实验，不保证泛化；工单验收的“变好”只在实测指标提升时成立'}
    (BASE.parent/'测试/微调实测结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINETUNE',before,after,flush=True)

if __name__=='__main__':train()
