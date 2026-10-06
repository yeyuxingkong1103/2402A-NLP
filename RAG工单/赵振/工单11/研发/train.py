"""工单11：中文金融Embedding问答数据生成、微调、前后检索评估。"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '研发/data'
TEST = ROOT / '测试'
BASE = 'moka-ai/m3e-small'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(name, value):
    ((TEST if name in ('before.json','comparison.json','training_log.json','judge_audit.json') else DATA) / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(name):
    return json.loads(((TEST if name in ('before.json','comparison.json','training_log.json','judge_audit.json') else DATA) / name).read_text(encoding='utf-8'))


def llm(prompt, system='只根据原文作答，不编造事实。', json_mode=False):
    body = {'model': 'deepseek-r1:7b', 'prompt': '<｜begin▁of▁sentence｜>' + system +
            '<｜User｜>' + prompt + '<｜Assistant｜><think>\n</think>\n', 'raw': True, 'stream': False,
            'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': 1200}}
    if json_mode:
        body['format'] = json_mode if isinstance(json_mode,dict) else 'json'
    request = urllib.request.Request('http://127.0.0.1:11434/api/generate',
              data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    response = json.load(OPENER.open(request, timeout=240))
    return re.sub(r'<think>.*?</think>', '', response['response'], flags=re.S).strip()


def encoder(path=None):
    torch.set_num_threads(4)
    cache = Path.home() / '.cache/huggingface/hub/models--moka-ai--m3e-small/snapshots'
    local = next((p for p in sorted(cache.glob('*')) if (p / 'modules.json').exists()), None)
    model = SentenceTransformer(str(path or local or BASE), device='cpu')
    model.max_seq_length = 256
    return model


def prepare(total=80,pdf_folder=None):
    if pdf_folder:
        import pymupdf
        pages=[{'file':p.name,'page':i+1,'text':'\n\n'.join(b[4] for b in page.get_text('blocks') if b[6]==0)}
               for p in sorted(Path(pdf_folder).glob('*.pdf')) for i,page in enumerate(pymupdf.open(p))]
        save('pages.json',pages)
    random.seed(42)
    corpus, used = [], set()
    words = ['收入', '资金', '股东', '发行', '利润', '客户', '采购', '研发', '关联', '风险', '技术']
    for p in read('pages.json'):
        body = re.sub(r'\s+', '', p['text'])
        for start in range(100, len(body)-180, 400):
            text = body[start:start+400]
            key = (p['file'], p['page'])
            if key not in used and len(text) >= 180 and sum(w in text for w in words) >= 2 and sum(c.isdigit() for c in text)/len(text) < 0.15:
                corpus.append({'id': p['file']+'-'+str(p['page']), 'file': p['file'], 'page': p['page'], 'text': text})
                used.add(key)
    random.shuffle(corpus)
    corpus = corpus[:400]
    save('corpus.json', corpus)
    generated = read('generated.json') if (DATA / 'generated.json').exists() else []
    for r in generated:
        q = r.get('question','')
        r['valid'] = r['valid'] and q.endswith(('？','?')) and len(q) <= 120
    done = {r['id'] for r in generated}
    for doc in corpus:
        if sum(r['valid'] for r in generated) >= total:
            break
        if doc['id'] in done:
            continue
        prompt = '编写一条用户会问的招股书事实问题，不能询问如何生成问题。示例：原文“甲公司注册资本100万元”，输出{"question":"甲公司的注册资本有多少？","answer":"注册资本100万元"}。现在根据下面原文写新的问题，换一种措辞。answer必须逐字复制原文的连续短语，不要改写，不要补单位。只返回JSON，字段question和answer。\n原文：' + doc['text']
        raw = llm(prompt, json_mode=True)
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            item = {}
        answer = re.sub(r'\s+', '', item.get('answer', ''))
        q = item.get('question','')
        valid = 8 <= len(q) <= 120 and q.endswith(('？','?')) and len(answer) >= 6 and answer in doc['text']
        generated.append({**doc, **item, 'valid': valid, 'raw': raw})
        save('generated.json', generated)
        print('生成', len(generated), '有效', sum(r['valid'] for r in generated), flush=True)
        if sum(r['valid'] for r in generated) >= total:
            break
    valid = [r for r in generated if r['valid']][:total]
    if len(valid) < total:
        raise ValueError('有效问答数量不足，请继续生成。')
    save('dataset.json', {'train': valid[:50], 'validation': valid[50:60], 'test': valid[-20:], 'development':valid[60:-20]})


def evaluate(model, samples, corpus):
    vectors = model.encode([d['text'] for d in corpus], batch_size=16, normalize_embeddings=True)
    queries = model.encode([s['question'] for s in samples], normalize_embeddings=True)
    ids, rows = [d['id'] for d in corpus], []
    for s, score in zip(samples, queries @ vectors.T):
        order = np.argsort(-score)
        rank = list(order).index(ids.index(s['id'])) + 1
        rows.append({'question': s['question'], 'gold_id': s['id'], 'rank': rank,
                     'top5': [{'id': ids[i], 'text': corpus[i]['text'], 'score': float(score[i])} for i in order[:5]]})
    ranks = [r['rank'] for r in rows]
    return {'MRR@10': float(np.mean([1/r if r <= 10 else 0 for r in ranks])),
            'Recall@1': float(np.mean([r <= 1 for r in ranks])),
            'Recall@5': float(np.mean([r <= 5 for r in ranks])),
            'Recall@10': float(np.mean([r <= 10 for r in ranks])),
            'NDCG@10': float(np.mean([1/np.log2(r+1) if r <= 10 else 0 for r in ranks])), 'details': rows}


def train(epochs,lr,batch_size):
    torch.manual_seed(42)
    random.seed(42)
    corpus, data = read('corpus.json'), read('dataset.json')
    model = encoder()
    before = evaluate(model, data['test'], corpus)
    baseline_val = evaluate(model, data['validation'], corpus)['MRR@10']
    save('before.json', before)
    for param in model.parameters():
        param.requires_grad=False
    for param in model[0].auto_model.encoder.layer[-1].parameters():
        param.requires_grad=True
    loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, logs = -1, []
    for epoch in range(epochs):
        model.train()
        samples = list(data['train'])
        random.shuffle(samples)
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start+batch_size]
            features = [model.tokenize([s[k] for s in batch]) for k in ('question', 'text')]
            optimizer.zero_grad()
            value = loss(features, None)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            logs.append({'epoch': epoch+1, 'step': len(logs)+1, 'loss': float(value.detach())})
            print(logs[-1], flush=True)
        model.eval()
        validation = evaluate(model, data['validation'], corpus)
        logs.append({'epoch': epoch+1, 'validation_MRR@10': validation['MRR@10']})
        save('training_log.json', logs)
        if validation['MRR@10'] > best:
            best = validation['MRR@10']
            model.save(str(ROOT / '优化/model'))
    after = evaluate(encoder(ROOT / '优化/model'), data['test'], corpus)
    save('comparison.json', {'base_model': BASE, 'seed': 42, 'epochs': epochs, 'batch_size': batch_size,
         'learning_rate': lr, 'trainable':'最后一个Transformer层，其余冻结', 'loss': 'MultipleNegativesRankingLoss', 'validation_before': baseline_val,
         'validation_best': best, 'before': before, 'after': after,
         'test_pages_in_train': len({(r['file'],r['page']) for r in data['train']} & {(r['file'],r['page']) for r in data['test']})})
    print('评估完成', {k:v for k,v in after.items() if k!='details'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--total', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--pdf-folder')
    args = parser.parse_args()
    prepare(args.total,args.pdf_folder) if args.action == 'prepare' else train(args.epochs,args.lr,args.batch_size)
