# -*- coding: utf-8 -*-
import requests

BASE = 'http://127.0.0.1:8000'

questions = [
    '平安银行2019年的董事长是谁？',
    '招商银行2019年的净利润是多少？',
    '中国人寿2020年的保费收入是多少？',
]

for q in questions:
    try:
        r = requests.get(f'{BASE}/api/search', params={'query': q, 'top_k': 3, 'strategy': 'hybrid'}, timeout=30)
        d = r.json()
        print(f'Q: {q[:30]}... | 命中: {d["total"]}')
        for res in d['results'][:2]:
            print(f'  score={res.get("score",0):.3f} | {res["page_content"][:80]}...')
    except Exception as e:
        print(f'Q: {q[:30]}... | 失败: {e}')
    print()
