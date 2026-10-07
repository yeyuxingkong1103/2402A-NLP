# -*- coding: utf-8 -*-
import requests

BASE = 'http://127.0.0.1:8000'

questions = [
    ('平安银行2019年的董事长是谁？', ['谢永林']),
    ('招商银行2019年的净利润是多少亿元？', ['928']),
    ('邮储银行2019年年报中提到的三大风险是什么？', ['信用风险', '市场风险']),
    ('中国人寿2020年的保费收入是多少？', ['6122']),
    ('国泰君安2021年的法定代表人是谁？', ['贺青']),
]

for q, keywords in questions:
    try:
        r = requests.get(f'{BASE}/api/search', params={
            'query': q, 'top_k': 3, 'strategy': 'hybrid', 'reranker': 'tfidf'
        }, timeout=30)
        d = r.json()
        ctx = ' '.join([x['page_content'] for x in d['results']])
        hit = any(k in ctx for k in keywords)
        print(f'{"✓" if hit else "✗"} {q[:35]}... | 命中: {d["total"]}')
    except Exception as e:
        print(f'✗ {q[:35]}... | 失败: {e}')
