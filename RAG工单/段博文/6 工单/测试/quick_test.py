# -*- coding: utf-8 -*-
import requests
BASE = 'http://127.0.0.1:8000'

# 新Q9召回
r = requests.get(BASE + '/api/search', params={'query': '武汉兴图新科电子股份有限公司的注册地址在哪里？', 'top_k': 5, 'strategy': 'hybrid'}, timeout=60)
d = r.json()
ctx = ' '.join(x['page_content'] for x in d['results'])
print('新Q9召回:', '地址' in ctx or '注册地址' in ctx, '| 命中', d['total'])

# NOT语法
r = requests.get(BASE + '/api/search', params={'query': '兴图 NOT 力源', 'top_k': 3, 'strategy': 'fulltext'}, timeout=30)
print('NOT语法命中:', r.json()['total'])

# 短语语法
r = requests.get(BASE + '/api/search', params={'query': '"法定代表人"', 'top_k': 3, 'strategy': 'fulltext'}, timeout=30)
print('短语命中:', r.json()['total'])
