# 工单编号：人工智能 NLP-RAG-RAG 性能瓶颈识别与优化
import json
from config_v13 import PROFILE_FILE, COMPARE_FILE

with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
    before = json.load(f)
with open(COMPARE_FILE, 'r', encoding='utf-8') as f:
    after = json.load(f)

print('=' * 70)
print('优化前后对比')
print('=' * 70)

b = before['average']
a = after['average']

before_retrieve = b.get('vector', 0) + b.get('bm25', 0)

rows = [
    ('embed',    b.get('embed', 0),   a.get('embed', 0)),
    ('retrieve', before_retrieve,     a.get('retrieve', 0)),
    ('rerank',   b.get('rerank', 0),  a.get('rerank', 0)),
    ('llm',      b.get('llm', 0),     a.get('llm', 0)),
    ('total',    b.get('total', 0),   a.get('total', 0)),
]

print('%-10s %10s %10s %10s' % ('阶段', '优化前', '优化后', '变化'))
print('-' * 70)
for name, bv, av in rows:
    if bv > 0:
        gain = (bv - av) / bv * 100
        print('%-10s %9.3fs %9.3fs %+9.1f%%' % (name, bv, av, gain))
    else:
        print('%-10s %9.3fs %9.3fs' % (name, bv, av))

print('=' * 70)
print('目标：总耗时 < 3.0s')
print('优化前：%.3fs' % b.get('total', 0))
print('优化后：%.3fs' % a.get('total', 0))