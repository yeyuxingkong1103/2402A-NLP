# -*- coding: utf-8 -*-
import json

d = json.load(open('patent_chunks.json', encoding='utf-8'))
ch = [c.get('content', '') for c in d['data']['chunks']]
total = sum(len(c) for c in ch)

out = []
p = out.append

p('=== 规模 ===')
p(f'chunk 数: {len(ch)} | 总字符: {total}')

# VLM 英文描述
vlm = sum(len(c) for c in ch if 'Visual Type' in c)
p(f'VLM 英文描述: {vlm} 字符 ({vlm/total*100:.1f}%)')

# 重复度：归一化后（去掉所有空白）用 15 字滑窗统计
K = 15


def norm(s):
    return ''.join(s.split())


dupe_total = 0
norm_total = 0
for c in ch:
    n = norm(c)
    norm_total += len(n)
    if len(n) < K:
        continue
    seen = set()
    repeated = set()
    for i in range(len(n) - K + 1):
        sh = n[i:i + K]
        if sh in seen:
            repeated.add(i)
        else:
            seen.add(sh)
    dupe_total += len(repeated)

p(f'归一化后字符: {norm_total}')
p(f'块内重复字符(滑窗K={K}): {dupe_total} ({dupe_total/norm_total*100:.1f}%)')

# 6 个问题关键信息
all_text = ''.join(ch)
p('')
p('=== 6 个问题的关键信息检索 ===')
probes = [
    ('Q1 发明人 P-吉特勒', ['P·吉特勒']),
    ('Q2 80至95%+台阶', ['80 至95%', '80至95%']),
    ('Q3 部件左右空间关系', ['部件4位于', '左侧', '位置关系', '相对于部件']),
    ('Q4 X1-X3 间隔含义', ['X1, 2, 3', '间隔x1到 x3', '间隔 x1']),
    ('Q5 气流经过先后顺序', ['先经过', '紧接着会经过']),
    ('Q6 h1 h2 用途(公式)', ['h2 + h1', 'X1, 2, 3=']),
]
for name, ks in probes:
    hit = [k for k in ks if k in all_text]
    mark = '[有]' if hit else '[缺失]'
    p(f'{mark} {name:22s} 命中: {hit if hit else "无"}')

# 视觉模型对两张图的识别
p('')
p('=== VLM 对附图 1 / 附图 2 的识别 ===')
for i, c in enumerate(ch):
    if 'Visual Type' in c:
        first = c.strip().split('\n')[0][:160]
        p(f'chunk {i+1} ({len(c)}字符): {first}')

open('analysis.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
