# -*- coding: utf-8 -*-
"""A/B/C comparison: stats, glyph fidelity, Q coverage, retrieval risk."""
import json, difflib

K = 15
RUNS = ['A', 'B', 'C']


def load(n):
    d = json.load(open(f'{n}_chunks.json', encoding='utf-8'))
    return [c.get('content', '') for c in d['data']['chunks']]


def norm(s):
    return ''.join(s.split())


def dup(chunks):
    dt = nt = 0
    for c in chunks:
        n = norm(c)
        nt += len(n)
        if len(n) < K:
            continue
        seen, rep = set(), set()
        for i in range(len(n) - K + 1):
            sh = n[i:i + K]
            if sh in seen:
                rep.add(i)
            else:
                seen.add(sh)
        dt += len(rep)
    return dt, nt


T = {n: load(n) for n in RUNS}
L = []
p = L.append

p('=' * 92)
p('A / B / C 三方对照')
p('  A = 基线 (旧 prompt + 旧 VLM) | B = 新 VLM + 旧 prompt | C = 新 VLM + 三元组 prompt')
p('=' * 92)
p('')
p(f'{"指标":<30}{"A":>14}{"B":>14}{"C":>14}')
p('-' * 92)


def rowm(label, fn, f='{}'):
    vals = [fn(T[n]) for n in RUNS]
    p(f'{label:<30}' + ''.join(f'{f.format(v):>14}' for v in vals))


rowm('chunk 数', lambda c: len(c))
rowm('总字符', lambda c: sum(len(x) for x in c))
rowm('VLM 描述字符', lambda c: sum(len(x) for x in c if 'Visual Type' in x or 'Position Triples' in x))
rowm('VLM 占比 %', lambda c: sum(len(x) for x in c if 'Visual Type' in x or 'Position Triples' in x) / sum(len(x) for x in c) * 100, '{:.1f}')
rowm('归一化后字符', lambda c: sum(len(norm(x)) for x in c))
rowm('块内重复字符', lambda c: dup(c)[0])
rowm('字符噪声占比 %', lambda c: dup(c)[0] / dup(c)[1] * 100, '{:.1f}')

p('')
p('=' * 92)
p('符号保真度（Q4/Q5/Q6 高危字形）')
p('=' * 92)
p(f'{"token":<34}{"A":>14}{"B":>14}{"C":>14}')
p('-' * 92)
GLYPH = [
    ("6'  ASCII 撇 U+0027", "6'"),
    ('6"  ASCII 引号 U+0022', '6"'),
    ('6′  Unicode 单撇 U+2032', '6′'),
    ('6″  Unicode 双撇 U+2033', '6″'),
    ('x1  ASCII', 'x1'),
    ('X1  ASCII', 'X1'),
    ('x₁  下标 U+2081', 'x₁'),
    ('h1  ASCII', 'h1'),
    ('h₁  下标 U+2081', 'h₁'),
    ('h₂  下标 U+2082', 'h₂'),
    ('全下标 ₁₂₃ 计数', None),
]
for label, tok in GLYPH:
    if tok is None:
        vals = [sum(1 for ch in ''.join(T[n]) if 0x2081 <= ord(ch) <= 0x2083) for n in RUNS]
    else:
        vals = [''.join(T[n]).count(tok) for n in RUNS]
    p(f'{label:<34}' + ''.join(f'{v:>14}' for v in vals))

p('')
p('=' * 92)
p('Q1~Q6 关键词覆盖')
p('=' * 92)
PROBES = {
    'Q1 发明人': ['P·吉特勒'],
    'Q2 80至95%台阶': ['80 至95%', '80至95%'],
    'Q3 部件4在部件5左侧': ['(4, 位于左侧, 5)', '4 位于左侧, 5', '左侧'],
    'Q3 反向/干扰：(4,下方,5)': ['(4, 位于下方, 5)'],
    'Q4 X1-X3=配气盘间隔': ['X1, 2, 3', '间隔x1到 x3', '间隔 x₁ 到 x₃', '配气带孔盘 6，6′，6″'],
    'Q5 气流经过 6″ 再 6′': ['先经过', '气流方向如箭头', '气流方向'],
    'Q5 盘序 6,6′,6″ 左→右': ['(6, 位于左侧, 6′)', '(6′, 位于左侧, 6″)'],
    'Q6 h1/h2 用途': ['h₂）大约为直径 D 的 35%', 'h₁ 为外壳直径 D 的 5%', 'h2 + h1'],
}
for name, ks in PROBES.items():
    p('')
    p(f'  {name}')
    for n in RUNS:
        allt = ''.join(T[n])
        hit = [k for k in ks if k in allt]
        p(f'    {n}: {"[有] " + str(hit) if hit else "[缺失]"}')

p('')
p('=' * 92)
p('检索风险：题面 ASCII 形态 vs 索引 Unicode 形态')
p('=' * 92)
p(f'{"题面检索词":<26}{"A 命中":>12}{"B 命中":>12}{"C 命中":>12}')
p('-' * 92)
QUERY = ['X1', 'X2', 'X3', "6'", '6"', '部件4', '部件5', '气流方向']
for q in QUERY:
    vals = [('是' if ''.join(T[n]).count(q) else '否') for n in RUNS]
    p(f'{q:<26}' + ''.join(f'{v:>12}' for v in vals))

p('')
p('=' * 92)
p('非 VLM 文本漂移（OCR 是否稳定）')
p('=' * 92)
nv = {}
for n in RUNS:
    nv[n] = norm(''.join(x for x in T[n] if 'Visual Type' not in x and 'Position Triples' not in x))
base = nv['B']
for n in RUNS:
    r = difflib.SequenceMatcher(None, base, nv[n], autojunk=False).ratio()
    p(f'  {n} 非VLM归一化字符={len(nv[n]):6d}  与B相似度={r:.4f}')
    if n != 'B':
        sm = difflib.SequenceMatcher(None, base, nv[n], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != 'equal':
                p(f'      {tag}: B[{i1}:{i2}]={base[i1:i2]!r}  {n}[{j1}:{j2}]={nv[n][j1:j2]!r}')

open('BC_REPORT.txt', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
