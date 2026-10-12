# -*- coding: utf-8 -*-
"""C vs D (with A/B context): stats, triple-pair conflict analysis, glyphs, Q verdicts."""
import json, re, difflib
from collections import defaultdict

K = 15
RUNS = ['A', 'B', 'C', 'D']
HORIZ = {'位于左侧', '位于右侧'}
VERT = {'位于上方', '位于下方'}


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

# ---------- stats ----------
p('=' * 96)
p('A / B / C / D 四方对照')
p('  A=旧模型+旧提示词  B=新模型+旧提示词  C=新模型+三元组v1  D=新模型+三元组v2')
p('=' * 96)
p('')
p(f'{"指标":<30}{"A":>13}{"B":>13}{"C":>13}{"D":>13}')
p('-' * 96)


def rowm(label, fn, f='{}'):
    p(f'{label:<30}' + ''.join(f'{f.format(fn(T[n])):>13}' for n in RUNS))


VLMKEY = lambda x: ('Visual Type' in x) or ('Position Triples' in x)
rowm('chunk 数', lambda c: len(c))
rowm('总字符', lambda c: sum(len(x) for x in c))
rowm('VLM 描述字符', lambda c: sum(len(x) for x in c if VLMKEY(x)))
rowm('VLM 占比 %', lambda c: sum(len(x) for x in c if VLMKEY(x)) / sum(len(x) for x in c) * 100, '{:.1f}')
rowm('块内重复字符', lambda c: dup(c)[0])
rowm('字符噪声占比 %', lambda c: dup(c)[0] / dup(c)[1] * 100, '{:.1f}')

# ---------- symbol fidelity ----------
p('')
p('=' * 96)
p('符号保真度')
p('=' * 96)
p(f'{"token":<30}{"A":>13}{"B":>13}{"C":>13}{"D":>13}')
p('-' * 96)
for label, tok in [('6′ U+2032', '6′'), ('6″ U+2033', '6″'), ("6' ASCII", "6'"),
                   ('下标 ₁₂₃ 合计', None), ('x₁', 'x₁'), ('h₁', 'h₁'), ('h₂', 'h₂'),
                   ('h1 ASCII', 'h1'), ('x1 ASCII', 'x1')]:
    if tok is None:
        vals = [sum(1 for ch in ''.join(T[n]) if 0x2081 <= ord(ch) <= 0x2083) for n in RUNS]
    else:
        vals = [''.join(T[n]).count(tok) for n in RUNS]
    p(f'{label:<30}' + ''.join(f'{v:>13}' for v in vals))

# ---------- triple conflict analysis ----------
TRIPLE = re.compile(r'\(([^,()]+),\s*(位于[^,()]+|重合),\s*([^,()]+)\)')


def triples(n):
    out = []
    for c in T[n]:
        for m in TRIPLE.finditer(c):
            a, r, b = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            out.append((a, r, b))
    return out


p('')
p('=' * 96)
p('三元组质量分析（Q3 矛盾是否消除）')
p('=' * 96)
for n in ['C', 'D']:
    tr = triples(n)
    pairs = defaultdict(set)
    for a, r, b in tr:
        pairs[frozenset((a, b))].add(r)
    both_hv = {k: v for k, v in pairs.items() if (v & HORIZ) and (v & VERT)}
    multi = {k: v for k, v in pairs.items() if len(v) > 1}
    p('')
    p(f'  --- {n} ---')
    p(f'    三元组总数: {len(tr)}   唯一无序对: {len(pairs)}')
    p(f'    同一对出现 >1 种关系: {len(multi)} 对')
    p(f'    **同一对同时含 水平(左/右) 与 垂直(上/下): {len(both_hv)} 对**')
    for k, v in sorted(both_hv.items(), key=lambda x: -len(x[1]))[:12]:
        p(f'        {sorted(k)} -> {sorted(v)}')

# Q3 pair specifically
p('')
p('  --- Q3 关键对 (4, 5) 的全部关系 ---')
for n in ['B', 'C', 'D']:
    rel = sorted({r for a, r, b in triples(n) if {a, b} == {'4', '5'}})
    p(f'    {n}: {rel if rel else "（无）"}')

p('')
p('  --- Q5 关键对 (6,6′),(6′,6″),(6,6″) 的全部关系 ---')
for n in ['B', 'C', 'D']:
    got = []
    for a, r, b in triples(n):
        if {a, b} in ({'6', '6′'}, {'6′', '6″'}, {'6', '6″'}):
            got.append(f'({a},{r},{b})')
    p(f'    {n}: {sorted(set(got)) if got else "（无）"}')

# ---------- airflow ----------
p('')
p('=' * 96)
p('气流顺序 (SECTION 4)')
p('=' * 96)
for n in ['B', 'C', 'D']:
    p('')
    p(f'  --- {n} ---')
    for c in T[n]:
        if 'Flow Path' in c or 'Airflow' in c:
            seg = c[c.find('SECTION 4'):]
            seg = seg[:seg.find('SECTION 5')] if 'SECTION 5' in seg else seg
            for ln in seg.splitlines():
                if ln.strip() and ('部件' in ln or re.match(r'^\s*\d+\.', ln) or '箭头' in ln or 'arrow' in ln.lower()):
                    p(f'    {ln.strip()}')

open('D_REPORT.txt', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
