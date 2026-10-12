# -*- coding: utf-8 -*-
"""D vs E focus (with A/B/C context)."""
import json, re
from collections import defaultdict

K = 15
RUNS = ['A', 'B', 'C', 'D', 'E']
HORIZ = {'位于左侧', '位于右侧'}
VERT = {'位于上方', '位于下方'}
TRIPLE = re.compile(r'\(([^,()]+),\s*(位于[^,()]+|重合),\s*([^,()]+)\)')


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


def triples(n):
    out = []
    for c in T[n]:
        for m in TRIPLE.finditer(c):
            out.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return out


p('=' * 100)
p('A / B / C / D / E 五方对照')
p('  C=三元组v1  D=v2(5条)  E=v3(加3条排他规则)')
p('=' * 100)
p('')
p(f'{"指标":<28}{"A":>12}{"B":>12}{"C":>12}{"D":>12}{"E":>12}')
p('-' * 100)


def rowm(label, fn, f='{}'):
    p(f'{label:<28}' + ''.join(f'{f.format(fn(T[n])):>12}' for n in RUNS))


VLMKEY = lambda x: ('Visual Type' in x) or ('Position Triples' in x)
rowm('chunk 数', lambda c: len(c))
rowm('总字符', lambda c: sum(len(x) for x in c))
rowm('VLM 描述字符', lambda c: sum(len(x) for x in c if VLMKEY(x)))
rowm('VLM 占比 %', lambda c: sum(len(x) for x in c if VLMKEY(x)) / sum(len(x) for x in c) * 100, '{:.1f}')
rowm('块内重复字符', lambda c: dup(c)[0])
rowm('字符噪声占比 %', lambda c: dup(c)[0] / dup(c)[1] * 100, '{:.1f}')

p('')
p('=' * 100)
p('排他规则达标情况（每对最多 1 条关系）')
p('=' * 100)
p(f'{"轮次":<8}{"三元组总数":>12}{"唯一对":>10}{"重复对(违反)":>14}{"其中含H+V":>12}')
p('-' * 100)
for n in ['C', 'D', 'E']:
    tr = triples(n)
    pairs = defaultdict(list)
    for a, r, b in tr:
        pairs[frozenset((a, b))].append(r)
    viol = {k: v for k, v in pairs.items() if len(v) > 1}
    hv = {k: v for k, v in pairs.items() if (set(v) & HORIZ) and (set(v) & VERT)}
    p(f'{n:<8}{len(tr):>12}{len(pairs):>10}{len(viol):>14}{len(hv):>12}')

p('')
p('=' * 100)
p('Q3 关键对 (4,5)')
p('=' * 100)
for n in ['B', 'C', 'D', 'E']:
    rel = []
    for a, r, b in triples(n):
        if {a, b} == {'4', '5'}:
            rel.append((a, r, b))
    vals = sorted({r for _, r, _ in rel})
    p(f'  {n}: {vals if vals else "（无）"}')
    for a, r, b in rel:
        p(f'        ({a}, {r}, {b})')

p('')
p('=' * 100)
p('Q3 选项映射（A=左侧 B=右侧 C=上方 D=下方；正解 A）')
p('=' * 100)
OPT = {'位于左侧': 'A ✔正解', '位于右侧': 'B', '位于上方': 'C', '位于下方': 'D'}
for n in ['C', 'D', 'E']:
    vals = sorted({r for a, r, b in triples(n) if {a, b} == {'4', '5'}})
    mapped = [m for m in (OPT.get(v) for v in vals) if m]
    p(f'  {n}: {mapped}  → {len(mapped)} 个候选')

p('')
p('=' * 100)
p('Q5 盘序与气流')
p('=' * 100)
for n in ['B', 'C', 'D', 'E']:
    got = []
    for a, r, b in triples(n):
        if {a, b} in ({'6', '6′'}, {'6′', '6″'}, {'6', '6″'}):
            got.append(f'({a},{r},{b})')
    p(f'  {n} 盘间关系: {sorted(set(got)) if got else "（无）"}')
p('')
for n in ['C', 'D', 'E']:
    for c in T[n]:
        if 'Flow Path' in c and '1' in c:
            seg = c[c.find('SECTION 4'):]
            seg = seg[:seg.find('SECTION 5')] if 'SECTION 5' in seg else seg
            seq = re.findall(r'^\s*\d+\.\s*(?:部件)?([0-9]+[′″]?)', seg, re.M)
            if seq and ('6' in ''.join(seq)):
                p(f'  {n} 附图1流向: {" -> ".join(seq)}')

p('')
p('=' * 100)
p('符号保真度')
p('=' * 100)
p(f'{"token":<26}{"A":>12}{"B":>12}{"C":>12}{"D":>12}{"E":>12}')
p('-' * 100)
for label, tok in [('6′ U+2032', '6′'), ('6″ U+2033', '6″'), ("6' ASCII", "6'"),
                   ('下标 ₁₂₃ 合计', None), ('x₁', 'x₁'), ('h₁', 'h₁'), ('h₂', 'h₂'), ('h1 ASCII', 'h1')]:
    if tok is None:
        vals = [sum(1 for ch in ''.join(T[n]) if 0x2081 <= ord(ch) <= 0x2083) for n in RUNS]
    else:
        vals = [''.join(T[n]).count(tok) for n in RUNS]
    p(f'{label:<26}' + ''.join(f'{v:>12}' for v in vals))

p('')
p('=' * 100)
p('尺寸符号误入三元组（要求 #3）')
p('=' * 100)
DIM = re.compile(r'^(h|H|x|X|D)[₀-₉0-9]*$')
for n in ['D', 'E']:
    bad = [(a, r, b) for a, r, b in triples(n) if DIM.match(a) or DIM.match(b)]
    p(f'  {n}: 含尺寸符号的三元组 {len(bad)} 条  {bad[:8]}')

open('E_REPORT.txt', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
