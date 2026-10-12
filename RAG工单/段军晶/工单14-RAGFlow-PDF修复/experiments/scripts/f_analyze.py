# -*- coding: utf-8 -*-
"""Final A..F comparison, focused on E vs F (code post-processor)."""
import json, re
from collections import defaultdict

K = 15
RUNS = ['A', 'B', 'C', 'D', 'E', 'F']
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


def triples(n, chunk=None):
    src = T[n] if chunk is None else [T[n][chunk]]
    out = []
    for c in src:
        for m in TRIPLE.finditer(c):
            out.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return out


p('=' * 104)
p('A~F 六方对照')
p('  C=三元组v1  D=v2(5条)  E=v3(+3条排他)  F=E提示词 + 代码后处理')
p('=' * 104)
p('')
p(f'{"指标":<26}{"A":>12}{"B":>12}{"C":>12}{"D":>12}{"E":>12}{"F":>12}')
p('-' * 104)


def rowm(label, fn, f='{}'):
    p(f'{label:<26}' + ''.join(f'{f.format(fn(T[n])):>12}' for n in RUNS))


VLMKEY = lambda x: ('Visual Type' in x) or ('Position Triples' in x)
rowm('chunk 数', lambda c: len(c))
rowm('总字符', lambda c: sum(len(x) for x in c))
rowm('VLM 描述字符', lambda c: sum(len(x) for x in c if VLMKEY(x)))
rowm('VLM 占比 %', lambda c: sum(len(x) for x in c if VLMKEY(x)) / sum(len(x) for x in c) * 100, '{:.1f}')
rowm('块内重复字符', lambda c: dup(c)[0])
rowm('字符噪声占比 %', lambda c: dup(c)[0] / dup(c)[1] * 100, '{:.1f}')

p('')
p('=' * 104)
p('★ 排他规则：同一部件对保留几条关系（越低越好，0 = 完全无歧义）')
p('=' * 104)
p(f'{"轮次":<8}{"chunk":>8}{"三元组":>10}{"唯一对":>10}{"违反对数":>12}{"含H+V":>10}')
p('-' * 104)
for n in ['C', 'D', 'E', 'F']:
    tot_v = tot_p = tot_t = 0
    for ci in range(len(T[n])):
        tr = triples(n, ci)
        if not tr:
            continue
        pr = defaultdict(list)
        for a, r, b in tr:
            pr[frozenset((a, b))].append(r)
        viol = {k: v for k, v in pr.items() if len(v) > 1}
        hv = {k: v for k, v in pr.items() if (set(v) & HORIZ) and (set(v) & VERT)}
        tot_t += len(tr); tot_p += len(pr); tot_v += len(viol)
        p(f'{n:<8}{ci+1:>8}{len(tr):>10}{len(pr):>10}{len(viol):>12}{len(hv):>10}')
    p(f'{"":<8}{"合计":>8}{tot_t:>10}{tot_p:>10}{tot_v:>12}')
    p('-' * 104)

p('')
p('=' * 104)
p('★ Q3 关键对 (4,5) —— 期望只剩 (4, 位于左侧, 5)')
p('=' * 104)
OPT = {'位于左侧': 'A ✔正解', '位于右侧': 'B ✘', '位于上方': 'C ✘', '位于下方': 'D ✘'}
for n in ['C', 'D', 'E', 'F']:
    rel = [(a, r, b) for a, r, b in triples(n) if {a, b} == {'4', '5'}]
    cands = sorted({OPT[r] for _, r, _ in rel if r in OPT})
    p(f'  {n}: {len(rel)} 条 -> {[(a,r,b) for a,r,b in rel]}')
    p(f'      候选选项: {cands if cands else "（无）"}  歧义度={len(cands)}')

p('')
p('=' * 104)
p('★ Q5 气流顺序 + 盘间三元组')
p('=' * 104)
for n in ['C', 'D', 'E', 'F']:
    seq = None
    for c in T[n]:
        if 'Flow Path' in c:
            seg = c[c.find('SECTION 4'):]
            seg = seg[:seg.find('SECTION 5')] if 'SECTION 5' in seg else seg
            s = re.findall(r'^\s*\d+\.\s*(?:部件)?([0-9]+[′″]?)', seg, re.M)
            if s and '6' in ''.join(s):
                seq = s
    disks = sorted({f'({a},{r},{b})' for a, r, b in triples(n)
                    if {a, b} in ({'6', '6′'}, {'6′', '6″'}, {'6', '6″'})})
    p(f'  {n} 附图1流向: {" -> ".join(seq) if seq else "（无）"}')
    p(f'      盘间三元组({len(disks)}): {disks if disks else "（无）"}')

p('')
p('=' * 104)
p('符号保真度')
p('=' * 104)
p(f'{"token":<24}' + ''.join(f'{n:>11}' for n in RUNS))
p('-' * 104)
for label, tok in [('6′ U+2032', '6′'), ('6″ U+2033', '6″'), ("6' ASCII", "6'"),
                   ('下标 ₁₂₃ 合计', None), ('x₁', 'x₁'), ('h₁', 'h₁'), ('h₂', 'h₂'), ('h1 ASCII', 'h1')]:
    if tok is None:
        vals = [sum(1 for ch in ''.join(T[n]) if 0x2081 <= ord(ch) <= 0x2083) for n in RUNS]
    else:
        vals = [''.join(T[n]).count(tok) for n in RUNS]
    p(f'{label:<24}' + ''.join(f'{v:>11}' for v in vals))

open('F_REPORT.txt', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
