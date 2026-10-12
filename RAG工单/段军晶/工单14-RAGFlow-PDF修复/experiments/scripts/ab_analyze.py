# -*- coding: utf-8 -*-
"""A/B analyzer: same methodology as .ragflow_tmp/analyze.py, applied to A and B."""
import json, re, sys

K = 15  # sliding window for in-chunk duplicate detection (same as baseline)


def norm(s):
    return ''.join(s.split())


def dup_rate(chunks):
    dupe = 0
    tot = 0
    for c in chunks:
        n = norm(c)
        tot += len(n)
        if len(n) < K:
            continue
        seen, rep = set(), set()
        for i in range(len(n) - K + 1):
            sh = n[i:i + K]
            if sh in seen:
                rep.add(i)
            else:
                seen.add(sh)
        dupe += len(rep)
    return dupe, tot


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    return d['data']['chunks']


def stats(name, chunks):
    ch = [c.get('content', '') for c in chunks]
    total = sum(len(c) for c in ch)
    vlm = sum(len(c) for c in ch if 'Visual Type' in c)
    dupe, ntot = dup_rate(ch)
    return {
        'name': name, 'n': len(ch), 'total': total,
        'vlm': vlm, 'vlm_pct': vlm / total * 100 if total else 0,
        'norm': ntot, 'dupe': dupe,
        'dupe_pct': dupe / ntot * 100 if ntot else 0,
    }


# ---- probe sets: expanded with EN + ZH + spatial-relation specific terms ----
PROBES = {
    'Q1 发明人(纯文本)': ['P·吉特勒', 'P-吉特勒', '吉特勒'],
    'Q2 80至95%台阶(纯文本)': ['80 至95%', '80至95%', '80 �95%', '台阶'],
    'Q3 部件4相对部件5空间关系': ['部件4位于', '左侧', '右侧', 'to the left', 'on the left',
                          'left of', 'right of', 'positional relationship'],
    'Q4 X1-X3 间隔含义': ['X1, 2, 3', '间隔x1到 x3', '间隔 x1', 'x₁', 'X1'],
    'Q5 气流先后顺序': ['先经过', '紧接着会经过', 'airflow', 'air flow', 'first pass',
                  'then pass', "6''", '6"', '6′′'],
    'Q6 h1 h2 用途': ['h2 + h1', 'X1, 2, 3=', 'h₂', 'h2', 'h1'],
}


def probes(chunks):
    allt = ''.join(c.get('content', '') for c in chunks)
    out = {}
    for name, ks in PROBES.items():
        out[name] = [k for k in ks if k in allt]
    return out


if __name__ == '__main__':
    A = load('A_chunks.json')
    B = load('B_chunks.json')
    sa, sb = stats('A', A), stats('B', B)

    L = []
    p = L.append
    p('=' * 78)
    p('A/B 对照 —— 同一文档 CN100342976C.pdf，仅 VLM 变量不同')
    p('=' * 78)
    p('')
    p(f'{"指标":<26}{"A (qwen-vl-plus?)":>22}{"B (qwen3-vl-plus)":>22}{"Δ":>10}')
    p('-' * 78)

    def row(label, ka, kb, fmt='{}', delta=False):
        va, vb = sa[ka], sb[kb]
        d = ''
        if delta:
            d = f'{vb - va:+g}'
        p(f'{label:<26}{fmt.format(va):>22}{fmt.format(vb):>22}{d:>10}')

    row('chunk 数', 'n', 'n')
    row('总字符', 'total', 'total', delta=True)
    row('VLM 描述字符', 'vlm', 'vlm', delta=True)
    row('VLM 占比 %', 'vlm_pct', 'vlm_pct', '{:.1f}', delta=True)
    row('归一化后字符', 'norm', 'norm', delta=True)
    row('块内重复字符', 'dupe', 'dupe', delta=True)
    row('重复率 %', 'dupe_pct', 'dupe_pct', '{:.1f}', delta=True)

    p('')
    p('=' * 78)
    p('Q1~Q6 信息覆盖（关键词命中）')
    p('=' * 78)
    pa, pb = probes(A), probes(B)
    for name in PROBES:
        ha, hb = pa[name], pb[name]
        p('')
        p(f'  {name}')
        p(f'    A: {"[有] " + str(ha) if ha else "[缺失]"}')
        p(f'    B: {"[有] " + str(hb) if hb else "[缺失]"}')

    open('ab_report.txt', 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))
