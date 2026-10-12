# -*- coding: utf-8 -*-
"""Unit-test dedup_component_triples against E's real VLM output."""
import re, logging, sys

# --- pull the function out of the patched file and exec it standalone ---
src = open('figure_parser_patched.py', encoding='utf-8').read()
seg = src[src.index('TRIPLE_RE = re.compile'):src.index('def vision_figure_parser_figure_data_wrapper')]
ns = {'re': re, 'logging': logging}
exec(seg, ns)
dedup = ns['dedup_component_triples']

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

for name in ['E_chunk17', 'E_chunk18']:
    text = open(f'../{name}.txt', encoding='utf-8').read()
    print('=' * 70)
    print(f'### {name}  INPUT triples:')
    for m in ns['TRIPLE_RE'].finditer(text):
        print('   ', m.group(0).strip())
    out = dedup(text, tag=f' [{name}]')
    print(f'--- {name}  OUTPUT SECTION 3:')
    h = re.search(r'#+\s*SECTION\s*3[^\n]*\n', out)
    s = h.end()
    n = re.search(r'\n#+\s*SECTION\s*4', out[s:])
    print(out[s:s + n.start()] if n else out[s:])
    print(f'--- other sections preserved? '
          f'SECTION4={"Flow Path" in out} SECTION5={"Captions" in out} '
          f'len_in={len(text)} len_out={len(out)}')
    print()
