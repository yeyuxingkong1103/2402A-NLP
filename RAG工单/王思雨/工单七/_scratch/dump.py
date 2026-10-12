# -*- coding: utf-8 -*-
import os, re, sys
PAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
code = sys.argv[1]
nums = set(int(x) for x in sys.argv[2:])
fn = [f for f in os.listdir(PAGES) if f.startswith(code)][0]
txt = open(os.path.join(PAGES, fn), encoding="utf-8").read()
chunks = re.split(r"\n===== PAGE (\d+) =====\n", txt)
for i in range(1, len(chunks), 2):
    if int(chunks[i]) in nums:
        print(f"===== PAGE {chunks[i]} =====\n{chunks[i+1]}")
