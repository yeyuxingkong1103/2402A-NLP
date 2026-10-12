# -*- coding: utf-8 -*-
"""Whitespace-insensitive search over page-marked txt files.
Usage: python search.py <code> <query...> [--win N]
Prints page numbers and the ORIGINAL excerpt (spaces preserved).
"""
import os, re, sys

PAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")

def load(code):
    fn = None
    for f in os.listdir(PAGES):
        if f.startswith(code):
            fn = f
    if not fn:
        raise SystemExit("no file for " + code)
    txt = open(os.path.join(PAGES, fn), encoding="utf-8").read()
    chunks = re.split(r"\n===== PAGE (\d+) =====\n", txt)
    # chunks[0] is preamble; then (num, body) pairs
    pages = []
    for i in range(1, len(chunks), 2):
        pages.append((int(chunks[i]), chunks[i+1]))
    return fn, pages

def compress(s, keep='　 '):
    """Return compressed string and index map (compressed idx -> orig idx)."""
    buf = []
    m = []
    for i, ch in enumerate(s):
        if ch.isspace():
            continue
        buf.append(ch)
        m.append(i)
    return "".join(buf), m

def find(code, query, win=160):
    fn, pages = load(code)
    key = re.sub(r"\s+", "", query)
    print(f"### {fn}  query(depaced)={key!r}")
    hits = 0
    for num, body in pages:
        comp, m = compress(body)
        start = 0
        while True:
            idx = comp.find(key, start)
            if idx < 0:
                break
            hits += 1
            o0 = m[max(0, idx - win)]
            o1 = m[min(len(m)-1, idx + len(key) + win)] + 1
            excerpt = body[o0:o1].replace("\n", " ")
            print(f"  -- p.{num} @comp{idx} --")
            print("     " + excerpt)
            start = idx + 1
    print(f"  total hits: {hits}\n")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    win = 160
    if "--win" in args:
        i = args.index("--win")
        win = int(args[i+1])
        del args[i:i+2]
    code = args[0]
    query = " ".join(args[1:])
    find(code, query, win)
