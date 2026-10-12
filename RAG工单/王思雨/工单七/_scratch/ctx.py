import os, re, sys
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
docs = {f[:6]: open(os.path.join(RAW,f), encoding="utf-8").read() for f in os.listdir(RAW)}
norm = lambda s: re.sub(r"[ \t]+", "", s)
code, ph, width = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 300
t = norm(docs[code])
start = 0
while True:
    i = t.find(ph, start)
    if i < 0: break
    print(f"--- {code} @{i} ---"); print(t[max(0,i-width):i+width].replace("\n"," / ")); print()
    start = i + 1
