import os, re
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
norm = lambda s: re.sub(r"\s+", "", s)
docs = {f[:6]: norm(open(os.path.join(RAW,f), encoding="utf-8").read()) for f in os.listdir(RAW)}
for ph in ["步入正轨","增速的新高","新高","巩固收入贡献","支柱地位","资金同业"]:
    hits = [c for c,t in docs.items() if ph in t]
    print(f"{ph:12s} -> {hits}")
print("\n=== 000001 董事长致辞 附近 ===")
t = docs["000001"]; i = t.find("董事长致辞")
print(t[i:i+700])
