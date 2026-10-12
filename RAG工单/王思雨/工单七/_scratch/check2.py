import os, re
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
norm = lambda s: re.sub(r"\s+", "", s)          # 去掉抽取产生的空格后再检索，避免"逾期 60 天"漏配
docs = {f[:6]: norm(open(os.path.join(RAW,f), encoding="utf-8").read()) for f in os.listdir(RAW)}
keys = ["逾越者联盟","逾期60天","董事长致辞","拨备覆盖率","零售业务","二〇一九","2020年3月","绿色金融"]
print("code    " + " ".join(f"{k:>10s}" for k in keys))
for c,t in sorted(docs.items()):
    print(f"{c}  " + " ".join(f"{t.count(k):>10d}" for k in keys))
# 打印"逾越者联盟"上下文，确认归属
for c,t in docs.items():
    i = t.find("逾越者联盟")
    if i>=0: print(f"\n[600036?] {c} 上下文: ...{t[i-90:i+90]}...")
