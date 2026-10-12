import os, re
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
docs = {f[:6]: open(os.path.join(RAW,f), encoding="utf-8").read() for f in os.listdir(RAW)}
keys = ["董事长致辞","逾越者联盟","拨备覆盖率","绿色金融","金融科技","逾期60天","零售业务",
        "生态","咖啡","出行预订","影票","资产负债匹配","数字化转型","资本充足率","不良贷款率"]
print("code    " + " ".join(f"{k[:4]:>6s}" for k in keys))
for c, t in sorted(docs.items()):
    print(f"{c}  " + " ".join(f"{t.count(k):>6d}" for k in keys))
print()
# 找出各文档自称的年份
for c, t in sorted(docs.items()):
    yrs = re.findall(r"(20\d\d)\s*年年度报告", t[:3000])
    print(c, "title-years:", set(yrs), "| 标题片段:", t[:60].replace("\n"," / "))
