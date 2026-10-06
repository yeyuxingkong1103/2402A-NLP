# 临时：预热 + 采集 step6 演示用 API 响应（工单四）
import json
import urllib.request

BASE = "http://127.0.0.1:8004"


def call(path, payload=None):
    if payload is None:
        with urllib.request.urlopen(BASE + path, timeout=180) as r:
            return json.loads(r.read().decode())
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


print("预热中...")
call("/api/v4/ask", {"question": "本次发行的发行股数是多少？",
                     "doc_id": "招股说明书1"})
print("预热完成")

CASES = [
    ("step6_api_id105_组织结构图", {
        "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，"
                    "其中大客户销售部有几个销售处构成？",
        "doc_id": "招股说明书2"}),
    ("step6_api_id106_增长率图", {
        "question": "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是"
                    "哪个行业？负增长的是哪个行业？",
        "doc_id": "招股说明书2"}),
    ("step6_api_id2_募集资金_表格", {
        "question": "本次发行的募集资金总额是多少？",
        "doc_id": "招股说明书1"}),
    ("step6_api_id11_english", {
        "question": "How many shares are issued in this offering?",
        "doc_id": "招股说明书1"}),
]
for name, payload in CASES:
    r = call("/api/v4/ask", payload)
    slim = {
        "question": r.get("question"),
        "route": r.get("route"),
        "answer": r.get("answer"),
        "latency_ms": r.get("latency_ms"),
        "breakdown": r.get("breakdown"),
        "references": [
            {k: ref.get(k) for k in ("type", "ref_id", "page", "image_id",
                                     "caption", "score")}
            for ref in (r.get("references") or [])
        ],
    }
    path = f"docs/screenshots/{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"{name}: {r.get('latency_ms')}ms -> {path}")
    print("  ", (r.get("answer") or "")[:120].replace("\n", " "))
