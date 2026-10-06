import json, time, urllib.request

body = json.dumps({
    "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
    "doc_id": "招股说明书2", "use_image": True,
}).encode()
t0 = time.time()
req = urllib.request.Request("http://127.0.0.1:8004/api/v4/ask",
                             data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    print("elapsed", round(time.time() - t0, 1), "latency_ms", d.get("latency_ms"))
    print("answer:", d.get("answer", "")[:300])
    print("images:", [im.get("image_id") for im in d.get("retrieved_images", [])])
except Exception as e:
    print("ERROR after", round(time.time() - t0, 1), repr(e))
