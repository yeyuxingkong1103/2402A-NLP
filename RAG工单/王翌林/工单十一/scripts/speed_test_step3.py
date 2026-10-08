# -*- coding: utf-8 -*-
# 工单四：测试 modelscope 多线程并发下载速度（人工智能NLP-RAG-图像内容解析及检索优化）
import concurrent.futures as cf
import time
import urllib.request

URL = "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/resolve/master/model-00001-of-00002.safetensors"
CHUNK = 2 * 1024 * 1024  # 每线程 2MB

def pull(i: int) -> int:
    start = i * CHUNK
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start+CHUNK-1}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(r.read())

for workers in (4, 12):
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(pull, range(workers)))
    dt = time.time() - t0
    print(f"workers={workers}: {total/1024/1024:.1f}MB in {dt:.1f}s "
          f"= {total/dt/1024/1024:.2f} MB/s")
