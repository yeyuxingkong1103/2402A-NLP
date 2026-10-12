# 工单16：人工智能NLP-RAG-微调专用视觉语言模型
"""下载 Qwen2-VL-2B-Instruct 到 D:/Pycharm/yzq/models/

**为什么不走 HuggingFace/hf-mirror**：HF 直连超时；hf-mirror 的小文件能下，
但大权重走 LFS 会 302 到 `cas-server.xethub.hf.co`，实测被 401 挡掉
（只下到 8.5MB 就断）。ModelScope 在本机可达，且它的 REST 接口不需要额外装包，
所以直接按文件列表一个个拉，支持断点续传（已存在且大小一致就跳过）。
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "Qwen/Qwen2-VL-2B-Instruct"
DEST = r"D:\Pycharm\yzq\models\Qwen2-VL-2B-Instruct"
API = "https://www.modelscope.cn/api/v1/models/{repo}/repo"


def list_files(repo):
    url = f"{API.format(repo=repo)}/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    files = (data.get("Data") or {}).get("Files") or []
    return [(f["Path"], f.get("Size", 0)) for f in files if f.get("Type") == "blob"]


def download_file(repo, path, size, dest_root):
    out = os.path.join(dest_root, path.replace("/", os.sep))
    if os.path.exists(out) and os.path.getsize(out) == size:
        return "skip", 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{API.format(repo=repo)}?Revision=master&FilePath={urllib.parse.quote(path)}"
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "modelscope-dl/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    got = os.path.getsize(out)
    if got != size:
        raise RuntimeError(f"{path} 大小不符：期望 {size} 实得 {got}")
    return "ok", time.time() - t0


def main():
    print(f"[下载] {REPO}  ->  {DEST}", flush=True)
    files = list_files(REPO)
    total = sum(s for _, s in files)
    print(f"[清单] {len(files)} 个文件，合计 {total / 1e9:.2f} GB", flush=True)

    done = 0
    t0 = time.time()
    for i, (path, size) in enumerate(files, 1):
        try:
            status, el = download_file(REPO, path, size, DEST)
            done += size
            flag = "跳过" if status == "skip" else f"{el:5.1f}s"
            if size > 10_000_000 or i % 5 == 0:
                print(f"  [{i:2d}/{len(files)}] {flag}  {size / 1e6:8.1f}MB  {path}", flush=True)
        except Exception as e:
            print(f"  [失败] {path}: {type(e).__name__} {e}", flush=True)
            sys.exit(1)

    print(f"\n[完成] {done / 1e9:.2f} GB，耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
