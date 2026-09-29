"""补全官方BGE精排模型：仅下载必要文件，断点续传，SHA256校验，不下载重复格式。"""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPOSITORY = "BAAI/bge-reranker-base"
FILES = {"config.json", "model.safetensors", "sentencepiece.bpe.model",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(row, directory):
    name, size, expected = row["Name"], row["Size"], row["Sha256"]
    target = directory / name
    if target.exists() and target.stat().st_size == size and sha256(target) == expected:
        print(f"VERIFIED {name}", flush=True)
        return
    partial = directory / (name + ".partial")
    # 保留旧的不完整文件，完整新文件验证通过后才替换。
    url = f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo?Revision=master&FilePath={urllib.parse.quote(name)}"
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= size:
            partial.unlink()
            offset = 0
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=30) as response:
                resumed = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                mode = "ab" if resumed else "wb"
                written = offset if resumed else 0
                next_report = written + 64 * 1024 * 1024
                with partial.open(mode) as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                        written += len(block)
                        if written >= next_report:
                            print(f"DOWNLOADING {name}: {written * 100 / size:.1f}%", flush=True)
                            next_report = written + 64 * 1024 * 1024
            if partial.stat().st_size != size:
                raise RuntimeError("下载长度与官方清单不符")
            if sha256(partial) != expected:
                partial.unlink()
                raise RuntimeError("SHA256与官方清单不符")
            partial.replace(target)
            print(f"VERIFIED {name}", flush=True)
            return
        except Exception as error:
            print(f"RETRY {name}: {type(error).__name__}", flush=True)
            if attempt == 2:
                raise RuntimeError(f"{name} 未完成，可重新执行续传") from None
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只验证现有文件，不下载")
    args = parser.parse_args()
    directory = BASE / "models" / "bge-reranker-base"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "verified-manifest.json"
    if args.check:
        if not manifest.exists():
            raise SystemExit("MISSING: 没有通过校验的模型清单，请先运行不带--check的命令")
        rows = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for row in rows:
            path = directory / row["Name"]
            if not path.exists() or path.stat().st_size != row["Size"] or sha256(path) != row["Sha256"]:
                raise SystemExit(f"INVALID: {row['Name']}")
    else:
        url = f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo/files?Revision=master&Recursive=true"
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.load(response)
        rows = [row for row in metadata["Data"]["Files"] if row["Name"] in FILES]
        if {row["Name"] for row in rows} != FILES:
            raise SystemExit("官方仓库文件清单不完整")
        for row in rows:
            download_file(row, directory)
        manifest.write_text(json.dumps({"repository": REPOSITORY, "files": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RERANK_FILES_OK: 文件完整；仍需在Ubuntu加载模型做实际精排验收")


if __name__ == "__main__":
    main()
