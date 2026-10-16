# -*- coding: utf-8 -*-
"""工单4 中文 CLIP 权重下载（download_clip）。

任务：**人工智能NLP-RAG-图像内容解析及检索优化**。

设计依据：``设计/图像解析方案.md`` §5.1。

------------------------------------------------------------------------
实测坑（本机，2026-10，务必保留）
------------------------------------------------------------------------
1. ``huggingface_hub.snapshot_download`` 在**本沙箱会失败**：
   它要把缓存 blob 软链到 ``snapshots/<rev>/``，而 sandbox 在
   ``<repo>/tmpxxxx/dummy_file_src`` 上直接 ``PermissionError``。
   → 因此本脚本**不走 huggingface_hub**，改为**纯 urllib HTTP 下载**到目标目录。
2. 官方仓库里的 ``clip_cn_vit-b-16.pt``（753,196,934 B）是 **cn_clip 训练检查点**，
   内含 ``{epoch, name, state_dict, step}`` 包装层，**不是** HF 的 state dict；
   直接用 ``ChineseCLIPModel.from_pretrained`` 加载会**全部权重随机初始化**（实测已复现）。
   → 本脚本默认只下 ``pytorch_model.bin``（753,177,983 B），并做**包装层检测**：
     若 ``state_dict`` 在顶层，则判定文件不可用并报错（绝不静默用随机权重）。
3. 端点：``hf-mirror.com`` 实测 HTTP 200 可用；``www.modelscope.cn`` 作为备用端点。
   两个端点都失败时**如实报错退出**，由调用方关闭图像语义通道（框图/图表链路不受影响）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "OFA-Sys/chinese-clip-vit-base-patch16"
# 只要这四个就能跑 ChineseCLIPModel + ChineseCLIPProcessor
REQUIRED_FILES = ["config.json", "preprocessor_config.json", "vocab.txt", "pytorch_model.bin"]

HF_ENDPOINT = "https://hf-mirror.com"
MS_ENDPOINT = "https://www.modelscope.cn/api/v1/models/AI-ModelScope/chinese-clip-vit-base-patch16/repo"

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "models" / "chinese-clip-vit-base-patch16"


def _urls(filename: str, endpoint: str) -> list[str]:
    if endpoint == "modelscope":
        return ["%s?Revision=master&FilePath=%s" % (MS_ENDPOINT, filename)]
    base = os.environ.get("HF_ENDPOINT", HF_ENDPOINT).rstrip("/")
    return ["%s/%s/resolve/main/%s" % (base, REPO, filename)]


def _head_size(url: str, timeout: float = 20.0) -> int | None:
    """尝试用 HEAD 拿 Content-Length（拿不到返回 None，不视为失败）。"""

    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download_file(url: str, dest: Path, *, timeout: float = 120.0, retries: int = 4) -> tuple[bool, str]:
    """带 Range 续传的下载。返回 (是否成功, 说明)。"""

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    expect = _head_size(url)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if expect and have >= expect:
            break
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                with part.open(mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                dt = max(0.001, time.time() - t0)
                print("    [%s] %s -> %d B (%.2f MB/s)%s" % (
                    dest.name, url.split("/")[2], have, (dest.stat().st_size - 0 if False else have) / dt / 1e6,
                    "" if not expect else " / %d" % expect))
            break
        except Exception as exc:
            wait = min(20.0, 2.0 ** attempt)
            print("    [warn] %s 第 %d 次失败: %s: %s（%.0fs 后重试）"
                  % (dest.name, attempt, type(exc).__name__, str(exc)[:120], wait))
            time.sleep(wait)
    if not part.exists():
        return False, "未下载到任何字节"
    if expect and part.stat().st_size < expect:
        return False, "不完整：%d / %d 字节" % (part.stat().st_size, expect)
    os.replace(part, dest)
    return True, "%d 字节" % dest.stat().st_size


def verify_weights(path: Path) -> tuple[bool, str]:
    """检测权重文件是否为 HF 可直接加载的 state dict（拦截 cn_clip 包装层）。"""

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return True, "跳过校验（torch 不可用：%s）" % exc
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, "torch.load 失败：%s: %s" % (type(exc).__name__, exc)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return False, ("这是 cn_clip 训练检查点（顶层键=%s），不是 HF state dict；"
                       "直接加载会让全部权重随机初始化" % sorted(obj.keys())[:8])
    if isinstance(obj, dict) and "vision_model.embeddings.class_embedding" in obj:
        return True, "HF state dict（%d 个张量）" % len(obj)
    if isinstance(obj, dict):
        return True, "dict（%d 个键，首个=%s）" % (len(obj), list(obj)[:3])
    return False, "未知对象类型：%s" % type(obj).__name__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下载并校验中文 CLIP 权重（纯 HTTP，不用 huggingface_hub）")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--endpoint", choices=["hf-mirror", "modelscope"], default="hf-mirror")
    ap.add_argument("--files", default=",".join(REQUIRED_FILES))
    ap.add_argument("--skip-verify", action="store_true", help="跳过权重格式校验（不建议）")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    files = [f.strip() for f in args.files.split(",") if f.strip()]
    print("[download_clip] 目标目录 = %s" % dest)
    print("[download_clip] 端点 = %s" % args.endpoint)

    manifest: dict[str, object] = {"repo": REPO, "endpoint": args.endpoint, "files": {}, "started": time.time()}
    ok_all = True
    for name in files:
        target = dest / name
        if target.exists() and target.stat().st_size > 200:
            print("  [skip] %s 已存在（%d B）" % (name, target.stat().st_size))
            manifest["files"][name] = {"ok": True, "bytes": target.stat().st_size, "skipped": True}
            continue
        ok, info = False, ""
        for url in _urls(name, args.endpoint):
            print("  [get ] %s" % name)
            ok, info = download_file(url, target)
            if ok:
                break
        print("  [%s] %s : %s" % ("ok  " if ok else "fail", name, info))
        manifest["files"][name] = {"ok": ok, "info": info}
        ok_all = ok_all and ok

    # 权重格式校验（拦截 cn_clip 包装层）
    weight = dest / "pytorch_model.bin"
    if weight.exists() and not args.skip_verify:
        good, info = verify_weights(weight)
        print("[download_clip] 权重复核: %s —— %s" % ("通过" if good else "不通过", info))
        manifest["weight_check"] = {"ok": good, "info": info}
        ok_all = ok_all and good

    # 记录 sha256
    for name in files:
        p = dest / name
        if p.exists():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            manifest["files"].setdefault(name, {})["sha256"] = h.hexdigest()  # type: ignore[index]
            print("  sha256 %s = %s" % (name, h.hexdigest()))
    manifest["finished"] = time.time()
    (dest / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[download_clip] %s" % ("全部成功" if ok_all else "有失败项（见 download_manifest.json）"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
