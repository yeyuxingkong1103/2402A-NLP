#!/usr/bin/env python3
"""预取候选重排器权重（ModelScope）并逐个校验。

为什么单独一个脚本：换重排器是 P8 的对策，但"哪个能用/能不能加载"不该在实验中途才发现。
这里统一走 ModelScope（云端实测最快、最稳），每个模型独立 try/except，最后给一张表 +
退出码（**有任何一个失败就非 0**，避免"以为下好了其实没下"）。

用法（云端）：
    python scripts/fetch_rerankers.py --out-root /root/autodl-tmp/rerankers
    python scripts/fetch_rerankers.py --models BAAI/bge-reranker-large --check-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

#: 候选清单（P8：法条级精度）。含基线 bge-reranker-v2-m3 作对照。
DEFAULT_MODELS = (
    "BAAI/bge-reranker-large",                          # 中文/英文，比 base 强
    "BAAI/bge-reranker-base",                           # 更小更快
    "maidalun1020/bce-reranker-base_v1",                # 中文优化（网易有道 BCEmbedding）
    "jinaai/jina-reranker-v2-base-multilingual",        # 多语，CrossEncoder 兼容
    "Qwen/Qwen3-Reranker-0.6B",                         # 新一代，需自定义打分（非 CrossEncoder）
)

#: 我们真正需要的文件后缀（CrossEncoder 加载所需：权重 + 配置 + 分词器）。
#: ⚠️ ``.bin`` 也得在内：有些仓库只发 ``pytorch_model.bin`` 不发 safetensors ——
#: 漏了它就会把那个候选筛成"缺权重"（测试抓过这个 bug）。有 safetensors 时再优先排掉 .bin。
_WANT_SUFFIXES = (".safetensors", ".bin", ".json", ".txt", ".model", ".jinja", ".bpe")
#: 明确不要的路径片段（onnx 目录、样例文件等）
_SKIP_PARTS = ("onnx/", ".gitattributes", "README", "LICENSE", ".md", ".png", ".jpg")


def _select_files(files: list[dict], *, prefer_safetensors: bool = True) -> list[str]:
    """从仓库文件清单里挑出加载重排器所需的那几个（纯函数，可单测）。"""
    paths = [str(item.get("Path") or item.get("path") or "") for item in files if item]
    paths = [p for p in paths if p and not any(part in p for part in _SKIP_PARTS)]
    has_safetensors = any(p.endswith(".safetensors") for p in paths)
    picked: list[str] = []
    for path in paths:
        if not path.endswith(_WANT_SUFFIXES):
            continue
        if prefer_safetensors and has_safetensors and path.endswith(".bin"):
            continue                      # 有 safetensors 就别再下 .bin
        picked.append(path)
    return picked


def _download(model: str, *, out_root: str = "") -> tuple[bool, str]:
    """逐文件下载并拼成 **flat 目录**，返回该目录（CrossEncoder 直接就能加载）。

    为什么不用 ``snapshot_download``：实测 modelscope 1.40.1 上 ``ignore_file_pattern`` /
    ``allow_file_pattern`` **不生效** —— 它会把 ``onnx/model.onnx_data``（2.24 G）与
    ``pytorch_model.bin``（2.24 G）连同 ``model.safetensors`` 一起下下来，
    一个 2.2 G 的模型实占 ~6.7 G，五个候选直接撑爆 15 G 数据盘。
    改成"列清单 → 只下需要的 7 个文件 → 自己拼目录"后，占用 ≈ 权重本身。
    """
    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:  # noqa: BLE001
        return False, f"未安装 modelscope：{exc}"

    try:
        files = HubApi().get_model_files(model)
    except Exception as exc:  # noqa: BLE001 - 单个模型失败不该中断整批
        return False, f"列文件失败：{type(exc).__name__}: {exc}"

    wanted = _select_files(files)
    if not wanted:
        return False, "仓库里没找到可用文件（既无 .safetensors 也无 .bin）"

    root = Path(out_root or "rerankers") / "flat" / model.replace("/", "--")
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(out_root or "rerankers") / "_cache"
    failed: list[str] = []
    for path in wanted:
        target = root / Path(path).name
        if target.is_file() and target.stat().st_size > 0:
            continue                       # 断点续跑：已有就不重复下
        try:
            local = model_file_download(model_id=model, file_path=path, cache_dir=str(cache))
            # **移动而非复制**：modelscope 的缓存布局我们不需要，复制会让磁盘占用翻倍
            # （实测 5 个候选 ≈7 G 权重，复制就是 14 G，直接撑爆 11 G 余量）。
            try:
                shutil.move(str(local), str(target))
            except OSError:                    # 跨设备等情况下退回复制
                shutil.copy2(local, target)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{path}（{type(exc).__name__}: {str(exc)[:80]}）")
    if failed:
        return False, "以下文件下载失败：" + "；".join(failed)
    return True, str(root)


def _check_loadable(path: str) -> tuple[bool, str]:
    """能过 CrossEncoder 的加载（Qwen3-Reranker 这类非 CrossEncoder 会失败，属预期）。"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # noqa: BLE001
        return False, f"未安装 sentence-transformers：{exc}"
    try:
        CrossEncoder(path)
        return True, "CrossEncoder 可加载"
    except Exception as exc:  # noqa: BLE001
        return False, f"CrossEncoder 不可加载（{type(exc).__name__}: {str(exc)[:120]}）"


def main() -> int:
    parser = argparse.ArgumentParser(description="预取候选重排器")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--out-root", default="", help="下载到该目录下的 models/（空 = 用 ModelScope 缓存）")
    parser.add_argument("--check-only", action="store_true", help="只校验能否加载，不下载")
    parser.add_argument("--out", default="", help="结果 JSON 落盘路径")
    args = parser.parse_args()

    if args.out_root:
        import os

        os.environ["MODELSCOPE_CACHE"] = str(Path(args.out_root) / "modelscope")

    results: list[dict] = []
    for model in args.models:
        print(f"=== {model}")
        if args.check_only:
            ok, detail = True, "跳过下载（--check-only）"
        else:
            ok, detail = _download(model, out_root=args.out_root or "rerankers")
        entry = {"model": model, "mode": "check-only" if args.check_only else "download",
                 # check-only 时**不谎报**下载成功：downloaded 置 None
                 "downloaded": None if args.check_only else ok,
                 "path": detail if (ok and not args.check_only) else "",
                 "error": "" if ok else detail}
        if ok and not args.check_only and Path(detail).is_dir():
            loadable, why = _check_loadable(detail)
            entry["cross_encoder"] = loadable
            entry["load_detail"] = why
        print(f"    {'OK  ' if ok else 'FAIL'} {detail[:160]}")
        results.append(entry)

    print("\n=== 汇总 ===")
    for entry in results:
        flag = "OK  " if entry["downloaded"] is not False else "FAIL"
        extra = "" if entry.get("cross_encoder", True) else "（非 CrossEncoder，需自定义打分器）"
        if entry["mode"] == "check-only":
            extra = "（check-only：只校验了参数，没下载）" + extra
        print(f"  {flag} {entry['model']} {extra}")
        if entry.get("error"):
            print(f"       {entry['error'][:160]}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写：{target}")
    return 0 if all(entry["downloaded"] is not False for entry in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
