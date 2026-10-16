# -*- coding: utf-8 -*-
"""工单7：下载中文 CLIP 权重（**优先走 Ollama 同类本地推理路径，权重落本仓库**）。

工单编号：人工智能NLP-RAG-功能测试及评估

背景（实测更正，2026-10-08）
--------------------------
本仓库最初沿用了「沙箱封禁外网出口、pip 会挂死」的**旧结论**，于是把「图像内容解析」
当作已知缺口（索引里 image 块 = 0）。经 captain 重新探测，该结论**是错的**：

    * DNS / TCP 443 全部正常；
    * **Python 走 OpenSSL 3.5.6 + certifi** 对 baidu / pypi.org / 清华源 / 阿里源 /
      hf-mirror / github **全部 TLS 握手成功且 HTTP 200**；
    * `pip index versions <pkg>` 秒回；
    * 失败的是 `curl.exe` 与 PowerShell 的 `Invoke-WebRequest` —— 报
      ``schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS``，
      即 **Windows 系统 TLS 凭证（schannel）不可用**，与「沙箱封网」无关。
      Python/pip 走 OpenSSL，不经过 schannel，因此畅通。

于是本工单恢复**原计划设计**：补齐中文 CLIP 图像语义编码能力。

权重放哪（为什么不是 Ollama）
---------------------------
任务要求「模型优先下载到 Ollama」，实测说明为什么**不能**只放 Ollama：
    * Ollama 的 `chinese-clip`/`clip` 类模型导出的是**文本塔**，能力标签是 ``embedding``，
      只能给文本出向量；本工单需要的是**图像塔**（把图片编码成向量做跨模态检索），
      Ollama 目前不提供图像塔服务，`POST /api/embed` 对图像输入也不可用；
    * 因此图像塔必须由进程内的 `ChineseCLIPModel` 加载 —— 这**正是工单6 原设计**
      （``研发/app/core/multimodal_encoder.py``：`DEFAULT_MODEL_DIR = 研发/data/models/chinese-clip-vit-base-patch16`）。
    * 下载端点仍**优先用镜像**（hf-mirror.com），等价于"优先用本地/就近源"，
      不回源 huggingface.co；文本侧推理继续走 Ollama（bge-m3 / qwen2.5）。

用法：
    python 测试/下载中文CLIP权重.py                # 下载到 研发/data/models/chinese-clip-vit-base-patch16
    python 测试/下载中文CLIP权重.py --check-only   # 只体检现有权重，不下载

退出码：0 成功或已完整；1 下载失败；2 参数错误。
"""

from __future__ import annotations

import argparse
import hashlib
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "研发" / "data" / "models" / "chinese-clip-vit-base-patch16"

#: 镜像优先（实测 200，速度可用）；回源 huggingface.co 作为兜底
ENDPOINTS: tuple[str, ...] = (
    "https://hf-mirror.com",
    "https://huggingface.co",
)
REPO_ID = "OFA-Sys/chinese-clip-vit-base-patch16"

#: 与 ``multimodal_encoder.REQUIRED_FILES`` 对齐。
#: 注意（实测）：``OFA-Sys/chinese-clip-vit-base-patch16`` **没有** ``tokenizer_config.json`` /
#: ``special_tokens_map.json``（两者 404），它的 tokenizer 由 ``vocab.txt`` +
#: ``preprocessor_config.json`` 驱动 —— 所以不要把不存在的文件写进必需清单，
#: 否则脚本会以「下载失败」收尾（第一版就是这样，白报两个失败）。
FILES: tuple[str, ...] = (
    "config.json",
    "preprocessor_config.json",
    "vocab.txt",
    "pytorch_model.bin",
)


def _human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def _remote_size(url: str, timeout: float = 20.0) -> tuple[int | None, bool]:
    """探测远端大小与可达性：返回 ``(字节数或 None, 是否可达)``。

    实测坑（2026-10-08）：HF 镜像的 ``resolve`` 会 302 到 ``/api/resolve-cache/...``，
    且响应**不带 Content-Length**（HEAD 与 GET 都不带）。
    因此**不能**用「Content-Length 是否为数字」来判断可达性 —— 那样会把可下载的端点误判为不可达
    （本脚本第一版就是这样，输出「所有端点都不可达」）。
    正确做法：HTTP 200 即视为可达；大小改由 ``Range: bytes=0-0`` 读 ``Content-Range`` 取得，
    拿不到就退化为「大小未知」（下载时不校验总长，只校验非空）。
    """
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gd7-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.headers.get("Content-Length")
            if raw and raw.isdigit():
                return int(raw), True
    except (urllib.error.URLError, OSError, ValueError):
        return None, False
    # Content-Length 缺失 → 用 Range 试探总长（不支持 Range 也无妨，视为未知）
    probe = urllib.request.Request(url, headers={"User-Agent": "gd7-downloader/1.0",
                                                 "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as resp:
            cr = resp.headers.get("Content-Range") or ""
            if "/" in cr:
                tail = cr.rsplit("/", 1)[1].strip()
                if tail.isdigit():
                    return int(tail), True
            raw = resp.headers.get("Content-Length")
            return (int(raw) if raw and raw.isdigit() else None), True
    except urllib.error.HTTPError as exc:
        # 416/501：端点可达但不支持 Range → 大小未知，仍算可达
        return None, exc.code in (416, 501)
    except (urllib.error.URLError, OSError, ValueError):
        return None, True      # HEAD 已 200，视为可达


def download_file(url: str, dest: Path, *, expect: int | None, timeout: float = 60.0) -> tuple[bool, str]:
    """带进度与大小校验的下载（先写 .part，校验通过再改名，避免半成品被当成完整权重）。"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gd7-downloader/1.0"})
    started = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            total = expect or (int(resp.headers["Content-Length"])
                               if str(resp.headers.get("Content-Length", "")).isdigit() else 0)
            last = 0.0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                got += len(block)
                now = time.time()
                if now - last > 3.0 or got == total:
                    last = now
                    pct = f"{got * 100 / total:5.1f}%" if total else "  ?  "
                    speed = got / max(now - started, 1e-6) / 1024 / 1024
                    print(f"      {pct} {_human(got)}/{_human(total) if total else '?'}  {speed:.2f} MB/s",
                          flush=True)
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"
    if expect is not None and got != expect:
        tmp.unlink(missing_ok=True)
        return False, f"大小不符：期望 {expect}，实际 {got}"
    tmp.replace(dest)
    sha = hashlib.sha256(dest.read_bytes()[:1 << 20]).hexdigest()[:16] if got else ""
    return True, f"{_human(got)}（首 1MB sha256 前缀 {sha}，用时 {time.time() - started:.1f}s）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单7：{WORK_ORDER} —— 下载中文 CLIP 权重")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help=f"权重目录（默认 {DEFAULT_DEST}）")
    parser.add_argument("--check-only", action="store_true", help="只体检，不下载")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    print("=" * 78)
    print(f"工单7 中文 CLIP 权重下载｜{WORK_ORDER}")
    print(f"仓库  ：{REPO_ID}")
    print(f"落盘  ：{dest}")
    print(f"端点  ：{' → '.join(ENDPOINTS)}（镜像优先）")
    print("=" * 78)

    # ---- 体检 ----
    missing = [f for f in FILES if not (dest / f).is_file()]
    if args.check_only:
        print(f"现有文件：{sorted(p.name for p in dest.glob('*')) if dest.is_dir() else '（目录不存在）'}")
        print("✅ 完整" if not missing else f"❌ 缺少 {missing}")
        return 0 if not missing else 1

    # ---- 选端点 ----
    base = ""
    for ep in ENDPOINTS:
        url = f"{ep}/{REPO_ID}/resolve/main/config.json"
        _size, reachable = _remote_size(url, timeout=15)
        if reachable:
            base = ep
            print(f"[端点] 可用：{ep}")
            break
    if not base:
        print("❌ 所有端点都不可达（请检查网络或代理）", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in FILES:
        target = dest / name
        url = f"{base}/{REPO_ID}/resolve/main/{name}"
        expect, _reachable = _remote_size(url)
        if target.is_file() and not args.force:
            cur = target.stat().st_size
            if expect is None or cur == expect:
                print(f"  = 已存在，跳过：{name}（{_human(cur)}）")
                continue
            print(f"  ! 大小不符（本地 {_human(cur)} / 远端 {_human(expect)}）→ 重新下载：{name}")
        print(f"  ↓ {name}（远端 {_human(expect) if expect else '大小未知'}）", flush=True)
        ok, info = download_file(url, target, expect=expect)
        print(f"      {'✅' if ok else '❌'} {info}")
        if not ok:
            failed.append(name)

    missing = [f for f in FILES if not (dest / f).is_file()]
    print("-" * 78)
    if failed or missing:
        print(f"❌ 未完成：下载失败 {failed or '无'}；仍缺 {missing or '无'}", file=sys.stderr)
        return 1
    total = sum((dest / f).stat().st_size for f in FILES)
    print(f"✅ 权重就绪：{dest}（{len(FILES)} 个文件 / {_human(total)}）")
    print("   下一步：")
    print("     1) 安装图像链路依赖：pip install --target 研发/第三方库 opencv-python-headless")
    print("     2) 抽取图像块：  python 研发/scripts/extract_images.py --all")
    print("     3) 重建索引（并入图像块）：见 部署/脚本/一键跑测试评估.ps1 的 index 步骤")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已中断]", file=sys.stderr)
        raise SystemExit(130)
