# -*- coding: utf-8 -*-
"""部署/脚本/verify_config_env.py —— 环境前置检查（**只读**，不安装、不下载、不建索引）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用途：验收人/新机器上第一件事跑它，判断「能不能跑起来」以及「缺什么」。
本脚本只读取文件与探测端口，**不写任何文件**；结论逐项打印 ok/warn/FAIL 并给出退出码。

    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py
    pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py --no-hash   # 跳过 718MB 权重哈希

退出码：``0`` 全部必检项通过（允许 warn） / ``1`` 存在必检项失败。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import sysconfig
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = REPO_ROOT / "研发" / "data" / "index"
INDEX_DIR = INDEX_ROOT / "bge-m3_1024"
RAW_DIR = REPO_ROOT / "研发" / "data" / "raw"
MODELS_DIR = REPO_ROOT / "研发" / "data" / "models" / "chinese-clip-vit-base-patch16"
CLIP_WEIGHT = MODELS_DIR / "pytorch_model.bin"
CLIP_SHA256 = "3b8ca5a9ba36830e008bef8f980184cf29eeff986f1f3c1be9e5153f8712e7cc"
#: 本机实测字节数（T8 2026-10-07 复核）：771,304,190 B。
#: ⚠️ 早期转述的 753,177,983 B 是 **hf-mirror 上报的 Content-Length**，不是落盘体积（架构师本轮已更正该节）。
#: 结论纪律：**体积以本地实测为准，正确性以 sha256 判等**；故本脚本只按 sha256 判等、不按体积判等，
#: 避免拿镜像上报值当基准制造假失败。
CLIP_WEIGHT_BYTES_MEASURED = 771_304_190

REQUIRED_MODULES = ["fastapi", "uvicorn", "pydantic", "numpy", "faiss", "fitz", "jieba",
                    "sklearn", "requests", "openai"]

results: list[tuple[str, str, str]] = []          # (级别, 项, 说明)；级别 ∈ ok/warn/FAIL


def record(level: str, name: str, detail: str) -> None:
    results.append((level, name, detail))


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def module_version(name: str) -> str:
    import importlib.metadata as md

    for candidate in (name, "PyMuPDF" if name == "fitz" else name, "faiss-cpu" if name == "faiss" else name,
                      "scikit-learn" if name == "sklearn" else name):
        try:
            return md.version(candidate)
        except Exception:  # noqa: BLE001
            continue
    return "?"


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_python() -> None:
    purelib = sysconfig.get_paths().get("purelib", "?")
    record("ok", "Python 解释器", f"{sys.version.split()[0]} @ {sys.executable}")
    record("ok", "python 版本满足 >=3.10", f"{sys.version.split()[0]}")
    in_workorder1 = "工单1" in str(purelib)
    record("warn" if in_workorder1 else "ok", "site-packages 落点",
           f"{purelib}" + ("（**只读的工单1 venv**：禁止裸 pip install，"
                           "需要装包用 --target 研发/第三方库）" if in_workorder1 else ""))


def check_deps() -> None:
    missing = [m for m in REQUIRED_MODULES if not has_module(m)]
    pairs = ", ".join(f"{m} {module_version(m)}" for m in REQUIRED_MODULES if has_module(m))
    if missing:
        record("FAIL", "必需依赖", f"缺失：{missing}（禁止裸 pip install，见 部署/README.md §1）")
    else:
        record("ok", "必需依赖（{0} 个）".format(len(REQUIRED_MODULES)), pairs)
    # 图像文字识别类依赖：本机**缺失**，方法学限制已登记（不是失败项）
    alt = [m for m in ("pytesseract",) if has_module(m)]
    record("warn", "图像文字识别类依赖",
           f"pytesseract 已装：{bool(alt)}；深度学习文字识别包（paddle/easy 系列）与 opencv 均**未安装**"
           " → 位图内文字不走识别路线，改用像素级几何抽取 + 字形模板匹配，详见 设计/图像解析方案.md §9")
    for name, why in (("streamlit", "真 Streamlit 前端（缺则用纯标准库 Web 兜底）"),):
        record("ok" if has_module(name) else "warn", name, "已装" if has_module(name) else f"未装：{why}")


def check_corpus() -> None:
    for name in ("招股说明书1.pdf", "招股说明书2.pdf"):
        path = RAW_DIR / name
        if path.is_file():
            record("ok", f"语料 {name}", f"{path.stat().st_size:,} B")
        else:
            record("FAIL", f"语料 {name}", f"缺失：{path}（只读语料，禁止重新下载/替换）")


def check_index() -> None:
    manifest = INDEX_ROOT / "index_manifest.json"
    if not manifest.is_file():
        record("FAIL", "索引 manifest", f"缺失：{manifest}（先跑 研发/scripts/build_index.py --incremental --include-images）")
        return
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        record("FAIL", "索引 manifest", f"JSON 解析失败：{exc}")
        return
    record("ok", "索引 manifest",
           "chunk_total={0} by_modality={1} page_basis={2}".format(
               meta.get("chunk_total"), meta.get("chunk_by_modality"), meta.get("page_basis")))
    for name in ("vectors.npy", "ids.json", "faiss.index", "bm25.pkl", "chunks_indexed.jsonl"):
        path = INDEX_DIR / name
        level = "ok" if path.is_file() else "FAIL"
        record(level, f"索引文件 {name}", f"{path.stat().st_size:,} B" if path.is_file() else f"缺失：{path}")
    ids_file = INDEX_DIR / "ids.json"
    if ids_file.is_file():
        try:
            ids = json.loads(ids_file.read_text(encoding="utf-8")).get("ids", [])
            total = meta.get("chunk_total")
            record("ok" if len(ids) == total else "FAIL", "索引一致性",
                   f"ids={len(ids)} vs manifest.chunk_total={total}")
        except json.JSONDecodeError as exc:
            record("FAIL", "索引一致性", f"ids.json 解析失败：{exc}")


def check_ollama() -> None:
    if not tcp_open("127.0.0.1", 11434):
        record("FAIL", "Ollama 服务", "127.0.0.1:11434 不可连接（检索嵌入 + 生成都依赖它；"
                                      "启动方式：ollama serve；也可用 RAG_LLM__BACKEND=extractive 走无 LLM 兜底）")
        return
    record("ok", "Ollama 服务", "127.0.0.1:11434 可连接")
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name") for m in tags.get("models", [])]
    except Exception as exc:  # noqa: BLE001
        record("warn", "Ollama 模型清单", f"探测失败：{type(exc).__name__}: {exc}")
        return
    for want in ("bge-m3:latest", "qwen2.5:3b"):
        hit = any(str(n).startswith(want.split(":")[0]) for n in names)
        record("ok" if hit else "FAIL", f"Ollama 模型 {want}",
               "已就位" if hit else f"缺失（已装 {len(names)} 个模型）")


def check_clip(no_hash: bool) -> None:
    if not CLIP_WEIGHT.is_file():
        record("warn", "中文 CLIP 权重",
               f"缺失：{CLIP_WEIGHT}（图像语义通道不可用；框图结构/图表数值不受影响）。"
               f"下载：pwsh -NoProfile -File run_py.ps1 研发/scripts/download_clip.py")
        return
    size = CLIP_WEIGHT.stat().st_size
    if no_hash:
        record("warn", "中文 CLIP 权重",
               f"{size:,} B（--no-hash：未校验 sha256；本机实测基准 {CLIP_WEIGHT_BYTES_MEASURED:,} B）")
        return
    started = time.perf_counter()
    got = sha256_of(CLIP_WEIGHT)
    record("ok" if got == CLIP_SHA256 else "FAIL", "中文 CLIP 权重",
           f"{size:,} B sha256={got[:16]}… ({'一致' if got == CLIP_SHA256 else '与基准不一致'}) "
           f"校验耗时 {round((time.perf_counter() - started) * 1000)} ms")


def check_ports() -> None:
    for port, who in ((8123, "纯标准库中文 Web（研发/app/main.py --serve）"), (8600, "FastAPI 服务（部署/脚本/serve_api.py）")):
        busy = tcp_open("127.0.0.1", port, 0.6)
        record("warn" if busy else "ok", f"端口 {port}",
               ("已被占用（若为本项目服务可直接复用，否则换端口）" if busy else "空闲") + f" —— {who}")


def check_docker() -> None:
    import shutil
    import subprocess

    exe = shutil.which("docker")
    if not exe:
        record("warn", "Docker CLI", "未安装：Docker 为**可选**交付（见 部署/docker/），不参与运行")
        return
    try:
        out = subprocess.run([exe, "version", "--format", "{{.Client.Version}}"], capture_output=True,
                             text=True, timeout=20, encoding="utf-8", errors="replace")
        client = (out.stdout or "").strip() or "?"
        daemon_ok = "permission denied" not in (out.stderr or "") and out.returncode == 0
        record("ok", "Docker CLI", f"client {client} @ {exe}")
        record("warn" if not daemon_ok else "ok", "Docker 守护进程",
               "命名管道可访问" if daemon_ok else
               "**沙箱拒绝命名管道** npipe:////./pipe/dockerDesktopLinuxEngine —— 构建/运行需提权 "
               "sandbox_permissions=danger-full-access；本项不影响本机原生运行")
    except Exception as exc:  # noqa: BLE001
        record("warn", "Docker CLI", f"探测失败：{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工单4 环境前置检查（只读）")
    parser.add_argument("--no-hash", action="store_true", help="跳过 718MB CLIP 权重 sha256 校验")
    parser.add_argument("--json", action="store_true", help="额外输出机器可读 JSON")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("工单4 环境前置检查（只读；不改任何文件、不装依赖、不下权重、不建索引）")
    print("=" * 78)
    check_python()
    check_deps()
    check_corpus()
    check_index()
    check_ollama()
    check_clip(args.no_hash)
    check_ports()
    check_docker()

    n_fail = sum(1 for level, _, _ in results if level == "FAIL")
    n_warn = sum(1 for level, _, _ in results if level == "warn")
    print("")
    for level, name, detail in results:
        print(f"  [{level:^4}] {name} —— {detail}")
    print("")
    print(f"汇总：ok={len(results) - n_fail - n_warn} warn={n_warn} FAIL={n_fail}")
    if args.json:
        print(json.dumps({"checks": [{"level": lv, "name": n, "detail": d} for lv, n, d in results],
                          "n_fail": n_fail, "n_warn": n_warn}, ensure_ascii=False, indent=2))
    if n_fail:
        print("结论：存在必检项失败，退出码 1（先按上面 FAIL 项处理，再跑 部署/脚本/run_app.ps1 -HealthCheckOnly）")
        return 1
    print("结论：环境满足运行条件，退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
