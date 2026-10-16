# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

部署优化真实测量脚本：``优化/脚本/测量部署优化指标.py``。

用途：把「镜像体积 / 冷启动 / 热路径 / 卷读写 / 并发 / 内存」六类指标的**真实数字**
     采集到 ``优化/评估结果/``，供 ``优化/部署优化说明.md`` 引用。

============================ 口径声明（禁止混淆）============================
* **实测值**：本机真实执行得到（服务启动耗时、HTTP 响应、卷读写、并发、内存、包体积）。
* **核算值**：由真实包体积/镜像层声明**加总**得到，标注 ``口径=核算``
  （本机 Docker daemon 被沙箱拒绝，无法 ``docker build`` 实测镜像体积，见
  ``部署/日志/docker提权失败证据.log``）。**绝不把核算值写成实测值。**
============================================================================

资源纪律（§共享约定 §5.5）：
* 单进程；**不使用** multiprocessing / ProcessPoolExecutor / joblib / Pool；
* 并发只用 ``ThreadPoolExecutor(max_workers≤8)`` 且全是等网络的 I/O 任务；
* 大目录/大文件流式处理，边跑边写；不把全部结果堆内存；
* 每个阶段打印进度并 append 到日志，便于中途观察。

用法（在 工单10/ 下，由「安全跑」包一层）::

    pwsh -NoProfile -File E:\\gao6gongdan\\_共享工具\\安全跑.ps1 `
         -WorkDir "E:\\gao6gongdan\\工单10" `
         -Command "pwsh -NoProfile -File run_py.ps1 优化/脚本/测量部署优化指标.py" `
         -LogName 工单10-优化测量

退出码：0 全部成功 / 1 部分指标缺失（仍会写出已采集到的结果）
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------- 路径与常量
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "优化" / "评估结果"
LOG_DIR = REPO_ROOT / "部署" / "日志"
VOL_ROOT = REPO_ROOT / "部署" / "卷"
PY = Path(r"E:\gao6gongdan\工单1\.venv\Scripts\python.exe")
VENV_SITE = Path(r"E:\gao6gongdan\工单1\.venv\Lib\site-packages")
CONDA_SITE = Path(r"E:\Anaconda\envs\fastapi_faq\Lib\site-packages")

API_BASE = os.environ.get("RAG_TEST_API_BASE", "http://127.0.0.1:8200").rstrip("/")
HEALTH_REPEAT = int(os.environ.get("OPT_HEALTH_REPEAT", "10"))
CONCURRENCY = int(os.environ.get("OPT_CONCURRENCY", "8"))
CONCURRENT_REQUESTS = int(os.environ.get("OPT_CONCURRENT_REQUESTS", "40"))
SESSION_FILES = int(os.environ.get("OPT_SESSION_FILES", "200"))

# 指标容器（边跑边填，最后统一落盘）
RESULT: dict[str, object] = {
    "脚本": "优化/脚本/测量部署优化指标.py",
    "工单编号": "人工智能NLP-RAG 项目-金融问答系统部署",
    "采集时间": time.strftime("%Y-%m-%d %H:%M:%S"),
    "口径说明": {
        "实测": "本机真实执行得到（HTTP/文件/进程；容器不可用，见 部署/日志/docker提权失败证据.log）",
        "核算": "由真实包体积与镜像层声明加总得到，非 docker build 实测",
    },
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "优化测量.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def http_get(path: str, timeout: float = 30.0) -> tuple[int, bytes, float]:
    """返回 (状态码, body, 耗时毫秒)。"""
    started = time.perf_counter()
    req = Request(API_BASE + path, headers={"user-agent": "gongdan10-optimize/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, body, round((time.perf_counter() - started) * 1000, 2)
    except URLError as exc:
        return 0, str(exc).encode(), round((time.perf_counter() - started) * 1000, 2)


def http_post(path: str, payload: dict, timeout: float = 180.0) -> tuple[int, dict, float]:
    import json as _json

    started = time.perf_counter()
    req = Request(API_BASE + path, data=_json.dumps(payload).encode("utf-8"),
                  headers={"content-type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read().decode("utf-8")), round(
                (time.perf_counter() - started) * 1000, 2)
    except URLError as exc:
        return 0, {"error": str(exc)}, round((time.perf_counter() - started) * 1000, 2)


# ================================================================== 1. 冷启动
def measure_cold_start() -> dict:
    """冷启动实测：在同一解释器里真实跑一次 warmup，分别计「关/开预热」的启动耗时。

    口径：与容器 ``uvicorn`` 启动时 ``Runtime.setup()`` 走的是**同一条代码路径**
    （``QAEngine`` + ``warmup``），因此可作为容器冷启动的可比数字。
    """
    log("阶段 1/6：冷启动实测（真实跑 QAEngine 加载索引，约 30~60 秒）")
    probe = r"""
import json, time, sys
mode = sys.argv[1]
t0 = time.perf_counter()
from app.core.config import get_config
from app.core.qa_engine import QAEngine
cfg = get_config()
t_import = time.perf_counter()
engine = QAEngine(cfg=cfg)
t_engine = time.perf_counter()
warm = {}
if mode == "warm":
    warm = engine.warmup(with_generate=False)
t_total = time.perf_counter()
print(json.dumps({
    "mode": mode,
    "import_s": round(t_import - t0, 3),
    "engine_init_s": round(t_engine - t_import, 3),
    "warmup_s": round(t_total - t_engine, 3),
    "total_s": round(t_total - t0, 3),
    "warmup_detail": warm,
}, ensure_ascii=False))
"""
    out: dict[str, object] = {}
    for mode in ("nowarm", "warm"):
        env = dict(os.environ)
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(REPO_ROOT / "研发"),
            "RAG_DATA__INDEX_DIR": str(VOL_ROOT / "index"),
            "RAG_LOG__DIR": str(VOL_ROOT / "logs"),
            "RAG_LLM__OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "DEPLOY_RUNTIME_DIR": str(VOL_ROOT / "runtime"),
            "DEPLOY_SESSION_DIR": str(VOL_ROOT / "sessions"),
        })
        started = time.perf_counter()
        proc = subprocess.run([str(PY), "-c", probe, mode], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=600,
                              cwd=str(REPO_ROOT / "研发"))
        wall = round(time.perf_counter() - started, 3)
        payload: dict = {"mode": mode, "wall_s": wall, "returncode": proc.returncode}
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                payload.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except json.JSONDecodeError:
                payload["parse_error"] = proc.stdout[-300:]
        else:
            payload["stderr"] = (proc.stderr or "")[-500:]
        out[mode] = payload
        log(f"  - {mode}: total={payload.get('total_s')}s wall={wall}s rc={proc.returncode}")

    nowarm = float(out.get("nowarm", {}).get("total_s") or 0)   # type: ignore[union-attr]
    warm = float(out.get("warm", {}).get("total_s") or 0)       # type: ignore[union-attr]
    out["对比"] = {
        "含义": "容器启动到『能服务』的耗时：关预热 = 只有 QAEngine 初始化；开预热 = 额外构建全文索引/实体标注",
        "预热代价_秒": round(warm - nowarm, 3) if warm and nowarm else None,
        "预热收益": "首请求不再承担 20 秒级全文索引构建；实测见『热路径』阶段",
        "容器策略": "HEALTHCHECK --start-period=90~150s，主镜像 DEPLOY_WARMUP=1 / WITH_GENERATE=0",
    }
    RESULT["冷启动"] = out
    return out


# ================================================================== 2. 热路径
def measure_hot_path() -> dict:
    """热路径实测：服务已在运行时，探针/健康检查/问答接口的响应耗时。"""
    log("阶段 2/6：热路径响应耗时")
    out: dict[str, object] = {}

    probes = {"/healthz": [], "/health": []}
    for path, bag in probes.items():
        for _ in range(HEALTH_REPEAT):
            code, body, ms = http_get(path)
            bag.append({"code": code, "ms": ms, "bytes": len(body)})
        ms_list = [x["ms"] for x in bag]
        out[path] = {
            "次数": len(ms_list), "全部200": all(x["code"] == 200 for x in bag),
            "最小ms": min(ms_list), "中位ms": round(statistics.median(ms_list), 2), "最大ms": max(ms_list),
            "响应字节": bag[0]["bytes"],
        }
        log(f"  - {path}: 中位 {out[path]['中位ms']} ms（{len(ms_list)} 次）")   # type: ignore[index]

    # 问答接口：3 个真实问题各 2 次（第二次大概率命中缓存）
    questions = [
        "平安银行2019年年度报告中，报告期末的资产总额是多少？",
        "招商银行2019年年度报告的营业收入是多少？",
    ]
    ask_rows = []
    for q in questions:
        for attempt in (1, 2):
            code, payload, ms = http_post("/api/v1/ask", {"question": q, "session_id": "opt-probe"})
            ask_rows.append({
                "问题": q[:24] + "…", "第几次": attempt, "code": code, "墙钟ms": ms,
                "服务端total_ms": payload.get("latency_ms", {}).get("total"),
                "retrieve_ms": payload.get("latency_ms", {}).get("retrieve"),
                "generate_ms": payload.get("latency_ms", {}).get("generate"),
                "命中缓存": (payload.get("cache") or {}).get("hit"),
                "生成方式": payload.get("generation_method"),
                "答案长度": len(payload.get("answer") or ""),
            })
            log(f"  - ask#{attempt}: {ms} ms（服务端 {payload.get('latency_ms', {}).get('total')} ms，"
                f"缓存={bool((payload.get('cache') or {}).get('hit'))}）")
    out["问答接口"] = ask_rows
    RESULT["热路径"] = out
    return out


# ================================================================== 3. 并发
def measure_concurrency() -> dict:
    log(f"阶段 3/6：并发探活（{CONCURRENCY} 线程 × {CONCURRENT_REQUESTS} 请求）")

    def one(_: int) -> dict:
        code, body, ms = http_get("/healthz", timeout=20)
        return {"code": code, "ms": ms}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        rows = [f.result() for f in as_completed([pool.submit(one, i) for i in range(CONCURRENT_REQUESTS)])]
    wall_ms = round((time.perf_counter() - started) * 1000, 2)
    lat = sorted(r["ms"] for r in rows)
    out = {
        "并发线程": CONCURRENCY, "请求数": CONCURRENT_REQUESTS,
        "失败数": sum(1 for r in rows if r["code"] != 200),
        "墙钟ms": wall_ms,
        "QPS": round(CONCURRENT_REQUESTS / (wall_ms / 1000), 2),
        "延迟ms_最小": lat[0], "延迟ms_中位": round(statistics.median(lat), 2),
        "延迟ms_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))], "延迟ms_最大": lat[-1],
    }
    RESULT["并发探活"] = out
    log(f"  - QPS={out['QPS']} 中位={out['延迟ms_中位']}ms p95={out['延迟ms_p95']}ms 失败={out['失败数']}")
    return out


# ================================================================== 4. 卷读写
def measure_volume_io() -> dict:
    """卷读写实测：会话文件（JSON，原子替换写）与索引卷随机读。"""
    log(f"阶段 4/6：卷读写实测（写 {SESSION_FILES} 个会话文件 + 索引卷读取）")
    out: dict[str, object] = {}

    session_dir = VOL_ROOT / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"session_id": "io-probe", "turns": [
        {"role": "assistant", "question": "平安银行资产总额？" * 3, "answer": "x" * 400,
         "cited_pages": [12, 13, 44], "trace_id": "io-probe", "ts": time.time()}]},
        ensure_ascii=False).encode("utf-8")
    started = time.perf_counter()
    nbytes = 0
    for i in range(SESSION_FILES):
        tmp = session_dir / f"io-probe-{i:04d}.json.tmp"
        final = session_dir / f"io-probe-{i:04d}.json"
        tmp.write_bytes(payload)
        os.replace(tmp, final)         # 与会话落盘完全一致的原子替换写
        nbytes += len(payload)
    write_ms = round((time.perf_counter() - started) * 1000, 2)

    started = time.perf_counter()
    read_total = 0
    for i in range(SESSION_FILES):
        read_total += len((session_dir / f"io-probe-{i:04d}.json").read_bytes())
    read_ms = round((time.perf_counter() - started) * 1000, 2)

    index_dir = VOL_ROOT / "index" / "bge-m3_1024"
    started = time.perf_counter()
    sizes = {}
    for name in ("vectors.npy", "faiss.index", "bm25.pkl", "chunks_indexed.jsonl", "ids.json"):
        p = index_dir / name
        sizes[name] = p.stat().st_size if p.exists() else 0
    stat_ms = round((time.perf_counter() - started) * 1000, 2)

    # 顺序全量读一次 vectors.npy（36 MB，流式分块，不整块进内存）
    vec = index_dir / "vectors.npy"
    started = time.perf_counter()
    read_bytes = 0
    if vec.exists():
        with vec.open("rb") as fh:
            while True:
                chunk = fh.read(4 * 1024 * 1024)      # 4MB 分块
                if not chunk:
                    break
                read_bytes += len(chunk)
    seq_ms = round((time.perf_counter() - started) * 1000, 2)

    # 清理自己写的探针文件（不留垃圾在卷里）
    removed = 0
    for i in range(SESSION_FILES):
        f = session_dir / f"io-probe-{i:04d}.json"
        if f.exists():
            f.unlink()
            removed += 1

    out = {
        "会话写（原子替换）": {"文件数": SESSION_FILES, "总字节": nbytes, "总耗时ms": write_ms,
                        "单文件均ms": round(write_ms / SESSION_FILES, 3),
                        "吞吐MB/s": round(nbytes / 1024 / 1024 / (write_ms / 1000), 2) if write_ms else None},
        "会话读": {"文件数": SESSION_FILES, "总字节": read_total, "总耗时ms": read_ms,
                 "单文件均ms": round(read_ms / SESSION_FILES, 3),
                 "吞吐MB/s": round(read_total / 1024 / 1024 / (read_ms / 1000), 2) if read_ms else None},
        "索引卷stat": {"耗时ms": stat_ms, "文件大小": sizes, "合计MB": round(sum(sizes.values()) / 1024 / 1024, 2)},
        "索引顺序读": {"文件": "vectors.npy", "字节": read_bytes, "耗时ms": seq_ms,
                   "吞吐MB/s": round(read_bytes / 1024 / 1024 / (seq_ms / 1000), 2) if seq_ms else None},
        "探针清理": {"删除文件数": removed},
    }
    RESULT["卷读写"] = out
    log(f"  - 会话写 {out['会话写（原子替换）']['单文件均ms']} ms/文件；"      # type: ignore[index]
        f"顺序读 {out['索引顺序读']['吞吐MB/s']} MB/s")                        # type: ignore[index]
    return out


# ================================================================== 5. 依赖体积
def dir_size_mb(path: Path, *, cap_files: int = 200000) -> float:
    """流式统计目录体积（跳过 __pycache__，逐文件累加后立即释放，不把清单堆内存）。"""
    if not path.exists():
        return 0.0
    total = 0
    count = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in {"__pycache__", ".git"}:
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                            count += 1
                            if count >= cap_files:
                                return round(total / 1024 / 1024, 2)
                    except OSError:
                        continue
        except OSError:
            continue
    return round(total / 1024 / 1024, 2)


def find_package_dir(name: str) -> Path | None:
    """在 venv/conda 两个 site-packages 里找包目录（本机 venv 是 --system-site-packages 型）。"""
    import glob

    for site in (VENV_SITE, CONDA_SITE):
        if not site.exists():
            continue
        for cand in (site / name, site / name.replace("-", "_")):
            if cand.is_dir():
                return cand
        hits = [Path(p) for p in glob.glob(str(site / f"{name.replace('-', '_')}*"))]
        for hit in hits:
            if hit.is_dir() and (hit / "__init__.py").exists():
                return hit
    return None


#: 参与核算的包（真实存在的包目录体积；- 表示本机没有该包）
PKG_SETS = {
    "本方案（精简闭包）": ["numpy", "faiss_cpu", "fastapi", "uvicorn", "fitz", "pymupdf",
                          "jieba", "redis", "starlette", "pydantic", "anyio"],
    "若在容器内跑本地大模型（被否决方案）": ["torch", "transformers", "sentence_transformers",
                                            "open_clip", "huggingface_hub", "tokenizers",
                                            "safetensors", "accelerate"],
}


def measure_package_sizes() -> dict:
    log("阶段 5/6：依赖体积核算（真实包目录体积，非 docker build 实测）")
    dists: dict[str, float] = {}
    missing: dict[str, list[str]] = {}
    for label, names in PKG_SETS.items():
        total = 0.0
        miss = []
        detail = {}
        for name in names:
            # 优先用 importlib.metadata 找到发行包位置，再用目录体积
            path = find_package_dir(name)
            size = dir_size_mb(path) if path else 0.0
            if size == 0.0:
                miss.append(name)
            detail[name] = {"目录": str(path) if path else None, "MB": size}
            total += size
        dists[label] = round(total, 2)
        missing[label] = miss
        log(f"  - {label}: 合计 {round(total, 2)} MB（缺失/未安装：{miss}）")

    # 版本信息（真实记录，供复核）
    versions = {}
    for dist in ("numpy", "faiss-cpu", "fastapi", "uvicorn", "pymupdf", "jieba", "redis",
                 "torch", "transformers"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None

    out = {
        "口径": "核算（各包在 site-packages 内的真实磁盘体积之和，非 docker build 实测）",
        "包集合合计MB": dists,
        "明细": {k: v for k, v in PKG_SETS.items()},
        "未安装包": missing,
        "包版本": versions,
        "体积优化结论": {
            "被剔除的本地大模型依赖MB": dists.get("若在容器内跑本地大模型（被否决方案）", 0),
            "保留的精简闭包MB": dists.get("本方案（精简闭包）", 0),
            "说明": "剔除 torch/transformers/open_clip 等，改为调用宿主机 Ollama 提供嵌入与生成",
        },
    }
    RESULT["依赖体积核算"] = out
    return out


# ================================================================== 6. 内存
def measure_memory() -> dict:
    """内存实测：系统可用/提交 + 本进程峰值；服务进程内存不可得时如实说明。"""
    log("阶段 6/6：内存占用实测")
    out: dict[str, object] = {}

    def counter(paths: list[str]) -> dict[str, float]:
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "& { $o=@{}; " + " ".join(
                     f"$o['{p}']=(Get-Counter '{p}').CounterSamples.CookedValue;" for p in paths) +
                 " $o | ConvertTo-Json -Compress }"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            if proc.returncode == 0 and proc.stdout.strip():
                return {k: round(float(v), 2) for k, v in json.loads(proc.stdout).items()}
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
            out["计数器错误"] = f"{type(exc).__name__}: {exc}"
        return {}

    sys_mem = counter([r"\Memory\Available MBytes", r"\Memory\Committed Bytes", r"\Memory\Commit Limit"])
    out["系统内存MB"] = sys_mem
    if sys_mem.get("\\Memory\\Commit Limit"):
        out["提交占比%"] = round(
            sys_mem["\\Memory\\Committed Bytes"] / sys_mem["\\Memory\\Commit Limit"] * 100, 2)

    # 哪个进程在监听 8200：**先确认端口真的可连**（实测：Get-NetTCPConnection 在沙箱里
    # 会漏报监听中的端口），再据此取进程内存。psutil 未安装 → 用 PowerShell 查。
    import socket as _socket

    listening = False
    try:
        with _socket.create_connection(("127.0.0.1", int(os.environ.get("OPT_API_PORT", "8200"))), timeout=3):
            listening = True
    except OSError as exc:
        out["API进程内存"] = {"note": f"8200 端口不可连（{type(exc).__name__}），无法读取服务进程内存"}
    if listening:
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "$c=Get-NetTCPConnection -LocalPort 8200 -State Listen -EA SilentlyContinue;"
                 "if($c){$p=Get-Process -Id $c[0].OwningProcess;"
                 "$o=[ordered]@{pid=$p.Id;name=$p.ProcessName;ws_mb=[math]::Round($p.WorkingSet64/1MB,1);"
                 "priv_mb=[math]::Round($p.PrivateMemorySize64/1MB,1)};$o|ConvertTo-Json -Compress}"
                 "else{'{}'}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            text = (proc.stdout or "").strip()
            if text and text != "{}":
                out["API进程内存"] = json.loads(text)
            else:
                # Get-NetTCPConnection 漏报时的兜底：按命令行特征找 uvicorn 进程
                proc2 = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-Process -Name python -EA SilentlyContinue | Sort-Object WorkingSet64 -Descending |"
                     "Select-Object -First 3 | ForEach-Object {"
                     "[ordered]@{pid=$_.Id;ws_mb=[math]::Round($_.WorkingSet64/1MB,1);"
                     "priv_mb=[math]::Round($_.PrivateMemorySize64/1MB,1)}} | ConvertTo-Json -Compress"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                out["API进程内存"] = {
                    "note": "Get-NetTCPConnection 未报告监听进程（沙箱下实测会漏报）；"
                            "以下为内存占用最大的 3 个 python 进程（含 API 与其子进程）",
                    "top_python": json.loads(proc2.stdout) if (proc2.stdout or "").strip() else [],
                }
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            out["API进程内存"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 本脚本自身峰值（Windows 无标准库 API → 用 PowerShell 读当前进程）
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$p=Get-Process -Id {os.getpid()};"
             "[ordered]@{ws_mb=[math]::Round($p.WorkingSet64/1MB,1);"
             "priv_mb=[math]::Round($p.PrivateMemorySize64/1MB,1)}|ConvertTo-Json -Compress"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if proc.stdout.strip():
            out["测量脚本自身内存"] = json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        out["测量脚本自身内存"] = {"error": f"{type(exc).__name__}: {exc}"}

    RESULT["内存"] = out
    log(f"  - 系统可用 {sys_mem.get(chr(92) + 'Memory' + chr(92) + 'Available MBytes')} MB；"
        f"提交占比 {out.get('提交占比%')}%；API 进程 {out.get('API进程内存')}")
    return out


# ================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description="工单10 部署优化指标测量")
    ap.add_argument("--skip-cold", action="store_true", help="跳过冷启动实测（耗时 60 秒以上）")
    ap.add_argument("--only", default="", help="只跑指定阶段：cold/hot/conc/vol/pkg/mem，逗号分隔")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("工单10 部署优化指标测量开始")
    log(f"API = {API_BASE}；输出 = {OUT_DIR}")
    log("=" * 70)

    code, body, ms = http_get("/healthz", timeout=5)
    RESULT["服务可达性"] = {"api": API_BASE, "healthz_code": code, "healthz_ms": ms,
                          "说明": "0 表示服务未启动；热路径/并发指标的『实测』口径依赖服务在跑"}

    stages = {s.strip() for s in args.only.split(",") if s.strip()}
    want = (lambda name: (not stages) or (name in stages))

    if want("cold") and not args.skip_cold:
        measure_cold_start()
    if want("hot") and code == 200:
        measure_hot_path()
    if want("conc") and code == 200:
        measure_concurrency()
    if want("vol"):
        measure_volume_io()
    if want("pkg"):
        measure_package_sizes()
    if want("mem"):
        measure_memory()

    # —— 落盘 ——
    # ⚠️ 实测教训：用 `--only mem` 之类的小范围重跑时，若直接覆盖 JSON 会把之前采集的
    #    阶段全部冲掉（本工单真踩过：重测内存后冷启动/热路径/并发/卷读写凭空消失）。
    #    因此这里做**阶段合并**：读入既有结果，只覆盖本次真正采集到的阶段。
    json_path = OUT_DIR / "部署优化实测.json"
    existing: dict[str, Any] = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged = dict(existing)
    merged.update({k: v for k, v in RESULT.items() if v not in (None, {}, [])})
    merged["本次运行时间"] = RESULT["采集时间"]
    merged["本次运行阶段"] = [k for k in ("冷启动", "热路径", "并发探活", "卷读写", "依赖体积核算", "内存")
                          if k in RESULT and RESULT[k]]
    merged["口径说明"] = RESULT["口径说明"]
    merged["脚本"] = RESULT["脚本"]
    merged["工单编号"] = RESULT["工单编号"]
    json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写出 {json_path}（本次阶段：{merged['本次运行阶段']}；已与历史结果合并，不丢旧阶段）")
    RESULT.clear()
    RESULT.update(merged)

    csv_path = OUT_DIR / "优化前后对比.csv"
    rows = [["类别", "指标", "优化前", "优化后", "口径"]]
    cold = RESULT.get("冷启动", {})                                # type: ignore[assignment]
    if isinstance(cold, dict) and cold:
        rows.append(["镜像体积", "本地大模型依赖体积(MB)",
                     RESULT.get("依赖体积核算", {}).get("被剔除的本地大模型依赖MB", ""),  # type: ignore[union-attr]
                     "0（改为宿主机 Ollama）", "核算"])
        rows.append(["镜像体积", "精简依赖闭包体积(MB)", "", 
                     RESULT.get("依赖体积核算", {}).get("包集合合计MB", {}).get("本方案（精简闭包）", ""),  # type: ignore[union-attr]
                     "核算"])
        rows.append(["冷启动", "QAEngine 初始化(s)", cold.get("nowarm", {}).get("engine_init_s", ""),
                     cold.get("warm", {}).get("engine_init_s", ""), "实测"])
        rows.append(["冷启动", "启动到可服务总耗时(s)", cold.get("nowarm", {}).get("total_s", ""),
                     cold.get("warm", {}).get("total_s", ""), "实测"])
    hot = RESULT.get("热路径", {})                                  # type: ignore[assignment]
    if isinstance(hot, dict) and hot:
        for ep in ("/healthz", "/health"):
            if ep in hot:
                rows.append(["热路径", f"{ep} 中位响应(ms)", "", hot[ep]["中位ms"], "实测"])   # type: ignore[index]
        asks = hot.get("问答接口") or []
        if asks:
            rows.append(["热路径", "问答接口首答(ms)", "", asks[0]["墙钟ms"], "实测"])
            if len(asks) > 1:
                rows.append(["热路径", "问答接口第二次(ms)", "", asks[1]["墙钟ms"], "实测"])
    conc = RESULT.get("并发探活", {})                                # type: ignore[assignment]
    if isinstance(conc, dict) and conc:
        rows.append(["并发", "健康探针 QPS", "", conc.get("QPS"), "实测"])
        rows.append(["并发", "健康探针 p95(ms)", "", conc.get("延迟ms_p95"), "实测"])
    vol = RESULT.get("卷读写", {})                                   # type: ignore[assignment]
    if isinstance(vol, dict) and vol:
        rows.append(["卷读写", "会话文件写(ms/文件)", "", vol["会话写（原子替换）"]["单文件均ms"], "实测"])  # type: ignore[index]
        rows.append(["卷读写", "索引顺序读(MB/s)", "", vol["索引顺序读"]["吞吐MB/s"], "实测"])              # type: ignore[index]
    mem = RESULT.get("内存", {})                                     # type: ignore[assignment]
    if isinstance(mem, dict) and mem:
        rows.append(["资源限制", "系统提交内存占比(%)", "", mem.get("提交占比%"), "实测"])
        api_mem = mem.get("API进程内存") or {}
        if isinstance(api_mem, dict) and api_mem.get("ws_mb"):
            rows.append(["资源限制", "API 进程工作集(MB)", "", api_mem.get("ws_mb"),
                         f"实测（限额参考 {os.environ.get('API_MEMORY', '2g')}）"])
    import csv

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    log(f"已写出 {csv_path}")

    missing = [k for k in ("冷启动", "热路径", "并发探活", "卷读写", "依赖体积核算", "内存")
               if k not in RESULT]
    if missing:
        log(f"[注意] 以下阶段未采集（服务未启动或显式跳过）：{missing}")
        return 1
    log("全部阶段采集完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
