#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoDL 算力云「一键（幂等）部署」脚本：SGLang 为主 / vLLM 为备。

事实基线（**不要凭猜测改这里的路径或版本**）
================================================
模型与卡点取自 ``handoff/CLOUD-ENV-FACTS.md``（用户 2026-09-20 提供的实例记录）；
**磁盘/环境位置在 t129 按实例实测更正过**（见下），**脚本里不再写死磁盘数字**：

* 模型：``Qwen3.8-27B-FP8``（ModelScope），约 **29 GB**；
* 真实路径：``/root/autodl-tmp/models/models/Qwen--Qwen3.8-27B-FP8/snapshots/master/``；
* 推荐软链：``/root/autodl-tmp/models/qwen27b`` -> 真实路径；
* **磁盘（2026-09-20 实测，别再照抄旧数字）**：系统盘 ``/`` 30 GB 中约 **27 GB 可用**；
  数据盘 ``/root/autodl-tmp`` 50 GB 中约 **22 GB 可用**（旧文档写的"仅剩 ~8 GB"已过期）。
  -> 本脚本**默认绝不重下权重**（先解析再用），需要下载时**现场用 ``shutil.disk_usage``
  读盘**再报"所需/可用"，**不写死容量**；
* 驱动 570.124.04（CUDA 12.8 驱动，环境内 CUDA 12.9）；
* **conda 环境位置自适应（t129）**：``which conda`` = ``/root/miniconda3/bin/conda``，
  但 ``conda env list`` 里**没有名为 ``rag`` 的环境**（被删过）。所以前缀按
  「``--env-prefix`` -> 探测 ``conda env list`` 里的 ``rag`` -> ``conda create -n rag``
  的默认前缀 ``/root/miniconda3/envs/rag``（**系统盘**）」三级决定，并在横幅里打印
  **实际使用的前缀与来源** —— 数据盘那点余量要留给模型与编译/KV 缓存；
* 原组合 ``sglang 0.5.19 + flashinfer-python 0.6.18`` **已知会炸**：
  ``nvcc fatal : Unknown option '--compress-mode=size'`` —— flashinfer 0.6.18 的 JIT
  硬编码了只有 CUDA 13.0 才认的选项，而实例是 CUDA 12.9。

本脚本做什么
------------
按「环境 -> 依赖 -> 引擎（含 flashinfer 处置）-> 模型解析与校验 -> 配置注入 ->
启动 -> 自检」的顺序，把上表里的事实固化成**可重复执行**的步骤：

* **幂等**：每一步先探测再动手；已满足就打印 ``SKIP``，不产生副作用；重复跑结果一致；
* ``--dry-run``：只打印**将要执行**的命令，不碰系统（连续两次输出逐字一致）；
* ``--engine sglang|vllm``：两条路都可走，SGLang 起不来时切 vLLM 的等价步骤写进手册；
* **模型优先**：解析软链 -> 真实路径 -> 校验 ``config.json`` + 权重分片完整性
  （**读 safetensors 头并核对文件长度**，不是"目录存在就算有"）；缺失才走 ModelScope，
  且**先打印所需空间、等确认**（空间判据**只作用于真的要走下载这一条腿**）；
* **配置以实际模型为准**：模型名/上下文长度从解析到的模型目录读，注入项目 ``.env``
  （``LLM_PROVIDER=openai_compat`` + ``OPENAI_COMPAT_BASE_URL=http://<host>:30000/v1``），
  **不另写一套模型名**；拓扑 A（云上只跑引擎）用 ``--skip-inject`` 跳过这一步。

诚实边界（**必须保留这句话**）
------------------------------
本脚本在**开发机上只验到**：Python 语法/导入、``--dry-run`` 计划输出、``--dry-run`` 两次
输出逐字一致（幂等）、``.sh`` 的结构自检。开发机实测有 RTX 2060(6GB)/驱动 577.00，但
**没有 CUDA 工具链（``nvcc`` 不可用）、也不是 Linux** -> **SGLang / vLLM 的真机启动与
性能未经验证**，真机证据必须由**实例侧**产出（命令与预期输出见 ``docs/CLOUD-DEPLOY.md``）。
脚本里的版本组合是**依据事实基线的固定组合 + 显式 pin**，不是"已验证可跑"。

用法（在实例上）
----------------
::

    python3 scripts/deploy_cloud.py --dry-run          # 先看它打算做什么（不改系统）
    python3 scripts/deploy_cloud.py --engine sglang    # 真跑：重建环境 + 装依赖 + 起服务
    python3 scripts/deploy_cloud.py --selfcheck        # 健康检查 + bench_serving 三档压测
    python3 scripts/deploy_cloud.py --diagnose         # 失败时的诊断清单（原样可复制）
    bash scripts/deploy_cloud.sh --help                # 薄入口，等价于本脚本

退出码：0 全部完成；10 参数/环境不满足；20 环境重建失败；30 依赖安装失败；
40 模型缺失或校验不通过；50 配置注入失败；60 启动失败；70 自检不达标。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 一、事实基线常量（改这里 = 改事实，必须同步 handoff/CLOUD-ENV-FACTS.md）
# ---------------------------------------------------------------------------
AUTODL_ROOT = Path("/root/autodl-tmp")
MODEL_ROOT = AUTODL_ROOT / "models"
MODEL_LINK = MODEL_ROOT / "qwen27b"
MODEL_REAL_DIR = MODEL_ROOT / "models" / "Qwen--Qwen3.8-27B-FP8" / "snapshots" / "master"
MODEL_EXPECT_BYTES = 29 * 1024 ** 3            # 约 29 GB（事实基线）
MODEL_MIN_BYTES = 20 * 1024 ** 3               # 低于这个数一律视为"不完整"
MODELSCOPE_MODEL_ID = "Qwen/Qwen3.8-27B-FP8"   # 与真实路径里的 Qwen--Qwen3.8-27B-FP8 对应
DISK_SAFETY_BYTES = 2 * 1024 ** 3              # 下载后至少还要留 2 GB

CONDA_ENV_NAME = "rag"
#: ``conda create -n rag`` 会用的默认前缀（= ``<conda 根>/envs/rag``）。
#: t129 更正：**系统盘**上的这个位置才是默认落点 —— 数据盘 22 GB 要留给模型与
#: 编译/KV 缓存，而原来的 ``/root/autodl-tmp/conda_envs/rag`` 从来没被验证过。
#: 运行时还会用 :func:`conda_default_prefix()` 按**实际的 conda**（``CONDA_EXE`` /
#: ``which conda``）再推导一次。
CONDA_HOME_DEFAULT = Path("/root/miniconda3")
CONDA_ENV_PREFIX = CONDA_HOME_DEFAULT / "envs" / "rag"
CONDA_PYTHON = CONDA_ENV_PREFIX / "bin" / "python"     # 名义默认值（实际以解析结果为准）
CONDA_PY_VERSION = "3.10"


def conda_default_prefix() -> Path:
    """``conda create -n rag`` 的默认前缀：``<conda 根>/envs/rag``。

    从 ``CONDA_EXE`` / ``which conda`` / ``/root/miniconda3/bin/conda`` 依次找 conda，
    认 ``<根>/bin/conda`` 这种布局并推出 ``<根>/envs/rag``；认不出来就用
    :data:`CONDA_ENV_PREFIX`（同样是系统盘）。
    """
    candidates = [str(os.environ.get("CONDA_EXE", "") or ""), shutil.which("conda") or "",
                  "/root/miniconda3/bin/conda"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.name == "conda" and path.parent.name == "bin":
            return path.parent.parent / "envs" / CONDA_ENV_NAME
    return CONDA_ENV_PREFIX

TORCH_PINS = ("torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu129"
SGLANG_KERNEL_INDEX_URL = "https://docs.sglang.ai/whl/cu129/"

#: 卡点对症：flashinfer-python 0.6.18 的 JIT 硬编码 CUDA 13.0 选项（实例是 CUDA 12.9）。
#: 固定组合 = **装完 sglang 后把 flashinfer 降到 SGLang 自己升级过的那个版本**
#: （SGLang 仓库有提交 ``chore: upgrade flashinfer 0.2.14.post1``），
#: 用 ``--no-deps`` 覆盖，避免 pip 又把 0.6.18 拉回来。
FLASHINFER_PIN = "0.2.14.post1"
#: ``flashinfer-cubin``（预编译内核）**只有 >= 0.4.0**（PyPI 实测），与 0.2.14.post1 不配对
#: -> 两条路二选一，别混装。用 ``--flashinfer-mode cubin`` 会改走这条路。
FLASHINFER_CUBIN_MIN = "0.4.0"

#: 原文已登记**无效**的四种试法（别再重复，脚本故意不提供这些开关）：
#: 1) FLASHINFER_DISABLE_JIT=1  2) --cuda-graph-backend-prefill=disabled
#: 3) --attention-backend triton --sampling-backend pytorch  4) 清 ~/.cache/flashinfer
INEFFECTIVE_FIXES = (
    "export FLASHINFER_DISABLE_JIT=1",
    "--cuda-graph-backend-prefill=disabled",
    "--attention-backend triton --sampling-backend pytorch",
    "rm -rf ~/.cache/flashinfer",
)

DEFAULT_PORT = 30000
DEFAULT_HOST_BIND = "0.0.0.0"
DEFAULT_OMP_THREADS = "8"
LOG_DIR = AUTODL_ROOT / "logs"
SERVER_LOG = LOG_DIR / "llm-server.log"
SERVER_PID = LOG_DIR / "llm-server.pid"
DEPLOY_LOG = LOG_DIR / "deploy-cloud.log"
BENCH_DIR = LOG_DIR / "bench"

BENCH_CONCURRENCY = (1, 4, 8)          # 与验收口径一起固化：并发 1/4/8
BENCH_INPUT_LEN = 1024                 # 输入 1k
BENCH_OUTPUT_LEN = 128                 # 输出 128
BENCH_PROMPTS_PER_CONC = 5             # 官方建议 num-prompts >= 5 * max-concurrency
TTFT_LIMIT_MS = 500.0                  # 及格线（以实测为准，不写死版本号）
ITL_LIMIT_MS = 50.0

ENV_BLOCK_BEGIN = "# >>> legal-rag cloud deploy (auto-generated) >>>"
ENV_BLOCK_END = "# <<< legal-rag cloud deploy (auto-generated) <<<"

# 退出码
EXIT_OK = 0
EXIT_USAGE = 10
EXIT_ENV = 20
EXIT_DEPS = 30
EXIT_MODEL = 40
EXIT_CONFIG = 50
EXIT_START = 60
EXIT_SELFCHECK = 70


def gbk_safe(text: str) -> str:
    """把要打印的文本转成 GBK 安全形式（控制台是 GBK 时不会因为一个字符丢整行）。"""
    out = []
    for char in text:
        try:
            char.encode("gbk")
            out.append(char)
        except UnicodeEncodeError:
            out.append("?")
    return "".join(out)


class Log:
    """极简日志：同时打屏 + 追加进 ``logs/deploy-cloud.log``（dry-run 不落盘）。"""

    def __init__(self, path: Path | None, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._fh = None
        if enabled and path is not None and str(path) != "-":
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = path.open("a", encoding="utf-8")
            except OSError:
                self._fh = None

    def __call__(self, message: str = "") -> None:
        line = gbk_safe(message)
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class Ctx:
    """一次部署执行的上下文（所有类都从这里拿参数，避免到处传一堆裸变量）。"""

    project_dir: Path
    dry_run: bool = False
    engine: str = "sglang"
    model_path: str = ""
    skip_model_download: bool = False
    force_recreate_env: bool = False
    flashinfer_mode: str = "pin"       # pin | cubin | none
    host: str = "127.0.0.1"            # 注入配置用的对外地址（127.0.0.1 = 同机网页）
    port: int = DEFAULT_PORT
    omp_threads: str = DEFAULT_OMP_THREADS
    assume_yes: bool = False
    sglang_version: str = ""           # 空 = 不 pin，装 pip 解析到的版本（版本会写进横幅）
    vllm_version: str = ""
    #: conda 环境前缀：空 = **先探测**（``conda env list`` 里名为 ``rag`` 的那条），
    #: 探测不到才回落到 ``CONDA_ENV_PREFIX``。实例的数据盘只剩 ~8 GB，环境很可能在系统盘，
    #: 所以这里**不能写死**（t129）。
    env_prefix: str = ""
    #: 日志目录：空 = 默认 ``/root/autodl-tmp/logs``（维持现状）。
    log_dir: str = ""
    #: 引擎监听地址：默认 ``0.0.0.0``（维持现状）；SSH 隧道场景设 ``127.0.0.1``。
    bind_host: str = DEFAULT_HOST_BIND
    #: 跳过第 5 步配置注入（拓扑 A：实例上只跑引擎、没有本项目）。
    skip_inject: bool = False
    log: Log = field(default_factory=lambda: Log(None, enabled=False))

    # ---------- 派生路径（一律走这里，别再引用模块级常量，否则开关不生效）----------
    def log_root(self) -> Path:
        """日志目录（``--log-dir`` 优先，默认 ``/root/autodl-tmp/logs``）。"""
        return Path(self.log_dir).expanduser() if self.log_dir else LOG_DIR

    def server_log(self) -> Path:
        return self.log_root() / "llm-server.log"

    def server_pid(self) -> Path:
        return self.log_root() / "llm-server.pid"

    def deploy_log(self) -> Path:
        return self.log_root() / "deploy-cloud.log"

    def bench_dir(self) -> Path:
        return self.log_root() / "bench"

    # ---------- 执行原语 ----------
    def run(self, cmd: list[str], *, check: bool = True, capture: bool = False,
            env: dict | None = None, cwd: Path | None = None) -> tuple[int, str]:
        """执行一条命令；dry-run 只打印。返回 (退出码, stdout)。"""
        pretty = " ".join(cmd)
        if self.dry_run:
            self.log(f"[DRY] {pretty}")
            return 0, ""
        merged = dict(os.environ)
        if env:
            merged.update(env)
        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=capture, text=True, env=merged,
                                  cwd=str(cwd) if cwd else None)
        except FileNotFoundError:
            self.log(f"[FAIL] 找不到命令：{cmd[0]}")
            return 127, ""
        out = (proc.stdout or "") if capture else ""
        self.log(f"[{'OK' if proc.returncode == 0 else 'FAIL'}] {pretty}"
                 f"（{time.time() - started:.1f}s，退出码 {proc.returncode}）")
        if out and capture:
            for line in out.strip().splitlines()[:20]:
                self.log("    | " + line)
        if check and proc.returncode != 0:
            raise SystemExit(proc.returncode)
        return proc.returncode, out

    def probe(self, cmd: list[str]) -> str:
        """探测型命令：dry-run 时**不执行**（保证 dry-run 不碰系统、输出可复现）。"""
        if self.dry_run:
            return ""
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except (FileNotFoundError, OSError):
            return ""
        return (proc.stdout or "").strip() if proc.returncode == 0 else ""


# ---------------------------------------------------------------------------
# 二、环境：conda env ``rag``（事实基线说它已被删除，要重建）
# ---------------------------------------------------------------------------
class CondaEnv:
    """conda 环境：**存在即复用**，除非显式重建。

    前缀的选法（t129，实例数据盘只剩 ~8 GB -> 环境很可能在系统盘，不能写死）：

    1. ``--env-prefix`` 显式给出 -> 用它（不存在的目录才去建）；
    2. 否则**先探测** ``conda env list`` 里名为 ``rag`` 的那条（解析出 prefix 就用它）；
    3. 都拿不到 -> 回落事实基线里的 ``/root/autodl-tmp/conda_envs/rag``。
    """

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self._prefix: Path | None = None
        self._origin = ""
        self._discovered = False          # 是否来自 `conda env list`（是 -> 直接复用）

    # ---------- 前缀解析 ----------
    def origin(self) -> str:
        """前缀是怎么定下来的（给人看的一句话；横幅与 SKIP 提示都用它）。"""
        self.prefix()
        return self._origin

    def prefix(self) -> Path:
        if self._prefix is None:
            self._prefix = self._resolve_prefix()
        return self._prefix

    def _resolve_prefix(self) -> Path:
        ctx = self.ctx
        if ctx.env_prefix:
            self._origin = "--env-prefix 指定"
            return Path(ctx.env_prefix).expanduser()
        found = self.discover()
        if found is not None:
            self._discovered = True
            self._origin = "conda env list 里发现的环境（复用，不重建）"
            return found
        fallback = conda_default_prefix()
        self._origin = ("默认前缀（--env-prefix 未给，且 conda env list 里没有名为 "
                        f"{CONDA_ENV_NAME} 的环境）-> `conda create -n {CONDA_ENV_NAME}` "
                        "的默认落点（系统盘，数据盘留给模型与缓存）")
        return fallback

    def discover(self) -> Path | None:
        """在 ``conda env list`` 里找名为 ``rag`` 的环境，返回它的 prefix（找不到返回 None）。

        解析规则：跳过 ``#`` 头与空行；每行取「环境名 + prefix（最后一个以 ``/`` 开头的段）」；
        环境名等于 ``rag`` **或** prefix 的 basename 等于 ``rag`` 都算命中
        （用 ``-p`` 建的 prefix 环境在列表里可能只显示路径）。拿不到列表就不猜。
        """
        if self.ctx.dry_run:
            return None                   # dry-run 不探测：输出必须可复现、也不碰系统
        out = self.ctx.probe([*self.conda_cmd(), "env", "list"])
        if not out:
            return None
        for raw in out.splitlines():
            line = raw.strip().lstrip("*").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1 and line.startswith("/"):
                # 只显示路径的写法（`-p` 建的 prefix 环境）
                if Path(line).name == CONDA_ENV_NAME:
                    return Path(line)
                continue
            if len(parts) < 2:
                continue
            name, candidate = parts[0], parts[-1]
            if not candidate.startswith("/"):
                continue
            if name == CONDA_ENV_NAME or Path(candidate).name == CONDA_ENV_NAME:
                return Path(candidate)
        return None

    # ---------- 基本操作 ----------
    def python(self) -> Path:
        return self.prefix() / "bin" / "python"

    def exists(self) -> bool:
        if self.ctx.dry_run:
            return False                     # dry-run 走"计划创建"分支，保证输出可复现
        if self._discovered:
            return True                      # conda 自己列出来的环境 -> 直接复用
        return self.python().is_file()

    def version(self) -> str:
        return self.ctx.probe([str(self.python()), "-V"])

    def conda_cmd(self) -> list[str]:
        found = shutil.which("conda") if not self.ctx.dry_run else "/root/miniconda3/bin/conda"
        return [found or "conda"]

    def ensure(self) -> None:
        ctx = self.ctx
        prefix = self.prefix()
        ctx.log("")
        ctx.log("== 第 1 步：conda 环境 ==")
        ctx.log(f"[INFO] 环境前缀：{prefix}（来源：{self.origin()}）")
        if self.exists() and not ctx.force_recreate_env:
            ctx.log(f"[SKIP] 复用 conda 环境：{prefix}（{self.version() or '未探测到版本'}）"
                    f" —— 幂等：不重建（要重建加 --force-recreate-env）")
            return
        if self.exists() and ctx.force_recreate_env:
            ctx.log(f"[ACTION] --force-recreate-env：删除 {prefix} 后重建"
                    f"（模型与数据盘其它目录**不动**）")
            ctx.run([*self.conda_cmd(), "env", "remove", "-p", str(prefix), "-y"])
        ctx.log(f"[ACTION] 创建 conda 环境 {CONDA_ENV_NAME}（python={CONDA_PY_VERSION}）"
                f" -> {prefix}")
        ctx.run([*self.conda_cmd(), "create", "-p", str(prefix),
                 f"python={CONDA_PY_VERSION}", "-y"])
        if not ctx.dry_run:
            ctx.run([str(self.python()), "-m", "pip", "install", "--upgrade", "pip", "wheel"])


# ---------------------------------------------------------------------------
# 三、依赖：pip 安装（幂等 = 版本已满足就不动）
# ---------------------------------------------------------------------------
class Deps:
    """PyTorch / sglang-kernel / 基础工具。每个包都先查已装版本，再决定装不装。"""

    def __init__(self, ctx: Ctx, env: CondaEnv) -> None:
        self.ctx = ctx
        self.env = env

    def installed_version(self, package: str) -> str:
        out = self.ctx.probe([str(self.env.python()), "-c",
                              "import importlib.metadata as m;"
                              f"print(m.version({package!r}))"])
        return out.strip()

    def ensure_python_tools(self) -> None:
        ctx = self.ctx
        ctx.log("")
        ctx.log("== 第 2 步：依赖 ==")
        for package, version in (("modelscope", ""), ("huggingface-hub", ""), ("uv", "")):
            current = self.installed_version(package)
            if current:
                ctx.log(f"[SKIP] {package} 已安装 {current}")
                continue
            spec = f"{package}=={version}" if version else package
            ctx.log(f"[ACTION] pip install {spec}")
            ctx.run([str(self.env.python()), "-m", "pip", "install", spec])

    def ensure_torch(self) -> None:
        ctx = self.ctx
        pins = TORCH_PINS
        versions = {pkg.split("==")[0]: pkg.split("==")[1] for pkg in pins}
        current = {name: self.installed_version(name) for name in versions}
        if all(current[name] == want for name, want in versions.items()):
            ctx.log(f"[SKIP] torch 组合已满足：{current}")
            return
        ctx.log(f"[ACTION] 安装 PyTorch 组合 {list(pins)}（索引 {TORCH_INDEX_URL}）"
                f"；当前 = {current}")
        ctx.run([str(self.env.python()), "-m", "pip", "install", *pins,
                 "--index-url", TORCH_INDEX_URL])

    def ensure_sglang_kernel(self) -> None:
        ctx = self.ctx
        current = self.installed_version("sglang-kernel")
        if current:
            ctx.log(f"[SKIP] sglang-kernel 已安装 {current}")
            return
        ctx.log(f"[ACTION] 安装 sglang-kernel（索引 {SGLANG_KERNEL_INDEX_URL}）")
        ctx.run([str(self.env.python()), "-m", "pip", "install", "sglang-kernel",
                 "--index-url", SGLANG_KERNEL_INDEX_URL])


# ---------------------------------------------------------------------------
# 四、引擎安装：SGLang / vLLM + flashinfer 对症处置
# ---------------------------------------------------------------------------
class EngineInstaller:
    """装推理引擎，并**对症**处理 flashinfer/CUDA 12.9 卡点。

    固定组合（写死在脚本里、可在启动横幅里回溯）：

    * SGLang：``pip install --pre sglang``（版本记为实际装到的那个，不写死）；
    * flashinfer：装完 sglang 后 ``pip install --no-deps flashinfer-python==0.2.14.post1``
      —— 这是 SGLang 自己升级过的版本（提交 ``chore: upgrade flashinfer 0.2.14.post1``），
      用来替换会触发 ``nvcc fatal : Unknown option '--compress-mode=size'`` 的 0.6.18；
    * 备选 ``--flashinfer-mode cubin``：装 ``flashinfer-cubin`` 预编译内核走"绕过 JIT"路线
      （注意 cubin 只有 >= 0.4.0，与 0.2.14.post1 不配对，**两者不可混装**）；
    * ``--flashinfer-mode none``：不装 flashinfer（彻底绕开，SGLang 若 import 失败则回退 vLLM）。
    """

    def __init__(self, ctx: Ctx, env: CondaEnv, deps: Deps) -> None:
        self.ctx = ctx
        self.env = env
        self.deps = deps

    # ---------- 工具 ----------
    def _pip(self, *args: str) -> None:
        self.ctx.run([str(self.env.python()), "-m", "pip", *args])

    def _pin_available(self, package: str, version: str) -> bool:
        """确认 pin 的版本在索引里真的存在（不猜）；拿不到版本列表时**不阻塞**只提示。"""
        out = self.ctx.probe([str(self.env.python()), "-m", "pip", "index", "versions",
                              package])
        if not out:
            self.ctx.log(f"[注意] 拿不到 {package} 的可用版本列表（pip index 不可用/离线），"
                         f"直接按 pin 安装：{package}=={version}")
            return True
        match = re.search(r"Available versions:\s*(.+)", out)
        if not match:
            return True
        available = [item.strip() for item in match.group(1).split(",")]
        if version in available:
            self.ctx.log(f"[OK] pin 可用：{package}=={version}")
            return True
        self.ctx.log(f"[注意] pin 不在索引里：{package}=={version}；"
                     f"索引里有：{', '.join(available[:8])} ...")
        return False

    # ---------- SGLang ----------
    def install_sglang(self) -> None:
        ctx = self.ctx
        spec = f"sglang=={ctx.sglang_version}" if ctx.sglang_version else "sglang"
        current = self.deps.installed_version("sglang")
        if current and (not ctx.sglang_version or current == ctx.sglang_version):
            ctx.log(f"[SKIP] sglang 已安装 {current}"
                    + ("（未指定 --sglang-version，不重装）" if not ctx.sglang_version else ""))
        else:
            if ctx.sglang_version:
                self._pin_available("sglang", ctx.sglang_version)
            ctx.log(f"[ACTION] pip install --pre {spec}（当前 = {current or '未安装'}）")
            self._pip("install", "--pre", spec)
        self.handle_flashinfer()
        self.ensure_transformers_note()

    def ensure_transformers_note(self) -> None:
        version = self.deps.installed_version("transformers")
        self.ctx.log(f"[INFO] transformers = {version or '未知'}"
                     f"（SGLang 会强制它自己的版本区间；冲突时以 pip 解析结果为准）")

    # ---------- flashinfer 卡点 ----------
    def handle_flashinfer(self) -> None:
        """对症卡点：把 JIT 会炸的 flashinfer 换成固定组合（或走 cubin / 彻底不装）。"""
        ctx = self.ctx
        current = self.deps.installed_version("flashinfer-python")
        mode = ctx.flashinfer_mode
        if mode == "none":
            if not current:
                ctx.log("[SKIP] flashinfer-python 未安装（--flashinfer-mode none，彻底绕开）")
                return
            ctx.log(f"[ACTION] 卸载 flashinfer-python {current}"
                    f"（--flashinfer-mode none：不走 JIT，绕开卡点）")
            self._pip("uninstall", "-y", "flashinfer-python")
            return
        if mode == "cubin":
            cubin = self.deps.installed_version("flashinfer-cubin")
            ctx.log("[注意] cubin 路线：用**预编译内核**绕开 JIT。"
                    f"PyPI 上 flashinfer-cubin 只有 >= {FLASHINFER_CUBIN_MIN}，"
                    f"与 flashinfer-python=={FLASHINFER_PIN} 不配对 -> 不要混装。")
            if cubin:
                ctx.log(f"[SKIP] flashinfer-cubin 已安装 {cubin}")
            else:
                ctx.log("[ACTION] pip install flashinfer-cubin（版本交给 pip 解析，"
                        "装完记得把版本写进证据）")
                self._pip("install", "flashinfer-cubin")
            return
        # 默认 pin 路线
        if current == FLASHINFER_PIN:
            ctx.log(f"[SKIP] flashinfer-python 已是固定组合 {current}")
            return
        if current and current.startswith("0.6."):
            ctx.log(f"[ACTION] 命中已知卡点组合：flashinfer-python {current}"
                    f"（JIT 硬编码 CUDA 13.0 选项 -> nvcc fatal）"
                    f"，替换为 {FLASHINFER_PIN}")
        else:
            ctx.log(f"[ACTION] pin flashinfer-python=={FLASHINFER_PIN}"
                    f"（当前 = {current or '未安装'}）")
        self._pin_available("flashinfer-python", FLASHINFER_PIN)
        self._pip("install", "--no-deps", f"flashinfer-python=={FLASHINFER_PIN}")
        ctx.log("[INFO] 已避开事实基线里登记**无效**的四种试法，别再去试：")
        for item in INEFFECTIVE_FIXES:
            ctx.log("       - " + item)

    # ---------- vLLM（备选，对 FP8 + CUDA 12.9 更成熟）----------
    def install_vllm(self) -> None:
        ctx = self.ctx
        spec = f"vllm=={ctx.vllm_version}" if ctx.vllm_version else "vllm"
        current = self.deps.installed_version("vllm")
        if current and (not ctx.vllm_version or current == ctx.vllm_version):
            ctx.log(f"[SKIP] vllm 已安装 {current}")
        else:
            if ctx.vllm_version:
                self._pin_available("vllm", ctx.vllm_version)
            ctx.log(f"[ACTION] pip install {spec}（当前 = {current or '未安装'}）"
                    "；vLLM 自带 FlashInfer/注意力后端，通常**不需要**单独折腾 flashinfer")
            self._pip("install", spec)
        self.handle_flashinfer()

    def install(self) -> None:
        ctx = self.ctx
        ctx.log("")
        ctx.log(f"== 第 3 步：推理引擎（{ctx.engine}）==")
        if ctx.engine == "sglang":
            self.install_sglang()
        else:
            self.install_vllm()


# ---------------------------------------------------------------------------
# 五、模型解析与校验（默认不重下；重下必须先报空间并确认）
# ---------------------------------------------------------------------------
@dataclass
class ModelInfo:
    path: Path
    name: str
    max_ctx: int | None
    total_bytes: int
    ok: bool
    source: str
    notes: list[str] = field(default_factory=list)


def safetensors_check(path: Path) -> tuple[bool, int, str]:
    """校验单个 ``*.safetensors``：读头（前 8 字节长度 + JSON 头）并核对文件长度。

    这是"不能只看目录存在"的核心：只 stat 大小看不出一半的分片被截断，
    而读头能同时验证**结构可解析**与**数据区完整**。
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(8)
            if len(head) != 8:
                return False, 0, "文件不足 8 字节（header 长度都读不到）"
            (header_len,) = struct.unpack("<Q", head)
            if not 0 < header_len <= 128 * 1024 * 1024:
                return False, 0, f"header 长度异常：{header_len}"
            raw = fh.read(header_len)
            if len(raw) != header_len:
                return False, 0, "header 未读完（文件被截断）"
    except OSError as exc:
        return False, 0, f"读不了：{type(exc).__name__}: {exc}"
    try:
        header = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return False, 0, f"header 不是合法 JSON：{type(exc).__name__}: {exc}"
    if not isinstance(header, dict):
        return False, 0, "header 不是对象"
    header.pop("__metadata__", None)
    end = 0
    for tensor, spec in header.items():
        if not isinstance(spec, dict) or "data_offsets" not in spec:
            return False, 0, f"张量 {tensor} 缺 data_offsets"
        begin, stop = spec["data_offsets"]
        if not (isinstance(begin, int) and isinstance(stop, int)) or stop < begin:
            return False, 0, f"张量 {tensor} 的 data_offsets 非法：{spec['data_offsets']}"
        end = max(end, stop)
    expected = 8 + header_len + end
    if size < expected:
        return False, expected, (f"文件被截断：按 header 至少要 {expected} 字节，"
                                 f"实际 {size} 字节")
    return True, expected, f"{len(header)} 个张量，{size / 1024 ** 3:.2f} GiB"


class ModelResolver:
    """模型**解析优先**：软链 -> 真实路径 -> 完整性校验；缺了才谈下载（且先要人确认）。"""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx

    # ---------- 解析 ----------
    def candidates(self) -> list[tuple[str, Path]]:
        ctx = self.ctx
        items: list[tuple[str, Path]] = []
        if ctx.model_path:
            items.append(("--model-path", Path(ctx.model_path).expanduser()))
        items.append((f"软链 {MODEL_LINK}", MODEL_LINK))
        items.append(("事实基线真实路径", MODEL_REAL_DIR))
        if MODEL_ROOT.is_dir() or ctx.dry_run:
            items.append(("models 目录下按名找", MODEL_ROOT / "models"))
        return items

    def resolve(self) -> ModelInfo | None:
        ctx = self.ctx
        ctx.log("")
        ctx.log("== 第 4 步：模型解析与完整性校验（默认不重下）==")
        seen: set[str] = set()
        for label, path in self.candidates():
            resolved = path
            if not ctx.dry_run:
                try:
                    resolved = path.resolve() if path.exists() else path
                except OSError:
                    resolved = path
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if ctx.dry_run:
                ctx.log(f"[PLAN] 解析 {label}：{path}"
                        f"（dry-run 不落盘探测；真实执行时按软链 -> 真实路径顺序找）")
                continue
            info = self.inspect(label, resolved)
            if info is not None and info.ok:
                ctx.log(f"[OK] 用 {label}：{info.path}")
                ctx.log(f"     模型名={info.name}  上下文长度={info.max_ctx or '未声明'}"
                        f"  权重合计={info.total_bytes / 1024 ** 3:.2f} GiB")
                for note in info.notes:
                    ctx.log("     - " + note)
                return info
            if info is not None:
                ctx.log(f"[注意] {label} 不合格：{'；'.join(info.notes[:3])}")
        if ctx.dry_run:
            return None
        info = self.handle_missing()
        if info is None or not info.ok:
            ctx.log("[FAIL] 下载后校验仍未通过：不要把不完整的目录当模型用。"
                    "请人工核对 --model-path，或删掉半成品目录后重跑（可续传）。")
            raise SystemExit(EXIT_MODEL)
        return info

    def inspect(self, label: str, path: Path) -> ModelInfo | None:
        """检查一个候选目录：``config.json`` 可解析 + 权重分片逐个校验 + tokenizer 齐全。

        判定**不看提示文本**，只看四条硬条件（文本是给人看的，条件才决定成败）：
        ① config.json 存在且能解析；② 权重分片全都在且逐个通过头校验（不是"目录存在"）；
        ③ tokenizer 至少有一个；④ 权重合计不低于下限。
        """
        if not path.is_dir():
            return None
        notes: list[str] = []
        config_path = path / "config.json"
        config: dict = {}
        config_ok = config_path.is_file()
        if not config_ok:
            notes.append(f"缺 config.json（{path} 不像模型目录）")
        else:
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                config_ok = False
                notes.append(f"config.json 解析失败：{type(exc).__name__}: {exc}")
        shards, shard_notes, weights_ok = self.check_weights(path)
        notes.extend(shard_notes)
        tokenizer_ok = any((path / name).is_file() for name in (
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model"))
        if not tokenizer_ok:
            notes.append("没有找到任何 tokenizer 文件（tokenizer.json / tokenizer_config.json "
                         "/ vocab.json / tokenizer.model）")
        total = sum(shard.stat().st_size for shard in shards) if shards else 0
        size_ok = bool(shards) and total >= MODEL_MIN_BYTES
        if shards and not size_ok:
            notes.append(f"权重合计只有 {total / 1024 ** 3:.2f} GiB，"
                         f"低于下限 {MODEL_MIN_BYTES / 1024 ** 3:.0f} GiB（不像 27B-FP8）")
        max_ctx = None
        for key in ("max_position_embeddings", "seq_length", "model_max_length",
                    "max_sequence_length"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                max_ctx = value
                break
        if max_ctx is None:
            tokenizer_config = path / "tokenizer_config.json"
            if tokenizer_config.is_file():
                try:
                    data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
                    value = data.get("model_max_length")
                    max_ctx = value if isinstance(value, int) and 0 < value < 10 ** 7 else None
                except (ValueError, OSError):
                    max_ctx = None
        name = self.derive_name(path, config)
        ok = config_ok and weights_ok and tokenizer_ok and size_ok
        return ModelInfo(path=path, name=name, max_ctx=max_ctx, total_bytes=total,
                         ok=ok, source=label, notes=notes)

    def check_weights(self, path: Path) -> tuple[list[Path], list[str], bool]:
        """权重分片：有 index.json 就按它点票（一个都不能少），再逐个读头校验。

        返回 ``(分片, 提示, 是否通过)`` —— **第三个值才是判定**，前两个只是给人看的。
        """
        notes: list[str] = []
        index_file = path / "model.safetensors.index.json"
        shards: list[Path] = []
        if index_file.is_file():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                weight_map = data.get("weight_map") or {}
                names = sorted(set(weight_map.values()))
            except (ValueError, OSError) as exc:
                return [], [f"index.json 解析失败：{type(exc).__name__}: {exc}"], False
            names = list(names)
            if not names:
                return [], ["index.json 里没有 weight_map（声明 0 个分片）"], False
            notes.append(f"index.json 声明 {len(names)} 个分片")
            missing = [name for name in names if not (path / name).is_file()]
            if missing:
                notes.append(f"缺分片：{', '.join(missing[:5])}"
                             f"{' 等' if len(missing) > 5 else ''}")
                return [], notes, False
            for name in names:
                shard = path / name
                ok, expected, detail = safetensors_check(shard)
                if not ok:
                    notes.append(f"分片 {name} 校验失败：{detail}")
                    return [], notes, False
                shards.append(shard)
            declared = (data.get("metadata") or {}).get("total_size")
            actual = sum(shard.stat().st_size for shard in shards)
            if isinstance(declared, int) and declared > 0:
                ratio = actual / declared
                notes.append(f"index 声明总大小 {declared / 1024 ** 3:.2f} GiB，"
                             f"实际 {actual / 1024 ** 3:.2f} GiB（比值 {ratio:.3f}）")
                if ratio < 0.98:
                    notes.append("实际与声明总大小不符（可能拿到的是压缩/稀疏文件或分片被替换）")
            return shards, notes, True
        for pattern in ("*.safetensors", "pytorch_model*.bin", "*.bin"):
            found = sorted(path.glob(pattern))
            if not found:
                continue
            if pattern == "*.safetensors":
                for shard in found:
                    ok, expected, detail = safetensors_check(shard)
                    if not ok:
                        notes.append(f"分片 {shard.name} 校验失败：{detail}")
                        return [], notes, False
                    shards.append(shard)
                notes.append(f"无 index.json，按 {len(shards)} 个 safetensors 逐个校验")
                if len(found) == 1:
                    notes.append("单文件权重（未分片）")
                return shards, notes, True
            total = sum(item.stat().st_size for item in found)
            notes.append(f"{pattern} 是旧格式：只核对总大小 {total / 1024 ** 3:.2f} GiB"
                         "（safetensors 才能读头校验）")
            return found, notes, True
        notes.append("没有找到任何权重文件（*.safetensors / pytorch_model*.bin）")
        return [], notes, False

    @staticmethod
    def derive_name(path: Path, config: dict) -> str:
        """模型名**从模型目录读**（不另写一套）：``_name_or_path`` -> 目录名（``--`` 还原成 ``/``）。"""
        declared = str(config.get("_name_or_path") or "").strip()
        if declared and declared not in (".", "./"):
            return declared
        for parent in [path, *path.parents]:
            name = parent.name
            if name in ("snapshots", "master", "models", "") or name.startswith("."):
                continue
            if "--" in name:
                return name.replace("--", "/")
            return name
        return path.name

    # ---------- 缺失才谈下载（先报空间、等确认）----------
    def handle_missing(self) -> ModelInfo | None:
        ctx = self.ctx
        ctx.log("")
        ctx.log("[注意] 候选目录里没有**完整可用**的模型。")
        # t129：空间判据**只作用于真的要走下载这一条腿** —— 既有模型可用时，
        # 绝不能因为"下载所需空间不足"把人拦在门外（实测数据盘 22 GB 可用，
        # 而完整重下需要 31 GiB，两者都成立时**该做的是复用模型**）。
        for label, path in self.candidates():
            if ctx.dry_run:
                break
            existing = self.inspect(label, path)
            if existing is not None and existing.ok:
                ctx.log(f"[SKIP] 候选里其实有可用模型：{existing.path}"
                        f" —— 不需要下载，也就不参与空间判据")
                return existing
        if ctx.skip_model_download:
            ctx.log("[FAIL] --skip-model-download：按你的要求**不下载**，到此为止。"
                    "请人工确认模型路径（--model-path）后重跑。")
            raise SystemExit(EXIT_MODEL)
        need = MODEL_EXPECT_BYTES + DISK_SAFETY_BYTES
        target_disk = MODEL_ROOT if MODEL_ROOT.exists() else Path("/")
        free = self.free_bytes(target_disk)
        ctx.log(f"[PLAN] 缺失 -> 走 ModelScope 拉取 {MODELSCOPE_MODEL_ID}")
        ctx.log(f"       预计需要：{need / 1024 ** 3:.1f} GiB（模型约 "
                f"{MODEL_EXPECT_BYTES / 1024 ** 3:.0f} GiB + 安全余量 "
                f"{DISK_SAFETY_BYTES / 1024 ** 3:.0f} GiB）")
        ctx.log(f"       目标盘 {target_disk} 当前可用：{free / 1024 ** 3:.1f} GiB"
                "（现场 shutil.disk_usage 实测，不是写死的数字）")
        if free < need:
            ctx.log(f"       [FAIL] 空间不够：还差 {(need - free) / 1024 ** 3:.1f} GiB。"
                    "**本条判据只针对「真的要下载」这一条腿** —— 若模型其实已在实例上，"
                    "请用 --model-path 指过去（或先修好软链），不要为了凑空间删模型。")
            ctx.log("       建议：腾出空间（或用更大的盘/更小的量化模型）后重跑；"
                    "**本脚本不会自动删你的数据**。")
            raise SystemExit(EXIT_MODEL)
        if not ctx.assume_yes:
            ctx.log(f"       继续将真的下载约 {MODEL_EXPECT_BYTES / 1024 ** 3:.0f} GB。"
                    "确认请加 --yes（或交互回答 y）。")
            if not sys.stdin.isatty():
                ctx.log("[FAIL] 非交互环境：不确认就不下载（安全默认）。加 --yes 重跑。")
                raise SystemExit(EXIT_MODEL)
            answer = input("       继续下载？输入 y 继续，其它任意键退出：").strip().lower()
            if answer != "y":
                ctx.log("[FAIL] 已按你的输入取消下载。")
                raise SystemExit(EXIT_MODEL)
        target = MODEL_REAL_DIR
        ctx.log(f"[ACTION] modelscope download --model {MODELSCOPE_MODEL_ID} "
                f"--local_dir {target}（支持断点续传：重跑同一条命令即可续）")
        ctx.run([str(CondaEnv(ctx).python()), "-m", "modelscope", "download",
                 "--model", MODELSCOPE_MODEL_ID, "--local_dir", str(target)])
        return self.inspect("ModelScope 下载结果", target)

    @staticmethod
    def free_bytes(path: Path) -> int:
        try:
            return shutil.disk_usage(str(path)).free
        except OSError:
            return 0

    def ensure_link(self, info: ModelInfo) -> None:
        """写推荐软链（事实基线里的 ``qwen27b``）：已指向同一处就 SKIP。"""
        ctx = self.ctx
        target = info.path
        if not ctx.dry_run and MODEL_LINK.exists():
            try:
                if MODEL_LINK.resolve() == target.resolve():
                    ctx.log(f"[SKIP] 软链已就绪：{MODEL_LINK} -> {target}")
                    return
                ctx.log(f"[ACTION] 软链指向别处（{MODEL_LINK} -> "
                        f"{MODEL_LINK.resolve()}），重指到 {target}")
                MODEL_LINK.unlink()
            except OSError as exc:
                ctx.log(f"[注意] 处理软链失败：{type(exc).__name__}: {exc}")
                return
        ctx.log(f"[ACTION] 建立软链 {MODEL_LINK} -> {target}")
        ctx.run(["ln", "-sfn", str(target), str(MODEL_LINK)])


# ---------------------------------------------------------------------------
# 六、配置注入（以实际模型为准；重复执行只在内容变化时落盘）
# ---------------------------------------------------------------------------
class ConfigInjector:
    """把解析到的模型/端口/地址注入项目 ``.env``（``scripts/run_api.py`` 会读它）。"""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx

    def block(self, info: ModelInfo | None) -> str:
        ctx = self.ctx
        name = info.name if info else "<未解析到模型，请先解决模型问题>"
        lines = [
            ENV_BLOCK_BEGIN,
            "# 由 scripts/deploy_cloud.py 生成；重跑会**整块替换**，手工改动请写在这块之外。",
            "# 模型名与上下文长度都从**实际解析到的模型目录**读，不在这里另写一套。",
            "LLM_PROVIDER=openai_compat",
            f"LLM_MODEL={name}",
            f"OPENAI_COMPAT_BASE_URL=http://{ctx.host}:{ctx.port}/v1",
            "# 本地 vLLM/SGLang 不校验 key，但客户端**没 key 会直接抛错**，所以给个非空占位。",
            "OPENAI_API_KEY=EMPTY",
            "# 云端显式指定，避免自动接到 deepseek(无 key) -> mock 上：",
            "LLM_FALLBACKS=",
            "# 27B 冷加载可能超过默认 180s：",
            "LLM_TIMEOUT=300",
            "# 云端有卡，/metrics 才会有 gpu_* 样本：",
            "SYS_GPU_ENABLED=true",
            f"# 采集口径提示：本机实例的引擎端口 = {ctx.port}",
            f"# 引擎监听地址 = {ctx.bind_host}"
            + ("（只有本机能连 -> 跨机请用 ssh -L 隧道，例如 "
               f"ssh -L {ctx.port}:127.0.0.1:{ctx.port} <user>@<实例>）"
               if ctx.bind_host in ("127.0.0.1", "localhost") else
               "（对外暴露 -> 注意安全组/防火墙，别把端口裸奔到公网）"),
        ]
        if info is not None and info.max_ctx:
            lines.append(f"# 模型声明的上下文长度 = {info.max_ctx}（启动参数会带上，"
                         "本项目的 CONTEXT_MAX_CHARS 仍按检索侧控制）")
        lines.append("# 向量库侧（Milvus）：云端建议 fail-closed，但**本脚本不动它**，"
                     "确认 Milvus 可达后再手工打开：")
        lines.append("# MILVUS_FALLBACK_TO_MEMORY=0")
        lines.append(ENV_BLOCK_END)
        return "\n".join(lines) + "\n"

    def inject(self, info: ModelInfo | None) -> None:
        ctx = self.ctx
        ctx.log("")
        ctx.log("== 第 5 步：配置注入（以实际模型为准）==")
        if ctx.skip_inject:
            # 拓扑 A（t129）：实例上只跑引擎、没有本项目 -> 别在 /root 下生成无用的 .env
            ctx.log("[SKIP] --skip-inject：跳过配置注入（实例上没有本项目时用这个开关；"
                    "本机侧请自己写 .env，口径见 docs/CLOUD-DEPLOY.md §0/§6）")
            return
        env_path = ctx.project_dir / ".env"
        block = self.block(info)
        old = ""
        if env_path.is_file():
            old = env_path.read_text(encoding="utf-8", errors="replace")
        if ENV_BLOCK_BEGIN in old:
            head, _, rest = old.partition(ENV_BLOCK_BEGIN)
            _, _, tail = rest.partition(ENV_BLOCK_END)
            new = head.rstrip("\n") + "\n\n" + block + tail.lstrip("\n")
        else:
            new = old + ("\n" if old and not old.endswith("\n") else "") + "\n" + block
        if ctx.dry_run:
            ctx.log(f"[DRY] 写 {env_path}（整块替换 {ENV_BLOCK_BEGIN} ...）")
            for line in block.splitlines():
                ctx.log("    | " + line)
            return
        if new == old:
            ctx.log(f"[SKIP] {env_path} 内容已是最新（幂等：不落盘）")
            return
        try:
            backup = env_path.with_name(env_path.name + ".bak")
            if env_path.is_file() and not backup.exists():
                shutil.copy2(env_path, backup)
                ctx.log(f"[INFO] 已备份原文件到 {backup}")
            env_path.write_text(new, encoding="utf-8")
        except OSError as exc:
            ctx.log(f"[FAIL] 写 {env_path} 失败：{type(exc).__name__}: {exc}")
            raise SystemExit(EXIT_CONFIG) from exc
        ctx.log(f"[ACTION] 已写入 {env_path}（模型名={info.name if info else '未知'}，"
                f"base_url=http://{ctx.host}:{ctx.port}/v1）")


# ---------------------------------------------------------------------------
# 七、启动（SGLang 为主 / vLLM 为备）
# ---------------------------------------------------------------------------
class ServerLauncher:
    """拼启动命令：SGLang 与 vLLM 各自的等价参数；离线 env 与线程数都可配。"""

    def __init__(self, ctx: Ctx, env: CondaEnv) -> None:
        self.ctx = ctx
        self.env = env

    def env_exports(self) -> dict:
        ctx = self.ctx
        return {
            "TRANSFORMERS_OFFLINE": "1",     # 事实基线的离线启动方式
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": ctx.omp_threads,
        }

    def argv(self, info: ModelInfo | None) -> list[str]:
        ctx = self.ctx
        model = info.path if info is not None else MODEL_LINK
        name = info.name if info is not None else "qwen27b"
        if ctx.engine == "sglang":
            argv = [str(self.env.python()), "-m", "sglang.launch_server",
                    "--model-path", str(model),
                    "--served-model-name", name,
                    "--port", str(ctx.port),
                    "--dtype", "auto",
                    "--host", ctx.bind_host]
            if info is not None and info.max_ctx:
                argv += ["--context-length", str(info.max_ctx)]
            return argv
        argv = [str(self.env.python()), "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model),
                "--served-model-name", name,
                "--host", ctx.bind_host,
                "--port", str(ctx.port),
                "--dtype", "auto",
                "--gpu-memory-utilization", "0.90"]
        if info is not None and info.max_ctx:
            argv += ["--max-model-len", str(info.max_ctx)]
        return argv

    def health_url(self) -> str:
        if self.ctx.engine == "sglang":
            return f"http://127.0.0.1:{self.ctx.port}/health_generate"
        return f"http://127.0.0.1:{self.ctx.port}/health"

    def is_listening(self) -> bool:
        out = self.ctx.probe(["bash", "-lc",
                              f"ss -ltn 2>/dev/null | grep -c ':{self.ctx.port} ' || true"])
        return (out.strip() or "0").isdigit() and int(out.strip() or "0") > 0

    def start(self, info: ModelInfo | None) -> None:
        ctx = self.ctx
        ctx.log("")
        ctx.log(f"== 第 6 步：启动 {ctx.engine} 服务 ==")
        argv = self.argv(info)
        exports = self.env_exports()
        prefix = " ".join(f"{key}={value}" for key, value in exports.items())
        ctx.log(f"[PLAN] {prefix} {' '.join(argv)}")
        if not ctx.dry_run and self.is_listening():
            ctx.log(f"[SKIP] 端口 {ctx.port} 已在监听 —— 幂等：不重复起服务"
                    "（要重启用 --restart）")
            return
        if ctx.dry_run:
            ctx.log(f"[DRY] 后台启动并把日志写到 {ctx.server_log()}"
                    f"（pid -> {ctx.server_pid()}）")
            return
        try:
            ctx.log_root().mkdir(parents=True, exist_ok=True)
            log_handle = ctx.server_log().open("a", encoding="utf-8")
        except OSError as exc:
            # 日志目录不可写要说人话（别甩一个 PermissionError 栈）：--log-dir 指到可写位置
            ctx.log(f"[FAIL] 日志目录不可写：{ctx.log_root()}（{type(exc).__name__}: {exc}）；"
                    "请用 --log-dir 指到可写目录后重跑")
            raise SystemExit(EXIT_START) from exc
        proc = subprocess.Popen(argv, env={**os.environ, **exports}, stdout=log_handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        ctx.server_pid().write_text(str(proc.pid), encoding="utf-8")
        ctx.log(f"[OK] 已后台启动 pid={proc.pid}，日志：{ctx.server_log()}")
        ctx.log("     等待健康检查（首次加载 27B 权重通常要几分钟）...")
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.health_ok():
                ctx.log("[OK] 健康检查通过")
                break
            time.sleep(10)
        else:
            ctx.log(f"[FAIL] 900s 内健康检查没过：先看 {ctx.server_log()} 末尾，"
                    "再按 --diagnose 的清单排查；SGLang 起不来时切 vLLM：")
            ctx.log("       python3 scripts/deploy_cloud.py --engine vllm")

    def health_ok(self) -> bool:
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(self.health_url(), timeout=10) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError):
            return False


# ---------------------------------------------------------------------------
# 八、自检：健康检查 + bench_serving 三档（条件与结果一起固化）
# ---------------------------------------------------------------------------
class SelfCheck:
    """``curl /health_generate`` + ``python -m sglang.bench_serving``（并发 1/4/8）。"""

    def __init__(self, ctx: Ctx, env: CondaEnv, launcher: ServerLauncher) -> None:
        self.ctx = ctx
        self.env = env
        self.launcher = launcher

    # ---------- 健康 ----------
    def health(self) -> bool:
        ctx = self.ctx
        url = self.launcher.health_url()
        ctx.log("")
        ctx.log("== 自检 1/2：健康检查 ==")
        if ctx.dry_run:
            ctx.log(f"[DRY] curl -fsS --max-time 30 {url}")
            return True
        ctx.log(f"[ACTION] curl -fsS --max-time 30 {url}")
        code, out = ctx.run(["curl", "-fsS", "--max-time", "30", url], check=False,
                            capture=True)
        ctx.log(f"[{'OK' if code == 0 else 'FAIL'}] 健康检查退出码 {code}")
        return code == 0

    # ---------- 压测 ----------
    def bench_argv(self, concurrency: int, output_file: Path) -> list[str]:
        ctx = self.ctx
        backend = "sglang" if ctx.engine == "sglang" else "vllm"
        argv = [str(self.env.python()), "-m", "sglang.bench_serving",
                "--backend", backend,
                "--host", "127.0.0.1", "--port", str(ctx.port),
                "--dataset-name", "random",
                "--random-input-len", str(BENCH_INPUT_LEN),
                "--random-output-len", str(BENCH_OUTPUT_LEN),
                "--max-concurrency", str(concurrency),
                "--num-prompts", str(concurrency * BENCH_PROMPTS_PER_CONC),
                "--output-file", str(output_file)]
        return argv

    def bench(self, concurrency: int) -> dict:
        ctx = self.ctx
        out_file = ctx.bench_dir() / f"bench-c{concurrency}.jsonl"
        argv = self.bench_argv(concurrency, out_file)
        ctx.log(f"[ACTION] {' '.join(argv)}")
        if ctx.dry_run:
            return {"concurrency": concurrency, "status": "dry-run"}
        ctx.bench_dir().mkdir(parents=True, exist_ok=True)
        code, output = ctx.run(argv, check=False, capture=True)
        result = {"concurrency": concurrency, "exit_code": code}
        parsed = self.parse_output(out_file, output)
        result.update(parsed)
        ttft = result.get("ttft_mean_ms")
        itl = result.get("itl_mean_ms")
        if code != 0:
            result["status"] = "failed"
        elif ttft is None or itl is None:
            result["status"] = "unparsed"
        else:
            ok = ttft < TTFT_LIMIT_MS and itl < ITL_LIMIT_MS
            result["status"] = "pass" if ok else "miss"
        return result

    @staticmethod
    def parse_output(out_file: Path, console: str) -> dict:
        """先读 JSONL（稳），读不到再从控制台文本里抠 TTFT/ITL 的均值。"""
        data: dict = {}
        if out_file.is_file():
            try:
                lines = [line for line in out_file.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
                if lines:
                    payload = json.loads(lines[-1])
                    mean_ttft = payload.get("mean_ttft_ms") or payload.get("mean_ttft")
                    mean_itl = payload.get("mean_itl_ms") or payload.get("mean_itl")
                    if mean_ttft is not None:
                        data["ttft_mean_ms"] = float(mean_ttft)
                    if mean_itl is not None:
                        data["itl_mean_ms"] = float(mean_itl)
            except (ValueError, OSError):
                pass
        if "ttft_mean_ms" not in data:
            match = re.search(r"Time to First Token \(TTFT, ms\).*?Mean:\s*([\d.]+)",
                              console, re.S)
            if match:
                data["ttft_mean_ms"] = float(match.group(1))
        if "itl_mean_ms" not in data:
            match = re.search(r"Inter-Token Latency \(ITL, ms\).*?Mean:\s*([\d.]+)",
                              console, re.S)
            if match:
                data["itl_mean_ms"] = float(match.group(1))
        return data

    def run(self) -> int:
        ctx = self.ctx
        healthy = self.health()
        if not healthy:
            ctx.log("[FAIL] 健康检查没过 —— 先修服务再压测（--diagnose 有清单）")
            self.print_vllm_fallback()
            return EXIT_SELFCHECK
        ctx.log("")
        ctx.log("== 自检 2/2：bench_serving（并发 "
                f"{'/'.join(str(item) for item in BENCH_CONCURRENCY)} × 输入 {BENCH_INPUT_LEN} "
                f"× 输出 {BENCH_OUTPUT_LEN}）==")
        ctx.log(f"     及格线：TTFT < {TTFT_LIMIT_MS:.0f} ms 且 ITL < {ITL_LIMIT_MS:.0f} ms"
                "（以实测为准；版本号只作回溯用，不写死判断）")
        if ctx.dry_run:
            results = [self.bench(level) for level in BENCH_CONCURRENCY]
            ctx.log("")
            ctx.log("[DRY] dry-run **没有真的跑压测**，所以这里没有结论、也不算达标；"
                    "命令就是上面三条，真机跑完再回来对表。")
            _ = results
            return EXIT_OK
        results = [self.bench(level) for level in BENCH_CONCURRENCY]
        ctx.log("")
        ctx.log("== 自检结论 ==")
        ctx.log("并发  TTFT(ms)  ITL(ms)  结论")
        misses = 0
        for item in results:
            ttft = item.get("ttft_mean_ms")
            itl = item.get("itl_mean_ms")
            status = item.get("status", "?")
            if status in ("miss", "failed", "unparsed"):
                misses += 1
            ctx.log(f"{item['concurrency']:>4}  "
                    f"{(f'{ttft:.1f}' if ttft is not None else '-'):>8}  "
                    f"{(f'{itl:.1f}' if itl is not None else '-'):>7}  {status}")
        report = ctx.log_root() / "selfcheck.json"
        ctx.log_root().mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({
            "engine": ctx.engine, "port": ctx.port,
            "conditions": {"concurrency": list(BENCH_CONCURRENCY),
                           "input_len": BENCH_INPUT_LEN, "output_len": BENCH_OUTPUT_LEN,
                           "num_prompts_per_concurrency": BENCH_PROMPTS_PER_CONC},
            "thresholds": {"ttft_ms": TTFT_LIMIT_MS, "itl_ms": ITL_LIMIT_MS},
            "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.log(f"[INFO] 自检报告：{report}")
        if misses:
            ctx.log(f"[FAIL] {misses}/{len(results)} 档没达标；SGLang 不达标/起不来时的"
                    "等价回退步骤：")
            self.print_vllm_fallback()
            return EXIT_SELFCHECK
        ctx.log("[OK] 三档全部达标")
        return EXIT_OK

    def print_vllm_fallback(self) -> None:
        ctx = self.ctx
        name = "<解析到的同名模型>"
        ctx.log("      1) python3 scripts/deploy_cloud.py --engine vllm --dry-run   # 先看计划")
        ctx.log("      2) python3 scripts/deploy_cloud.py --engine vllm             # 装 vLLM 并起服务")
        ctx.log("      3) python3 scripts/deploy_cloud.py --engine vllm --selfcheck # 同一套口径压测")
        ctx.log(f"      等价启动命令（人工执行时）：")
        ctx.log(f"         TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS="
                f"{ctx.omp_threads} python3 -m vllm.entrypoints.openai.api_server "
                f"--model {MODEL_LINK} --served-model-name {name} --host {ctx.bind_host} "
                f"--port {ctx.port} --dtype auto --gpu-memory-utilization 0.90")
        ctx.log("      4) 配置里**不用改模型名**：脚本注入的 LLM_MODEL 就是模型目录里读到的那个")


# ---------------------------------------------------------------------------
# 九、诊断（失败时原样可复制）
# ---------------------------------------------------------------------------
class Diagnostics:
    def __init__(self, ctx: Ctx, env: CondaEnv) -> None:
        self.ctx = ctx
        self.env = env

    def commands(self) -> list[str]:
        python = self.env.python()
        return [
            f"nvidia-smi",
            f"nvcc --version",
            f"{python} -c \"import torch;print('torch',torch.__version__,"
            "torch.version.cuda,torch.cuda.is_available())\"",
            f"{python} -m pip list | grep -Ei 'torch|sglang|vllm|flashinfer|transformers|"
            "tokenizers|triton'",
            f"tail -n 120 {self.ctx.server_log()}",
            f"curl -fsS --max-time 30 {self.engine_url()}",
            "df -h /root/autodl-tmp",
        ]

    def engine_url(self) -> str:
        port = self.ctx.port
        return (f"http://127.0.0.1:{port}/health_generate" if self.ctx.engine == "sglang"
                else f"http://127.0.0.1:{port}/health")

    def print_all(self) -> None:
        ctx = self.ctx
        ctx.log("")
        ctx.log("== 诊断清单（逐条复制执行；报错时把输出一起贴出来）==")
        for index, cmd in enumerate(self.commands(), 1):
            ctx.log(f"  {index}) {cmd}")
        ctx.log("")
        ctx.log("已知坑（事实基线里登记过的，别再重复试）：")
        ctx.log("  - nvcc fatal : Unknown option '--compress-mode=size' = flashinfer JIT 硬编码了")
        ctx.log("    CUDA 13.0 选项；对症做法是 pin flashinfer（本脚本默认做），不是换启动参数。")
        for item in INEFFECTIVE_FIXES:
            ctx.log(f"  - [无效] {item}")
        ctx.log("  - SGLang 实在起不来就切 vLLM（对 FP8 + CUDA 12.9 更成熟）。")


# ---------------------------------------------------------------------------
# 十、启动横幅（把**实际装到的版本**写进去，便于回溯）
# ---------------------------------------------------------------------------
class Banner:
    def __init__(self, ctx: Ctx, env: CondaEnv, deps: Deps) -> None:
        self.ctx = ctx
        self.env = env
        self.deps = deps

    def packages(self) -> dict:
        names = ("torch", "sglang", "vllm", "sglang-kernel", "flashinfer-python",
                 "flashinfer-cubin", "transformers", "tokenizers", "modelscope")
        return {name: (self.deps.installed_version(name) or "-") for name in names}

    def print_banner(self, info: ModelInfo | None, phase: str) -> None:
        ctx = self.ctx
        ctx.log("")
        ctx.log(f"-------------------- 部署横幅（{phase}）--------------------")
        ctx.log(f"引擎          : {ctx.engine}（端口 {ctx.port}，绑定 {ctx.bind_host}"
                + ("，只对本机开放 -> SSH 隧道场景）"
                   if ctx.bind_host in ("127.0.0.1", "localhost") else "，对外暴露）"))
        # 注意：这里打印**实际使用的**前缀与来源（t129）—— 不再一律打印硬编码值
        ctx.log(f"conda env     : {self.env.prefix()}"
                f"（来源：{self.env.origin()}；"
                f"{self.env.version() or ('dry-run 未探测' if ctx.dry_run else '未就绪')}）")
        ctx.log(f"日志目录      : {ctx.log_root()}")
        ctx.log(f"配置注入      : "
                + ("已跳过（--skip-inject：实例上只跑引擎）" if ctx.skip_inject
                   else f"写入 {ctx.project_dir / '.env'}"))
        ctx.log(f"conda 命令    : {' '.join(self.env.conda_cmd())}")
        if info is not None:
            ctx.log(f"模型          : {info.name}")
            ctx.log(f"模型路径      : {info.path}")
            ctx.log(f"上下文长度    : {info.max_ctx or '未声明'}")
            ctx.log(f"权重合计      : {info.total_bytes / 1024 ** 3:.2f} GiB")
        else:
            ctx.log("模型          : 未解析（dry-run 或有问题，见上面的步骤输出）")
        ctx.log("flashinfer 口径: " + {
            "pin": f"flashinfer-python=={FLASHINFER_PIN}（固定组合，替换会炸的 0.6.x）",
            "cubin": f"flashinfer-cubin（预编译内核，>= {FLASHINFER_CUBIN_MIN}）",
            "none": "不装 flashinfer（彻底绕开 JIT）",
        }.get(ctx.flashinfer_mode, ctx.flashinfer_mode))
        versions = self.packages()
        ctx.log("实际装到的版本: " + "  ".join(f"{name}={ver}" for name, ver in versions.items()))
        if not ctx.dry_run:
            driver = ctx.probe(["nvidia-smi", "--query-gpu=driver_version,name",
                                "--format=csv,noheader"])
            cuda = ctx.probe(["nvcc", "--version"])
            cuda_line = ""
            for line in cuda.splitlines():
                if "release" in line:
                    cuda_line = line.strip()
                    break
            ctx.log(f"驱动/GPU      : {driver or 'nvidia-smi 不可用'}")
            ctx.log(f"nvcc          : {cuda_line or 'nvcc 不可用（纯 pip 轮子也能跑）'}")
        ctx.log("诚实边界      : 本脚本在**开发机上只做到 dry-run / 语法 / 计划自检**；"
                 "开发机（Windows）实测有 RTX 2060(6GB)、驱动 577.00，"
                 "但**没有 CUDA 工具链（nvcc 不可用）、也不是 Linux**，"
                 "-> SGLang/vLLM 的**真机启动与性能未验证**，"
                 "真机证据必须由实例侧产出（命令见 docs/CLOUD-DEPLOY.md）。")
        ctx.log("-----------------------------------------------------------")


# ---------------------------------------------------------------------------
# 十一、编排
# ---------------------------------------------------------------------------
class DeployRunner:
    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.env = CondaEnv(ctx)
        self.deps = Deps(ctx, self.env)
        self.engines = EngineInstaller(ctx, self.env, self.deps)
        self.models = ModelResolver(ctx)
        self.config = ConfigInjector(ctx)
        self.launcher = ServerLauncher(ctx, self.env)
        self.selfcheck = SelfCheck(ctx, self.env, self.launcher)
        self.diagnostics = Diagnostics(ctx, self.env)
        self.banner = Banner(ctx, self.env, self.deps)

    def deploy(self, restart: bool) -> int:
        ctx = self.ctx
        ctx.log("=" * 78)
        ctx.log("法律 RAG · AutoDL 云上部署（幂等）"
                + ("  [DRY-RUN：只打印，不动系统]" if ctx.dry_run else ""))
        ctx.log("事实基线：模型 Qwen3.8-27B-FP8（约 29 GB）+ handoff/CLOUD-ENV-FACTS.md 的版本与卡点；"
                "磁盘用量**现场读**（见第 4 步），不写死数字")
        ctx.log("=" * 78)
        self.banner.print_banner(None, "开始")
        if ctx.dry_run:
            ctx.log("[PLAN] dry-run 不探测系统、不落盘：下面的命令就是真跑时会执行的顺序。")
        if restart and not ctx.dry_run:
            self.stop_server()
        self.env.ensure()
        self.deps.ensure_python_tools()
        self.deps.ensure_torch()
        if ctx.engine == "sglang":
            self.deps.ensure_sglang_kernel()
        self.engines.install()
        info = self.models.resolve()
        if info is not None:
            self.models.ensure_link(info)
        self.config.inject(info)
        self.launcher.start(info)
        self.banner.print_banner(info, "结束/待自检")
        ctx.log("")
        ctx.log("下一步（在**实例的**项目根目录下复制执行）：")
        ctx.log("  python3 scripts/deploy_cloud.py --selfcheck    # 健康检查 + 三档压测")
        ctx.log("  python3 scripts/deploy_cloud.py --diagnose     # 出问题时的诊断清单")
        ctx.log("  python3 scripts/run_api.py                     # 起本项目（读 .env 注入的那套）")
        ctx.log("  运行手册：docs/CLOUD-DEPLOY.md（含回滚/清理与已知坑）")
        return EXIT_OK

    def stop_server(self) -> None:
        ctx = self.ctx
        pid_file = ctx.server_pid()
        if not pid_file.is_file():
            ctx.log("[SKIP] 没有 pid 文件，无需停止")
            return
        pid = pid_file.read_text(encoding="utf-8").strip()
        ctx.log(f"[ACTION] --restart：停止旧进程 pid={pid}")
        ctx.run(["bash", "-lc", f"kill {pid} 2>/dev/null || true"])
        ctx.run(["bash", "-lc", "sleep 3"])
        try:
            pid_file.unlink()
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy_cloud.py",
        description="AutoDL 上的一键（幂等）部署：重建 conda 环境 -> 依赖 -> SGLang/vLLM"
                    "（含 flashinfer 卡点对症）-> 模型解析与校验 -> 配置注入 -> 启动 -> 自检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python3 scripts/deploy_cloud.py --dry-run\n"
               "  python3 scripts/deploy_cloud.py --engine sglang --yes\n"
               "  python3 scripts/deploy_cloud.py --engine vllm --skip-model-download\n"
               "  python3 scripts/deploy_cloud.py --selfcheck\n"
               "  # 拓扑 A（云上只跑引擎，项目/Milvus 在本机走 ssh -L 隧道）：\n"
               "  python3 scripts/deploy_cloud.py --bind-host 127.0.0.1 --skip-inject\n"
               "  python3 scripts/deploy_cloud.py --env-prefix /root/miniconda3/envs/rag"
               " --log-dir /root/logs\n",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不改动系统")
    parser.add_argument("--engine", choices=("sglang", "vllm"), default="sglang",
                        help="推理引擎：sglang（主）/ vllm（备）")
    parser.add_argument("--model-path", default="", help="模型目录（默认按事实基线解析软链）")
    parser.add_argument("--skip-model-download", action="store_true",
                        help="模型缺失时**不**下载，直接失败退出（省磁盘）")
    parser.add_argument("--force-recreate-env", action="store_true",
                        help="删掉 conda 环境重建（默认存在即复用）")
    parser.add_argument("--env-prefix", default="",
                        help="conda 环境前缀（默认：先查 `conda env list` 里名为 rag 的环境，"
                             "找不到才用 `conda create -n rag` 的默认前缀 "
                             "/root/miniconda3/envs/rag，即系统盘）")
    parser.add_argument("--log-dir", default="",
                        help="日志目录（部署日志/引擎日志/压测结果都落这里；"
                             "默认 /root/autodl-tmp/logs）")
    parser.add_argument("--bind-host", default=DEFAULT_HOST_BIND,
                        help="引擎监听地址（默认 0.0.0.0；SSH 隧道场景设 127.0.0.1 只绑本机）")
    parser.add_argument("--skip-inject", action="store_true",
                        help="跳过配置注入（拓扑 A：实例上只跑引擎，没有本项目）")
    parser.add_argument("--flashinfer-mode", choices=("pin", "cubin", "none"), default="pin",
                        help="flashinfer 处置：pin（默认固定组合）/ cubin（预编译内核）/ none（不装）")
    parser.add_argument("--host", default="127.0.0.1",
                        help="注入 OPENAI_COMPAT_BASE_URL 用的主机名（同机网页用 127.0.0.1）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="引擎端口（默认 30000）")
    parser.add_argument("--omp-threads", default=DEFAULT_OMP_THREADS,
                        help="OMP_NUM_THREADS（事实基线用 8）")
    parser.add_argument("--sglang-version", default="", help="pin sglang 版本（默认不 pin）")
    parser.add_argument("--vllm-version", default="", help="pin vllm 版本（默认不 pin）")
    parser.add_argument("--project-dir", default="", help="项目根目录（默认脚本上一级）")
    parser.add_argument("--yes", action="store_true", help="下载权重等破坏性动作不再交互确认")
    parser.add_argument("--restart", action="store_true", help="先停掉 pid 文件里的旧服务")
    parser.add_argument("--selfcheck", action="store_true",
                        help="只做：健康检查 + bench_serving 三档（并发 1/4/8）")
    parser.add_argument("--diagnose", action="store_true", help="只打印诊断清单")
    parser.add_argument("--log-file", default="",
                        help="部署日志路径（默认 <--log-dir>/deploy-cloud.log；- 表示不落盘）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = (Path(args.project_dir) if args.project_dir
                   else Path(__file__).resolve().parents[1])
    ctx = Ctx(project_dir=project_dir, dry_run=args.dry_run, engine=args.engine,
              model_path=args.model_path, skip_model_download=args.skip_model_download,
              force_recreate_env=args.force_recreate_env, flashinfer_mode=args.flashinfer_mode,
              host=args.host, port=args.port, omp_threads=args.omp_threads,
              assume_yes=args.yes, sglang_version=args.sglang_version,
              vllm_version=args.vllm_version, env_prefix=args.env_prefix,
              log_dir=args.log_dir, bind_host=args.bind_host, skip_inject=args.skip_inject)
    # 部署日志：--log-file 显式给了就用它；否则跟着 --log-dir（默认维持原路径）
    log_path = (Path(args.log_file) if args.log_file
                else (ctx.deploy_log() if args.log_dir else DEPLOY_LOG))
    log = Log(log_path, enabled=not args.dry_run)
    ctx.log = log
    runner = DeployRunner(ctx)
    try:
        if args.diagnose:
            runner.banner.print_banner(None, "诊断")
            runner.diagnostics.print_all()
            return EXIT_OK
        if args.selfcheck:
            runner.banner.print_banner(None, "自检")
            return runner.selfcheck.run()
        return runner.deploy(restart=args.restart)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_USAGE
        ctx.log(f"[EXIT] 退出码 {code}")
        return code
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
