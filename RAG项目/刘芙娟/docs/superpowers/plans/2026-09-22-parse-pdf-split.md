# parse_pdf.py 拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 338 行的 `backend/parse_pdf.py` 按职责拆成 4 个文件，运行方式与行为完全不变。

**Architecture:** 入口 `backend/parse_pdf.py` 保留不动的位置和命令，只留输入发现、编排、命令行三件事；共享常量与 `ScriptError` 下沉到 `backend/parse/__init__.py`；MinerU 外部依赖进 `mineru_env.py`；磁盘产物操作进 `artifacts.py`。两个子模块只被入口 import，互不反向依赖。

**Tech Stack:** Python 3.12.14（`rag/python.exe`），标准库 only，无测试框架。

## Global Constraints

- 运行解释器固定为 `D:/zg6_Project/9/med_rag/rag/python.exe`，不得改用其它解释器。
- 运行命令保持为 `rag/python.exe backend/parse_pdf.py [INPUT] [选项]`，不得改变。
- 退出码保持 `0 成功 / 1 参数错误 / 2 校验失败 / 3 外部依赖不可用`。
- 选项集合保持 `--backend` `--force` `--config` `--pipeline-models` `--vlm-models` `--dry-run`。
- 不引入任何新依赖，不引入测试框架。
- **本仓库无 git**：没有 commit 步骤。替代的安全网是 Task 1 的物理备份，且每个 Task 末尾必须重跑验证命令。
- 所有终端输出检查命令一律带 `PYTHONIOENCODING=utf-8`，否则中文在 GBK 控制台变乱码，无法肉眼比对。

### 关于导入方式（含一次已被证伪的判断，留档）

本计划初稿曾写：「`backend/` 下没有 `__init__.py`，不是包，`python -m backend.parse_pdf`
无法运行，故相对导入分支是死代码，入口直接写 `from parse import ...`」。

**这个判断是错的。** Python 3 会把没有 `__init__.py` 的目录当作**隐式命名空间包**，
`-m` 照样能导入。实测：

```
>>> importlib.util.find_spec('backend')
backend spec: None | submodule_search_locations: True   ← 是命名空间包
```

按错误判断照做之后，`python -m backend.parse_pdf` 从「能跑」变成
`ModuleNotFoundError: No module named 'parse'`（因为该模式下 `sys.path[0]` 是仓库根，
`parse` 不在其下）。这是计划自身引入的行为回归，由 Task 5 的最终整体审查发现。

**已修正**：入口采用设计文档 §导入方式 原本给出的 `try/except` 双分支写法，
两种调用方式均恢复可用。教训是「没有 `__init__.py` 就不是包」在 Python 3 已不成立，
不要据此删掉兜底分支。

---

### Task 1: 备份原文件并采集对拍基线

这一步必须在动任何代码之前完成。原文件一旦改动且无备份，本仓库无 git 可回滚。

**Files:**
- Create: `backend/parse_pdf.py.bak`（原文件副本）
- Create: `docs/superpowers/plans/baseline-dryrun.txt`
- Create: `docs/superpowers/plans/baseline-artifacts.txt`

**Interfaces:**
- Produces: `baseline-dryrun.txt`、`baseline-artifacts.txt` 两份基线，供 Task 5、Task 6 逐字比对。

- [ ] **Step 1: 物理备份原文件**

```bash
cd /d/zg6_Project/9/med_rag
cp backend/parse_pdf.py backend/parse_pdf.py.bak
wc -l backend/parse_pdf.py.bak
```

Expected: `338 backend/parse_pdf.py.bak`

- [ ] **Step 2: 采集 dry-run 基线**

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe backend/parse_pdf.py --dry-run \
    > docs/superpowers/plans/baseline-dryrun.txt 2>&1
cat docs/superpowers/plans/baseline-dryrun.txt
```

Expected（七行，中文不得是乱码）：

```
仓库根 D:\zg6_Project\9\med_rag
后端   pipeline（权重 C:\Users\Lenovo\.cache\modelscope\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master）
配置   D:\zg6_Project\9\med_rag\.mineru\mineru.json
产物   D:\zg6_Project\9\med_rag\data\parsed
待处理 1 份
  doc_id=d6da41b5d356  国家基层高血压防治管理指南2025版.pdf
（dry-run：未写配置、未调用 MinerU）
```

- [ ] **Step 3: 采集产物解析基线**

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe - <<'EOF' > docs/superpowers/plans/baseline-artifacts.txt 2>&1
import sys
from pathlib import Path
sys.path.insert(0, "backend")
import parse_pdf as P

doc_dir = Path("data/parsed/d6da41b5d356")
path = P.locate_content_list(doc_dir)
print("定位到:", path.name)
items = P.load_content_items(path)
P.validate_page_idx(items)
dist = P.build_type_distribution(items, "pipeline")
print("块数:", dist["total_blocks"])
print("类型分布:", dict(sorted(dist["type_counts"].items())))
print("未知类型:", dist["unknown_types"])
print("可跳过:", P.is_artifact_complete(doc_dir))
EOF
cat docs/superpowers/plans/baseline-artifacts.txt
```

Expected：

```
定位到: 国家基层高血压防治管理指南2025版_content_list.json
块数: 304
类型分布: {'footer': 1, 'header': 19, 'image': 2, 'list': 1, 'page_footnote': 1, 'page_number': 15, 'table': 8, 'text': 257}
未知类型: []
可跳过: True
```

---

### Task 2: 新建 `backend/parse/__init__.py`

**Files:**
- Create: `backend/parse/__init__.py`

**Interfaces:**
- Consumes: 无
- Produces: `EXIT_OK` `EXIT_ARG` `EXIT_VALIDATION` `EXIT_DEPENDENCY`（int）、`BACKENDS`（tuple[str,...]）、`DEFAULT_BACKEND`（str）、`REPO_ROOT` `SOURCE_DIR` `PARSED_DIR` `DEFAULT_CONFIG_PATH` `DEFAULT_PIPELINE_MODELS` `DEFAULT_VLM_MODELS`（Path）、`ScriptError(exit_code: int, message: str)`。Task 3、4、5 全部依赖本模块。

本模块不得 import `parse/` 下其它模块，以杜绝循环导入。

- [ ] **Step 1: 写文件**

```python
# -*- coding: utf-8 -*-
"""S2 解析的共享定义：退出码、仓库路径、默认权重目录、ScriptError。

只放被多方共享、且不属于任何一方的常量与异常，不放逻辑——因此本模块不 import
包内其它模块，无循环导入风险。
"""

from __future__ import annotations

import os
from pathlib import Path

EXIT_OK, EXIT_ARG, EXIT_VALIDATION, EXIT_DEPENDENCY = 0, 1, 2, 3

BACKENDS = ("pipeline", "vlm-engine")
DEFAULT_BACKEND = "pipeline"

# 三层 parent：本文件 → parse/ → backend/ → 仓库根。
# 不能写成两层——那会指向 backend/，产物会被静默写到 backend\data\parsed\，
# 不报错，只是找不着。
REPO_ROOT = Path(
    os.environ.get("MED_RAG_REPO_ROOT") or Path(__file__).resolve().parent.parent.parent)
SOURCE_DIR = REPO_ROOT / "data" / "source_data"
PARSED_DIR = REPO_ROOT / "data" / "parsed"
DEFAULT_CONFIG_PATH = REPO_ROOT / ".mineru" / "mineru.json"

# 两套权重均已在本地。此处仅为默认值，可用命令行选项覆盖。
DEFAULT_PIPELINE_MODELS = Path(
    r"C:\Users\Lenovo\.cache\modelscope\models"
    r"\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master")
DEFAULT_VLM_MODELS = Path(
    r"C:\Users\Lenovo\.cache\modelscope\models"
    r"\OpenDataLab--MinerU2.5-Pro-2605-1.2B\snapshots\master")


class ScriptError(Exception):
    """携带退出码的显式失败。禁止用 except 吞掉它。"""

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
```

- [ ] **Step 2: 验证 REPO_ROOT 落在仓库根**

这才是本次拆分最容易踩的坑，必须单独验：

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe -c "
import sys; sys.path.insert(0, 'backend')
import parse
print('REPO_ROOT :', parse.REPO_ROOT)
print('PARSED_DIR:', parse.PARSED_DIR)
print('退出码    :', parse.EXIT_OK, parse.EXIT_ARG, parse.EXIT_VALIDATION, parse.EXIT_DEPENDENCY)
"
```

Expected（`REPO_ROOT` 必须是 `med_rag`，**不是** `med_rag\backend`）：

```
REPO_ROOT : D:\zg6_Project\9\med_rag
PARSED_DIR: D:\zg6_Project\9\med_rag\data\parsed
退出码    : 0 1 2 3
```

若 `REPO_ROOT` 打印出 `...\med_rag\backend`，说明 parent 层数写错了，回到 Step 1 改成三层。

- [ ] **Step 3: 验证环境变量覆盖仍然生效**

```bash
cd /d/zg6_Project/9/med_rag
MED_RAG_REPO_ROOT=fakeroot PYTHONIOENCODING=utf-8 rag/python.exe -c "
import sys; sys.path.insert(0, 'backend')
import parse
print('REPO_ROOT :', parse.REPO_ROOT)
"
```

Expected: `REPO_ROOT : fakeroot`

注意**不要**用 `/tmp/fakeroot` 这类 POSIX 绝对路径做这个测试：Git Bash 会把 `/tmp/...`
自动转换成 `C:\Users\Lenovo\AppData\Local\Temp\...` 再传给原生 Python，
打印出来的值与预期对不上，会让人误以为是覆盖逻辑坏了。用一个不含斜杠的裸名字
`fakeroot` 才对得上。（已实测：`/tmp/fakeroot` → `C:\...\AppData\Local\Temp\fakeroot`；
`fakeroot` → `fakeroot`。覆盖逻辑本身正确。）

---

### Task 3: 新建 `backend/parse/artifacts.py`

**Files:**
- Create: `backend/parse/artifacts.py`

**Interfaces:**
- Consumes: `parse.EXIT_VALIDATION`、`parse.ScriptError`
- Produces: `KNOWN_BLOCK_TYPES`（frozenset[str]）、`locate_content_list(doc_dir: Path) -> Path`、`load_content_items(content_list_path: Path) -> list[dict[str, Any]]`、`validate_page_idx(items: list[dict[str, Any]]) -> None`、`is_artifact_complete(doc_dir: Path) -> bool`、`build_type_distribution(items: list[dict[str, Any]], backend: str) -> dict[str, Any]`、`print_report(distribution: dict[str, Any], mineru_version: str) -> None`。Task 5 依赖全部六个函数。

本模块不调用 MinerU，可脱开外部依赖独立运行。

- [ ] **Step 1: 写文件**

```python
# -*- coding: utf-8 -*-
"""产物定位、读取与校验。

不调用 MinerU——本模块只关心"磁盘上有什么"，可脱开外部依赖独立执行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import EXIT_VALIDATION, ScriptError

# docs/04 §5 记录的 3.x pipeline 后端类型集合。集合外的类型意味着可能有内容被无声
# 跳过，必须停止并人工决策，不得静默忽略。
KNOWN_BLOCK_TYPES = frozenset({
    "text", "image", "table", "chart", "equation", "code", "list",
    "header", "footer", "page_number", "aside_text", "page_footnote", "discarded",
})


def locate_content_list(doc_dir: Path) -> Path:
    """容错定位 content_list.json。

    两个都不能写死：目录层级随版本变（3.4.5 是 {pdf名}/{auto|txt}/），文件名也带
    PDF 名前缀（{pdf名}_content_list.json）。故用 * 兜住前缀、rglob 兜住层级。
    不能写成 *content_list*.json——那会连 _content_list_v2.json 一起捞进来。
    """
    hits = sorted(doc_dir.rglob("*content_list.json"))
    if not hits:
        raise ScriptError(
            EXIT_VALIDATION,
            f"未找到 content_list.json（目录：{doc_dir}）\n"
            f"       MinerU 输出契约可能已变化，请人工检查产物目录结构。")
    if len(hits) > 1:
        print(f"WARN: 发现 {len(hits)} 个 content_list.json，取 {hits[0]}，请人工确认")
    return hits[0]


def load_content_items(content_list_path: Path) -> list[dict[str, Any]]:
    """读取并做最基本的完整性校验。空产物一律视为失败，不得冒充成功。"""
    try:
        data = json.loads(content_list_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScriptError(EXIT_VALIDATION, f"content_list.json 不是合法 JSON：{exc}") from exc
    if not isinstance(data, list):
        raise ScriptError(EXIT_VALIDATION, f"content_list.json 顶层不是数组：{content_list_path}")
    if not data:
        raise ScriptError(EXIT_VALIDATION,
                          f"content_list.json 为空：{content_list_path}\n"
                          f"       拒绝写出空产物冒充成功。")
    return data


def validate_page_idx(items: list[dict[str, Any]]) -> None:
    """页码是引用功能的物理前提。缺失即退出码 2，禁止以 0/null 填充（docs/04 §11）。"""
    missing = [index for index, item in enumerate(items) if "page_idx" not in item]
    if missing:
        raise ScriptError(
            EXIT_VALIDATION,
            f"content_list 中有 {len(missing)} 个块缺少 page_idx（索引示例：{missing[:5]}）\n"
            f"       禁止以 0/null 填充后继续——那会让引用指向错误的页码。")


def is_artifact_complete(doc_dir: Path) -> bool:
    """判定产物是否可跳过。这是探针而非错误吞噬：产物损坏即视为未完成并重跑。"""
    if not doc_dir.is_dir():
        return False
    try:
        return len(load_content_items(locate_content_list(doc_dir))) > 0
    except (ScriptError, OSError):
        return False


def build_type_distribution(items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    """统计类型分布，用于核对 MinerU 实际输出契约（docs/04 §15 待验证项 P1）。"""
    counts: dict[str, int] = {}
    for item in items:
        block_type = item.get("type", "<缺失>")
        counts[block_type] = counts.get(block_type, 0) + 1
    pages = [item["page_idx"] for item in items]
    return {
        "backend": backend,
        "total_blocks": len(items),
        "type_counts": counts,
        "page_idx_range": [min(pages), max(pages)],
        "unknown_types": sorted(t for t in counts if t not in KNOWN_BLOCK_TYPES),
    }


def print_report(distribution: dict[str, Any], mineru_version: str) -> None:
    """按固定格式打印类型分布报告（契约见 specs/001-mineru-pdf-parse/contracts/cli.md）。"""
    print(f"--- 类型分布 (backend={distribution['backend']}, mineru={mineru_version}) ---")
    for block_type, count in sorted(distribution["type_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {block_type:<20s} {count:>6d}")
    print(f"  {'合计':<19s} {distribution['total_blocks']:>6d}")
    print(f"  page_idx 范围        {distribution['page_idx_range']}")
    unknown = distribution["unknown_types"]
    print(f"  未知类型             {'、'.join(unknown) if unknown else '无'}")
```

- [ ] **Step 2: 验证输出与 Task 1 基线逐字一致**

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe - <<'EOF' > /tmp/after-artifacts.txt 2>&1
import sys
from pathlib import Path
sys.path.insert(0, "backend")
from parse import artifacts as A

doc_dir = Path("data/parsed/d6da41b5d356")
path = A.locate_content_list(doc_dir)
print("定位到:", path.name)
items = A.load_content_items(path)
A.validate_page_idx(items)
dist = A.build_type_distribution(items, "pipeline")
print("块数:", dist["total_blocks"])
print("类型分布:", dict(sorted(dist["type_counts"].items())))
print("未知类型:", dist["unknown_types"])
print("可跳过:", A.is_artifact_complete(doc_dir))
EOF
diff docs/superpowers/plans/baseline-artifacts.txt /tmp/after-artifacts.txt && echo "基线一致 ✅"
```

Expected: `基线一致 ✅`（`diff` 无输出）

- [ ] **Step 3: 验证校验失败路径仍抛正确退出码**

三条异常路径都必须真的被执行到。空产物用临时文件构造，不要靠"文件恰好不存在"
来蒙混过去：

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe - <<'EOF'
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "backend")
from parse import artifacts as A, EXIT_VALIDATION, ScriptError

def expect_validation_error(label, fn):
    try:
        fn()
    except ScriptError as exc:
        assert exc.exit_code == EXIT_VALIDATION, f"{label}: 退出码 {exc.exit_code} != 2"
        print(f"{label} -> 退出码 {exc.exit_code} ✅")
    else:
        print(f"{label} -> FAIL: 没抛异常 ❌")

tmp = Path(tempfile.mkdtemp())

# 1) 缺 page_idx
expect_validation_error("缺 page_idx", lambda: A.validate_page_idx([{"type": "text"}]))

# 2) 空产物（合法 JSON 的空数组）
empty = tmp / "empty_content_list.json"
empty.write_text("[]", encoding="utf-8")
expect_validation_error("空产物", lambda: A.load_content_items(empty))

# 3) 非法 JSON
broken = tmp / "broken_content_list.json"
broken.write_text("{ not json", encoding="utf-8")
expect_validation_error("非法 JSON", lambda: A.load_content_items(broken))

# 4) 顶层不是数组
notlist = tmp / "obj_content_list.json"
notlist.write_text('{"a": 1}', encoding="utf-8")
expect_validation_error("顶层非数组", lambda: A.load_content_items(notlist))

# 5) 目录不存在不得抛异常，应静默返回 False
assert A.is_artifact_complete(Path("data/parsed/does-not-exist")) is False
print("不存在的目录 -> is_artifact_complete=False ✅")

# 6) 损坏产物所在的目录同样返回 False，而不是抛出去
assert A.is_artifact_complete(tmp) is False
print("只有损坏产物的目录 -> is_artifact_complete=False ✅")
EOF
```

Expected:

```
缺 page_idx -> 退出码 2 ✅
空产物 -> 退出码 2 ✅
非法 JSON -> 退出码 2 ✅
顶层非数组 -> 退出码 2 ✅
不存在的目录 -> is_artifact_complete=False ✅
只有损坏产物的目录 -> is_artifact_complete=False ✅
```

---

### Task 4: 新建 `backend/parse/mineru_env.py`

**Files:**
- Create: `backend/parse/mineru_env.py`

**Interfaces:**
- Consumes: `parse.EXIT_ARG`、`parse.EXIT_DEPENDENCY`、`parse.REPO_ROOT`、`parse.ScriptError`
- Produces: `write_mineru_config(config_path: Path, pipeline_models: Path, vlm_models: Path) -> None`、`build_mineru_env(config_path: Path) -> dict[str, str]`、`check_weights(backend: str, pipeline_models: Path, vlm_models: Path) -> Path`、`resolve_mineru_exe() -> Path`、`probe_mineru_version(mineru_exe: Path, env: dict[str, str]) -> str`、`run_mineru(mineru_exe: Path, pdf_path: Path, out_dir: Path, backend: str, env: dict[str, str]) -> None`。Task 5 依赖全部六个函数。

本模块不读写产物目录。

- [ ] **Step 1: 写文件**

```python
# -*- coding: utf-8 -*-
"""MinerU 外部依赖：配置写入、环境构造、可执行文件定位、子进程调用。

不读写产物目录——本模块只关心"怎么把 MinerU 跑起来"。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import EXIT_ARG, EXIT_DEPENDENCY, REPO_ROOT, ScriptError


def write_mineru_config(config_path: Path, pipeline_models: Path, vlm_models: Path) -> None:
    """写项目内 MinerU 配置。只写模型路径与来源，不写任何凭据（constitution 原则 III）。"""
    payload = {"models-dir": {"pipeline": str(pipeline_models), "vlm": str(vlm_models)},
               "model-source": "local", "config_version": "1.3.2"}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")


def build_mineru_env(config_path: Path) -> dict[str, str]:
    """构造子进程环境。只在副本上叠加，不修改当前进程与系统环境变量。"""
    env = dict(os.environ)
    env["MINERU_TOOLS_CONFIG_JSON"] = str(config_path.resolve())
    env["MINERU_MODEL_SOURCE"] = "local"  # 强制离线，禁止联网兜底
    env["MINERU_DEVICE_MODE"] = "cpu"     # 本机无 NVIDIA 显卡；-d 参数自 3.0.0 已删除
    return env


def check_weights(backend: str, pipeline_models: Path, vlm_models: Path) -> Path:
    """校验所选后端的权重目录存在。不存在即失败——拒绝联网下载（docs/04 §11）。"""
    chosen = pipeline_models if backend == "pipeline" else vlm_models
    if not chosen.is_dir():
        raise ScriptError(
            EXIT_ARG,
            f"{backend} 后端权重目录不存在：{chosen}\n"
            f"       拒绝联网下载。请先手动下载权重，或用选项指向已有目录。")
    return chosen


def resolve_mineru_exe() -> Path:
    """按受支持运行时派生 mineru 可执行文件路径（constitution 原则 I）。"""
    bindir = Path(sys.executable).parent
    for candidate in (bindir / "Scripts" / "mineru.exe", bindir / "mineru.exe",
                      bindir / "Scripts" / "mineru", bindir / "mineru"):
        if candidate.exists():
            return candidate
    found = shutil.which("mineru")
    if found:
        return Path(found)
    raise ScriptError(
        EXIT_DEPENDENCY,
        f"未找到 mineru 可执行文件（已查找 {bindir} 及其 Scripts 子目录）\n"
        f"       请先安装：{sys.executable} -m pip install -r {REPO_ROOT / 'requirements.txt'}")


def probe_mineru_version(mineru_exe: Path, env: dict[str, str]) -> str:
    """取 MinerU 版本号，用于产物溯源与 docs/04 §12 的参数回填（FR-014）。"""
    try:
        proc = subprocess.run([str(mineru_exe), "-v"], env=env,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<探测失败: {exc.__class__.__name__}>"
    output = (proc.stdout or proc.stderr or "").strip()
    return output.splitlines()[0] if output else "<未知>"


def run_mineru(mineru_exe: Path, pdf_path: Path, out_dir: Path, backend: str,
               env: dict[str, str]) -> None:
    """子进程调用 MinerU。崩溃不影响主进程，退出码被精确映射。"""
    command = [str(mineru_exe), "-p", str(pdf_path), "-o", str(out_dir), "-b", backend]
    print(f"      执行: {' '.join(command)}")
    try:
        proc = subprocess.run(command, env=env)
    except OSError as exc:
        raise ScriptError(EXIT_DEPENDENCY, f"无法启动 MinerU 子进程：{exc}") from exc
    if proc.returncode != 0:
        raise ScriptError(EXIT_DEPENDENCY, f"MinerU 子进程退出码 {proc.returncode}")
```

- [ ] **Step 2: 验证纯函数行为**

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "backend")
from parse import mineru_env as M, EXIT_ARG, EXIT_DEPENDENCY, ScriptError

# 环境变量：三项必须都设上，且不污染 os.environ
env = M.build_mineru_env(Path(".mineru/mineru.json"))
import os
print("MINERU_MODEL_SOURCE:", env["MINERU_MODEL_SOURCE"])
print("MINERU_DEVICE_MODE :", env["MINERU_DEVICE_MODE"])
print("CONFIG_JSON 结尾   :", env["MINERU_TOOLS_CONFIG_JSON"].endswith(".mineru\\mineru.json") or
                              env["MINERU_TOOLS_CONFIG_JSON"].endswith(".mineru/mineru.json"))

# 权重校验：存在的目录返回自身
w = M.check_weights("pipeline",
                    Path(r"C:\Users\Lenovo\.cache\modelscope\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master"),
                    Path(r"C:\Users\Lenovo\.cache\modelscope\models\OpenDataLab--MinerU2.5-Pro-2605-1.2B\snapshots\master"))
print("权重目录存在 ->", w.is_dir())

# 权重校验：不存在的目录必须抛退出码 1
try:
    M.check_weights("pipeline", Path("Z:/nope"), Path("Z:/nope"))
    print("FAIL: 没抛异常")
except ScriptError as exc:
    assert exc.exit_code == EXIT_ARG
    print("权重缺失 -> 退出码", exc.exit_code, "✅")

# exe 定位
print("mineru exe:", M.resolve_mineru_exe().name)
EOF
```

Expected:

```
MINERU_MODEL_SOURCE: local
MINERU_DEVICE_MODE : cpu
CONFIG_JSON 结尾   : True
权重目录存在 -> True
权重缺失 -> 退出码 1 ✅
mineru exe: mineru.exe
```

---

### Task 5: 把 `backend/parse_pdf.py` 改写成薄入口

**Files:**
- Modify: `backend/parse_pdf.py`（整体替换）

**Interfaces:**
- Consumes: Task 2 的 `parse` 包常量与 `ScriptError`；Task 3 的 `locate_content_list` `load_content_items` `validate_page_idx` `is_artifact_complete` `build_type_distribution` `print_report`；Task 4 的 `write_mineru_config` `build_mineru_env` `check_weights` `resolve_mineru_exe` `probe_mineru_version` `run_mineru`
- Produces: `compute_doc_id(pdf_path: Path) -> str`、`discover_documents(input_path: Path) -> list[tuple[Path, str]]`、`process_one(pdf_path: Path, doc_id: str, args: argparse.Namespace, mineru_exe: Path, mineru_version: str, env: dict[str, str]) -> str`、`parse_args(argv: list[str] | None = None) -> argparse.Namespace`、`main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: 整体替换文件内容**

先用 `backend/parse_pdf.py.bak` 确认原文件 338 行仍可读，再写新内容。

**注意**：下面的 import 块是本计划**修正后**的版本。初稿误判 `-m` 不可能运行而删掉了
`try/except` 兜底，导致回归；修正理由见本文件开头「关于导入方式」。

```python
# -*- coding: utf-8 -*-
"""S2 解析：用本机 MinerU 把 PDF 解析为带页码的结构化产物。

输入   data/source_data/*.pdf
输出   data/parsed/{doc_id}/{pdf名}/{auto|txt}/{pdf名}_content_list.json
       doc_id = PDF 内容 sha256 前 12 位；每项带 page_idx（页码）
       middle.json / *.md / *.pdf 为调试产物，不入管线
配置   .mineru/mineru.json（项目内配置，不碰用户全局配置）

执行   D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py [INPUT] [选项]
       --dry-run 只看将要做什么   --force 强制重解析   --backend vlm-engine
退出码 0 成功 / 1 参数错误 / 2 校验失败 / 3 外部依赖不可用

本文件是入口薄壳：输入发现 + 编排 + 命令行。共享常量见 parse/__init__.py，
MinerU 外部依赖见 parse/mineru_env.py，产物操作见 parse/artifacts.py。
决策与偏离记录见 specs/001-mineru-pdf-parse/ 与 docs/04，此处不复述。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# 两种调用方式都要支持：
#   `python -m backend.parse_pdf`  __package__ == "backend" → 用相对导入
#   `python backend/parse_pdf.py`  __package__ 为空，sys.path[0] 为 backend/ → 用绝对导入
# backend/ 虽无 __init__.py，但在 Python 3 是隐式命名空间包，`-m` 是能导入的，
# 所以上面的相对导入分支不是死代码，不能删。
try:
    from .parse import BACKENDS, DEFAULT_BACKEND, DEFAULT_CONFIG_PATH
    from .parse import DEFAULT_PIPELINE_MODELS, DEFAULT_VLM_MODELS
    from .parse import EXIT_ARG, EXIT_OK, PARSED_DIR, REPO_ROOT, SOURCE_DIR, ScriptError
    from .parse import artifacts, mineru_env
except ImportError:
    from parse import BACKENDS, DEFAULT_BACKEND, DEFAULT_CONFIG_PATH
    from parse import DEFAULT_PIPELINE_MODELS, DEFAULT_VLM_MODELS
    from parse import EXIT_ARG, EXIT_OK, PARSED_DIR, REPO_ROOT, SOURCE_DIR, ScriptError
    from parse import artifacts, mineru_env


# --- 输入发现 --------------------------------------------------------------- #

def compute_doc_id(pdf_path: Path) -> str:
    """doc_id = 文件内容 sha256 前 12 位；内容一变 doc_id 即变（docs/04 §10.1）。"""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def discover_documents(input_path: Path) -> list[tuple[Path, str]]:
    """返回 [(pdf 路径, doc_id), ...]，按内容去重。"""
    if input_path.is_dir():
        candidates = sorted(p for p in input_path.iterdir() if p.suffix.lower() == ".pdf")
        if not candidates:
            raise ScriptError(EXIT_ARG, f"目录中没有 PDF 文件：{input_path}")
    elif input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ScriptError(EXIT_ARG, f"不是 PDF 文件：{input_path}")
        candidates = [input_path]
    else:
        raise ScriptError(EXIT_ARG, f"路径不存在：{input_path}")

    unique: dict[str, Path] = {}
    for path in candidates:
        doc_id = compute_doc_id(path)
        if doc_id in unique:
            print(f"WARN: {path.name} 与 {unique[doc_id].name} 内容相同，按同一文档处理，跳过")
            continue
        unique[doc_id] = path
    return [(path, doc_id) for doc_id, path in sorted(unique.items())]


# --- 单文档流程与命令行 ------------------------------------------------------ #

def process_one(pdf_path: Path, doc_id: str, args: argparse.Namespace, mineru_exe: Path,
                mineru_version: str, env: dict[str, str]) -> str:
    """处理一份文档，返回 'parsed' 或 'skipped'。"""
    doc_dir = PARSED_DIR / doc_id
    if not args.force and artifacts.is_artifact_complete(doc_dir):
        return "skipped"
    if doc_dir.exists():
        shutil.rmtree(doc_dir)  # 清理上次中途失败的残留，避免半份产物被误判为完成
    doc_dir.mkdir(parents=True, exist_ok=True)

    mineru_env.run_mineru(mineru_exe, pdf_path, doc_dir, args.backend, env)

    items = artifacts.load_content_items(artifacts.locate_content_list(doc_dir))
    artifacts.validate_page_idx(items)
    distribution = artifacts.build_type_distribution(items, args.backend)
    artifacts.print_report(distribution, mineru_version)

    if distribution["unknown_types"]:
        raise ScriptError(
            artifacts.EXIT_VALIDATION,
            f"出现 docs/04 §5 未覆盖的块类型：{'、'.join(distribution['unknown_types'])}\n"
            f"       停止并需人工决策——未知类型意味着可能有内容被无声跳过。")
    print(f"      产物: {doc_dir}")
    return "parsed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本机 MinerU 解析 PDF，产出带页码的结构化内容（S2 解析步骤）")
    parser.add_argument("input", nargs="?", default=str(SOURCE_DIR),
                        help=f"PDF 文件或目录（默认：{SOURCE_DIR}）")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help=f"MinerU 后端（默认 {DEFAULT_BACKEND}；可选 {'/'.join(BACKENDS)}）")
    parser.add_argument("--force", action="store_true", help="强制重新解析，忽略已有产物")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="项目内配置文件路径")
    parser.add_argument("--pipeline-models", default=str(DEFAULT_PIPELINE_MODELS),
                        help="pipeline 权重目录")
    parser.add_argument("--vlm-models", default=str(DEFAULT_VLM_MODELS), help="vlm 权重目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做什么，不调用 MinerU")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.backend not in BACKENDS:
            raise ScriptError(
                EXIT_ARG, f"--backend 取值非法：{args.backend}（可选：{'/'.join(BACKENDS)}）")
        pipeline_models, vlm_models = Path(args.pipeline_models), Path(args.vlm_models)
        weights = mineru_env.check_weights(args.backend, pipeline_models, vlm_models)
        documents = discover_documents(Path(args.input))

        if args.dry_run:
            print(f"仓库根 {REPO_ROOT}\n后端   {args.backend}（权重 {weights}）")
            print(f"配置   {args.config}\n产物   {PARSED_DIR}\n待处理 {len(documents)} 份")
            for pdf_path, doc_id in documents:
                print(f"  doc_id={doc_id}  {pdf_path.name}")
            print("（dry-run：未写配置、未调用 MinerU）")
            return EXIT_OK

        mineru_exe = mineru_env.resolve_mineru_exe()
        mineru_env.write_mineru_config(Path(args.config), pipeline_models, vlm_models)
        env = mineru_env.build_mineru_env(Path(args.config))
        mineru_version = mineru_env.probe_mineru_version(mineru_exe, env)

        print(f"MinerU {mineru_version} | 后端 {args.backend} | 待处理 {len(documents)} 份\n")
        failures: list[int] = []
        for index, (pdf_path, doc_id) in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] doc_id={doc_id}  file={pdf_path.name}")
            try:
                status = process_one(pdf_path, doc_id, args, mineru_exe, mineru_version, env)
            except ScriptError as exc:
                failures.append(exc.exit_code)
                print(f"ERROR: {exc}", file=sys.stderr)
                continue
            if status == "skipped":
                print("      status=skipped (产物已存在且完整)")
            print()

        # 任一文档失败即返回该失败码，不因后续文档成功而回退为 0
        return max(failures) if failures else EXIT_OK
    except ScriptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
```

注意 `process_one` 里那处 `artifacts.EXIT_VALIDATION`——`EXIT_VALIDATION`
不在入口的 import 清单内，故经由 `artifacts` 模块取用，避免多加一行 import。

- [ ] **Step 2: 验证 dry-run 输出与基线逐字一致**

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe backend/parse_pdf.py --dry-run \
    > /tmp/after-dryrun.txt 2>&1
diff docs/superpowers/plans/baseline-dryrun.txt /tmp/after-dryrun.txt && echo "dry-run 基线一致 ✅"
```

Expected: `dry-run 基线一致 ✅`

若 diff 出 `仓库根`/`后端` 等行有差异，多半是 `REPO_ROOT` 层数问题，回 Task 2 Step 2 复查。

- [ ] **Step 3: 验证参数校验路径仍返回正确退出码**

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe backend/parse_pdf.py --backend bogus; echo "退出码=$?"
PYTHONIOENCODING=utf-8 rag/python.exe backend/parse_pdf.py Z:/nope.pdf; echo "退出码=$?"
```

Expected：

```
ERROR: --backend 取值非法：bogus（可选：pipeline/vlm-engine）
退出码=1
ERROR: 路径不存在：Z:\nope.pdf
退出码=1
```

- [ ] **Step 4: 确认入口行数已下降**

```bash
cd /d/zg6_Project/9/med_rag
wc -l backend/parse_pdf.py backend/parse/__init__.py backend/parse/artifacts.py backend/parse/mineru_env.py
```

Expected（入口应远小于原 338 行，各模块均在 40–140 行区间）：

```
  ~150 backend/parse_pdf.py
   ~45 backend/parse/__init__.py
  ~110 backend/parse/artifacts.py
   ~85 backend/parse/mineru_env.py
```

---

### Task 6: 端到端验证并清理

**Files:**
- Delete: `backend/parse_pdf.py.bak`（仅在全部验证通过后）

**Interfaces:**
- Consumes: Task 5 的完整入口
- Produces: 无（收尾）

- [ ] **Step 1: 完整跑一次，确认走 skipped 且退出码 0**

产物已存在且完整，本条不会重新解析，秒回：

```bash
cd /d/zg6_Project/9/med_rag
PYTHONIOENCODING=utf-8 rag/python.exe backend/parse_pdf.py; echo "退出码=$?"
```

Expected：

```
MinerU mineru, version 3.4.5 | 后端 pipeline | 待处理 1 份

[1/1] doc_id=d6da41b5d356  file=国家基层高血压防治管理指南2025版.pdf
      status=skipped (产物已存在且完整)

退出码=0
```

版本号那行是 `MinerU mineru, version 3.4.5`，不是 `MinerU 3.4.5`——这是 MinerU 3.4.5
的 `-v` 输出的原样首行，脚本只是把它接在 `MinerU ` 之后。

（`MinerU` 后的版本号取自实测；若显示 `<探测失败: …>` 说明 mineru.exe 没找到，
不要继续，先排查安装。）

- [ ] **Step 2: 确认产物目录未被写坏**

```bash
cd /d/zg6_Project/9/med_rag
find data/parsed -name "*content_list.json" | head
ls -d backend/data 2>&1
```

Expected：第一行列出 `data/parsed/d6da41b5d356/.../…_content_list.json`；
第二行必须是 `No such file or directory`——**若 `backend/data` 真的存在，
说明 REPO_ROOT 推断错了，产物被写到了错误位置，立即停止并回 Task 2。**

- [ ] **Step 3: 清理备份**

只有前面每一步都通过才执行：

```bash
cd /d/zg6_Project/9/med_rag
rm backend/parse_pdf.py.bak
rm -rf backend/__pycache__ backend/parse/__pycache__
ls backend/
```

Expected: `parse  parse_pdf.py`

（`rm -rf` 而非 `rmdir`：`__pycache__` 里有 `.pyc`，目录非空，`rmdir` 会失败。
缓存目录删掉即可，下次运行会自动重建。）

---

## 已知遗留

按设计文档决定，本次**不同步更新** `specs/001-mineru-pdf-parse/` 中四处会变陈旧的
描述性文字：`plan.md:11`、`plan.md:76`、`tasks.md:7`、`tasks.md:49`。
`contracts/cli.md` 与 `quickstart.md` 不受影响，无需改动。
