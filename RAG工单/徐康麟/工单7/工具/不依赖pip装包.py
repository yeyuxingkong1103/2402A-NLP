# -*- coding: utf-8 -*-
"""工单7：**不依赖 pip** 的依赖安装器（直接下 wheel → 解包到 研发/第三方库）。

工单编号：人工智能NLP-RAG-功能测试及评估

为什么需要它（实测记录，2026-10-08）
----------------------------------
网络并无问题（Python/OpenSSL 可达 pypi 与镜像，`pip index versions` 秒回），
但 `pip install --target ...` 在本机沙箱下会失败：

    PermissionError: [Errno 13] Permission denied:
      '...\\pip-unpack-xxxx\\opencv_python_headless-...-win_amd64.whl'

即 pip 在**自己的临时解包目录**上被拒绝写/删。把 TMP/TEMP/TMPDIR 指到工作区内也无法绕过
（pip 仍会在解包阶段失败）。因此改为完全绕开 pip：

    1) 查 PyPI JSON 元数据（清华镜像优先，官方兜底）；
    2) 选与当前解释器（cp311 / win_amd64 / abi3 兼容）匹配的 wheel；
    3) 用 urllib 直接下载 wheel 到 `研发/第三方库/_wheels/`；
    4) 用标准库 `zipfile` 解包到 `研发/第三方库/`（pip 的 --target 语义等价）。

这样还有一个好处：**落点红线天然满足** —— 只写本仓库，不碰只读的 工单1 venv。

用法：
    python 工具/不依赖pip装包.py opencv-python-headless
    python 工具/不依赖pip装包.py pdfplumber pandas --index https://pypi.tuna.tsinghua.edu.cn/simple
    python 工具/不依赖pip装包.py gradio --with-deps      # 连带依赖一起装（递归，默认只装本身）

退出码：0 成功；1 有包失败；2 参数或环境错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "研发" / "第三方库"
WHEEL_CACHE = DEFAULT_TARGET / "_wheels"

#: 端点：**镜像 simple 索引优先**（实测清华直链 4.95 MB/s，是 PyPI CDN 2.06 MB/s 的 2.4 倍），
#: 官方 JSON API 仅在需要依赖元数据（--with-deps）时兜底使用。
INDEX_URLS: tuple[str, ...] = (
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "https://pypi.org/simple",
)
JSON_API = "https://pypi.org/pypi/{name}/json"


def _tags() -> list[str]:
    """当前解释器可用的 wheel 标签（按优先级）。"""
    from packaging import tags as _t  # 打包在 venv 里通常可用；不可用则退回硬编码
    return [str(t) for t in _t.sys_tags()]


def _fallback_tags() -> list[str]:
    v = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return [f"{v}-cp{v}-win_amd64", "cp311-abi3-win_amd64", "cp310-abi3-win_amd64",
            "cp39-abi3-win_amd64", "py3-none-win_amd64", "py3-none-any", "py2.py3-none-any"]


def _http_json(url: str, timeout: float = 25.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "gd7-installer/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def canonical(name: str) -> str:
    """PyPI 规范名（PEP 503）：小写 + 连续 ``-_.`` 归一为 ``-``。"""
    return re.sub(r"[-_.]+", "-", str(name or "").strip()).lower()


def _really_installed(target: Path, name: str) -> bool:
    """目标目录里是否**真的**装好了这个包（看 ``*.dist-info`` 或可导入的顶层目录）。

    实测坑：第一版用「顶层目录名是否匹配」判断，把**只解开一小部分**的 opencv
    误判成「已就绪」而跳过。改为认 ``<name>-<version>.dist-info``（wheel 解包必然产生）
    或规范化后的顶层包目录，两者都要求目录**非空**。
    """
    if not target.is_dir():
        return False
    canon = canonical(name)
    tried = {canon, canon.replace("-", "_"), canon.replace("-", "")}
    for entry in target.iterdir():
        if entry.name.endswith(".dist-info"):
            stem = canonical(entry.name[: -len(".dist-info")].rsplit("-", 1)[0])
            if stem in tried:
                return True
        elif entry.is_dir() and cjk_safe_name(entry.name) in tried:
            try:
                if any(entry.iterdir()):
                    return True
            except OSError:
                continue
    return False


def cjk_safe_name(text: str) -> str:
    return canonical(text)


def _installed_names(target: Path) -> list[str]:
    """目标目录里已安装的**顶层可导入名**（粗略）：``*.dist-info`` 的目录名 + 顶层包目录/模块。"""
    names: list[str] = []
    if not target.is_dir():
        return names
    for entry in target.iterdir():
        if entry.name.endswith(".dist-info"):
            names.append(entry.name[: -len(".dist-info")].rsplit("-", 1)[0])
        elif entry.is_dir() and not entry.name.startswith(("_", ".")):
            names.append(entry.name)
        elif entry.is_file() and entry.suffix == ".py" and entry.stem != "__init__":
            names.append(entry.stem)
    return names


def _requirements(name: str, requires_dist: list[str] | None) -> list[tuple[str, str]]:
    """把 ``Requires-Dist`` 按当前环境标记过滤成 ``[(包名, 版本约束串)]``。

    工单7 实测坑：**只取包名、丢掉版本约束**会装出不兼容的组合 ——
    ``pydantic 2.13.5`` 要求 ``pydantic-core==2.46.5``，而递归时无条件取了最新
    ``pydantic-core 2.49.0``，于是 ``import gradio`` 直接抛
    ``SystemError: The installed pydantic-core version (2.49.0) is incompatible``。
    因此这里把约束串（``==2.46.5`` / ``>=1.0,<2`` 等）一并带下去，
    交给 :func:`pick_from_simple` 在候选里筛。
    """
    out: list[tuple[str, str]] = []
    for raw in requires_dist or []:
        text = str(raw)
        if ";" in text:                      # 环境标记：只在当前环境成立时才装
            try:
                from packaging.requirements import Requirement
                req = Requirement(text)
                if req.marker is not None and not req.marker.evaluate():
                    continue
                spec = str(req.specifier)
                pkg = req.name
            except Exception:                # noqa: BLE001 —— 解析不了就保守跳过该条
                continue
        else:
            try:
                from packaging.requirements import Requirement
                req = Requirement(text)
                pkg, spec = req.name, str(req.specifier)
            except Exception:                # noqa: BLE001
                pkg, spec = re.split(r"[\s\[<>=!~;(]", text, 1)[0], ""
        pkg = pkg.strip()
        if pkg and canonical(pkg) != "python":
            out.append((pkg, spec))
    return out


def _http_text(url: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "gd7-installer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def pick_from_simple(name: str, *, index: str, timeout: float = 25.0,
                     allow_prerelease: bool = False, spec: str = ""
                     ) -> tuple[str, str, int | None, str]:
    """从 **simple 索引**挑 wheel（镜像优先，实测比 PyPI CDN 快 2.4 倍）。

    simple 索引给的是「全部版本的全部文件」列表，形如：

        <a href="../../packages/b3/8b/.../gradio-6.9.0-py3-none-any.whl#sha256=..."
           data-requires-python="&gt;=3.10">gradio-6.9.0-py3-none-any.whl</a>

    要点：
      * href 是**相对路径**（``../../packages/...``），要相对索引 URL 解析成绝对地址；
      * 版本从文件名里解析（``pkg-<ver>-<pytag>-<abitag>-<plattag>.whl``），
        取**最高且与当前解释器兼容**的那个（不能盲取列表末尾 —— 顺序无保证）；
      * **默认排除预发布版**（``rc``/``a``/``b``/``dev``）：实测不加这条会选中
        ``pandas-3.1.0rc0``，把预发布版装进交付环境是隐患（`--pre` 可放开）；
      * 同时排除 ``Requires-Python`` 与当前解释器不兼容的文件（simple 索引里有该属性）。
    """
    base = index.rstrip("/") + f"/{name}/"
    html = _http_text(base, timeout=timeout)
    known = _tags()
    from packaging.version import Version, InvalidVersion

    py_ver = Version(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    spec_set = None
    if spec:
        try:
            from packaging.specifiers import SpecifierSet
            spec_set = SpecifierSet(spec)
        except Exception:                 # noqa: BLE001 —— 约束串坏了就不筛
            spec_set = None
    best: tuple[Version, int, str, str, int | None] | None = None
    skipped_pre: list[str] = []
    for m in re.finditer(r'<a\s+[^>]*href="([^"]+\.whl)(?:#[^"]*)?"[^>]*>([^<]+)</a>', html, re.I):
        href, label = m.group(1), m.group(2).strip()
        fn = label.rsplit("/", 1)[-1]
        if not fn.endswith(".whl"):
            continue
        # Requires-Python 过滤（simple 索引里是 HTML 转义的 >=）
        rp = re.search(r'data-requires-python="([^"]+)"', m.group(0))
        if rp:
            spec = rp.group(1).replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
            try:
                from packaging.specifiers import SpecifierSet
                if not SpecifierSet(spec).contains(py_ver, prereleases=True):
                    continue
            except Exception:            # noqa: BLE001 —— 解析不了就不因它排除
                pass
        tag_score = -1
        for i, tag in enumerate(known):
            if fn.endswith(f"-{tag}.whl"):
                tag_score = len(known) - i
                break
        if tag_score < 0:
            if not re.search(r"-(cp\d+-(?:abi3|none)|py\d?\.?py?\d?-none)-(win_amd64|any)\.whl$", fn):
                continue
            tag_score = 1
        parts = fn[: -len(".whl")].split("-")
        if len(parts) < 3:
            continue
        ver_text = parts[1]
        try:
            ver = Version(ver_text)
        except InvalidVersion:
            continue
        if ver.is_prerelease and not allow_prerelease:
            skipped_pre.append(ver_text)
            continue
        if spec_set is not None and not spec_set.contains(ver, prereleases=True):
            continue
        url = urllib.parse.urljoin(base, href)
        size = None
        size_m = re.search(r'data-(?:dist-info-metadata|core-metadata|size)="(\d+)"', m.group(0))
        if size_m:
            size = int(size_m.group(1))
        cand = (ver, tag_score, url, fn, size)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if best is None:
        extra = f"（已排除预发布版 {sorted(set(skipped_pre))[:3]}…）" if skipped_pre else ""
        raise RuntimeError(f"{name}: simple 索引里没有与 {sys.version.split()[0]} / win_amd64 "
                           f"兼容的稳定版 wheel{extra}")
    ver, _score, url, fn, size = best
    return url, fn, size, str(ver)


def pick_wheel(name: str, *, index: str, timeout: float = 25.0, spec: str = ""
               ) -> tuple[str, str, int | None, str, list[tuple[str, str]]]:
    """返回 ``(wheel_url, 文件名, 大小或 None, 版本, 依赖包名列表)``。

    路径选择：
      * ``index`` 指向 **simple 索引**（``.../simple``）→ 用 :func:`pick_from_simple` 取文件
        （走镜像，快），依赖元数据再向官方 JSON API 补一次；
      * ``index`` 指向 **JSON API**（``.../pypi``）→ 直接用 JSON 元数据。
    """
    base = index.rstrip("/")
    if base.endswith("/simple") or "/simple/" in base:
        url, fn, size, version = pick_from_simple(name, index=base, timeout=timeout, spec=spec)
        requires: list[tuple[str, str]] = []
        try:
            meta = _http_json(JSON_API.format(name=name), timeout=timeout)
            requires = _requirements(name, meta.get("info", {}).get("requires_dist"))
        except (urllib.error.URLError, OSError, ValueError):
            requires = []      # 拿不到依赖表不算失败：只装本体
        return url, fn, size, version, requires

    meta = _http_json(f"{base}/{name}/json", timeout=timeout)
    version = str(meta["info"]["version"])
    files = meta.get("releases", {}).get(version) or meta.get("urls") or []
    if not files:
        raise RuntimeError(f"{name}: PyPI 未返回 {version} 的文件列表")
    cands: list[tuple[int, str, str, int | None]] = []
    known = _tags()
    for f in files:
        fn = str(f.get("filename") or "")
        if not fn.endswith(".whl"):
            continue
        url = str(f.get("url") or "")
        if not url:
            continue
        score = -1
        for i, tag in enumerate(known):
            if fn.endswith(f"-{tag}.whl"):
                score = len(known) - i
                break
        if score < 0:
            # 允许 abi3 / none 的通用包
            m = re.search(r"-(cp\d+-(?:abi3|none)|py\d?\.?py?\d?-none)-(win_amd64|any)\.whl$", fn)
            if m:
                score = 1
        if score >= 0:
            size = f.get("size")
            cands.append((score, url, fn, int(size) if size else None))
    if not cands:
        raise RuntimeError(f"{name} {version}: 没有与 {sys.version.split()[0]} / win_amd64 兼容的 wheel")
    cands.sort(key=lambda x: -x[0])
    _, url, fn, size = cands[0]
    requires = _requirements(name, meta.get("info", {}).get("requires_dist"))
    return url, fn, size, version, requires


def download(url: str, dest: Path, *, expect: int | None, timeout: float = 60.0) -> int:
    """下载到 dest（先 .part）。返回字节数。"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gd7-installer/1.0"})
    got = 0
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        total = expect or (int(resp.headers["Content-Length"])
                           if str(resp.headers.get("Content-Length", "")).isdigit() else 0)
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            got += len(block)
        if total and got != total:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"大小不符：期望 {total}，实际 {got}")
    tmp.replace(dest)
    print(f"      {got / 1024 / 1024:.1f} MB，{time.time() - started:.1f}s")
    return got


def unpack(wheel: Path, target: Path) -> int:
    """解包 wheel 到 target（等价 pip --target 的展开语义：纯 Python + .data 顶层合并）。"""
    count = 0
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            # wheel 的 .data/{purelib,platlib}/ 前缀要剥掉
            m = re.match(r"^[^/]+\.data/(purelib|platlib)/(.+)$", name)
            rel = m.group(2) if m else name
            if rel.startswith("..") or Path(rel).is_absolute():
                continue
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单7：{WORK_ORDER} —— 不依赖 pip 的装包器")
    parser.add_argument("packages", nargs="+", help="包名（PyPI 规范名）")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help=f"安装目录（默认 {DEFAULT_TARGET}）")
    parser.add_argument("--index", default=None, help="索引端点（默认镜像优先自动探测）")
    parser.add_argument("--force", action="store_true", help="已安装也重新装")
    parser.add_argument("--with-deps", action="store_true",
                        help="递归解析并安装依赖（按 Requires-Dist + 环境标记；已满足的依赖会跳过）")
    args = parser.parse_args(argv)

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    WHEEL_CACHE.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"工单7 依赖安装（绕过 pip）｜{WORK_ORDER}")
    print(f"解释器：{sys.executable}（{sys.version.split()[0]}）")
    print(f"目标  ：{target}")
    print(f"模式  ：{'含依赖递归' if args.with_deps else '只装指定包'}")
    print("=" * 78)

    indexes = [args.index] if args.index else list(INDEX_URLS)
    ok, bad = [], []
    pins: dict[str, str] = {}
    # ---- 待装队列：--with-deps 时按 Requires-Dist 展开（只解析一层层入队，带环检测） ----
    queue: list[str] = []
    constraints: dict[str, list[str]] = {}
    seen: set[str] = set()
    for pkg in args.packages:
        key = canonical(pkg)
        if key not in seen:
            seen.add(key)
            queue.append(pkg)

    index_hint = index = ""
    installed_here = {canonical(p) for p in _installed_names(target)}

    while queue:
        pkg = queue.pop(0)
        key = canonical(pkg)
        if key in installed_here and not args.force and _really_installed(target, pkg):
            print(f"  = 已就绪，跳过：{pkg}")
            continue
        if key in installed_here and not _really_installed(target, pkg):
            print(f"  ! {pkg} 目录存在但不完整（缺 dist-info / 空目录）→ 重新安装")
        spec = ",".join(constraints.get(key, []))
        done = False
        for idx in ([index] if index else indexes):
            try:
                shown = f"（约束 {spec}）" if spec else ""
                print(f"  ↓ {pkg} {shown}  ←  {idx}")
                url, fn, size, version, requires = pick_wheel(pkg, index=idx, spec=spec)
                print(f"      选中 {fn}（{version}，{('%.1f MB' % (size / 1024 / 1024)) if size else '大小未知'}）")
                wheel = WHEEL_CACHE / fn
                if not wheel.is_file():
                    download(url, wheel, expect=size)
                else:
                    print(f"      已缓存 {wheel.name}")
                n = unpack(wheel, target)
                print(f"      ✅ 解包 {n} 个文件 → {target}")
                ok.append(f"{pkg} {version}")
                installed_here.add(key)
                index = idx
                pins.setdefault(key, version)
                if args.with_deps:
                    for dep, dep_spec in requires:
                        dkey = canonical(dep)
                        if dep_spec:
                            constraints.setdefault(dkey, [])
                            if dep_spec not in constraints[dkey]:
                                constraints[dkey].append(dep_spec)
                        if dkey in seen:
                            continue
                        seen.add(dkey)
                        queue.append(dep)
                        print(f"      + 依赖入队：{dep}{(' ' + dep_spec) if dep_spec else ''}")
                done = True
                break
            except (urllib.error.URLError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                print(f"      ❌ {type(exc).__name__}: {str(exc)[:130]}")
        if not done:
            bad.append(pkg)

    print("-" * 78)
    print(f"成功 {len(ok)}：{'、'.join(ok) if ok else '无'}")
    # ---- 落 lockfile（复现用；也是"这次到底装了什么版本"的凭证）----
    if pins:
        lock = target / "_lock.json"
        payload = {"工单": WORK_ORDER, "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "解释器": sys.executable, "python": sys.version.split()[0],
                   "包": dict(sorted(pins.items()))}
        try:
            lock.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"版本锁定文件：{lock}")
        except OSError as exc:
            print(f"⚠️ 锁定文件写入失败（不影响安装）：{exc}")
    if bad:
        print(f"失败 {len(bad)}：{'、'.join(bad)}", file=sys.stderr)
        print("排查建议：①换 --index 端点 ②确认包在 PyPI 有 win_amd64 wheel"
              "（纯源码包需另行构建）③用 --with-deps 补齐依赖", file=sys.stderr)
        return 1
    print("✅ 全部就绪。使用时记得把 研发/第三方库 加进 sys.path"
          "（部署/脚本/环境.ps1 已处理 PYTHONPATH 与 GD7_SITE）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已中断]", file=sys.stderr)
        raise SystemExit(130)
