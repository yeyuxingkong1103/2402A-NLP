# -*- coding: utf-8 -*-
"""重新下载 PDF 回归样本并校验（data/pdf_samples/ 的复现入口）。

为什么需要它：`data/pdf_samples/` 下的 4 份 PDF 是 PDF 解析链路的回归素材，
来自公开政府网站。真实解析验收要求"样本可溯源、别人能自己拉一份出来"，
本脚本就是那个入口 —— 从原始 URL 重下，再用 PyMuPDF 校验页数与文字层，
确保下载到的东西和 `README.md` 记录的样本一致（政府站点改版/换附件时会立刻暴露）。

只读：只往 `--out-dir` 写 PDF，不读库、不写 Redis、不调任何模型 API。

用法：
    python scripts/e2e/fetch_pdf_samples.py                 # 缺哪份下哪份，然后全部校验
    python scripts/e2e/fetch_pdf_samples.py --force         # 全部重下
    python scripts/e2e/fetch_pdf_samples.py --verify-only   # 只校验本地已有文件（不联网）

退出码：0 = 全部校验通过；1 = 有样本与预期不符（页数/文字层不一致）；
        2 = 前置不满足（PyMuPDF 未安装 / 目录不存在 / 样本缺失且网络不可用）。
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

# 沙箱注入的代理会劫持外网请求（政府站点尤其容易被拦成 SSL/502），先剔除
for _name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_name, None)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "pdf_samples"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# (文件名, URL, 法规名, 预期页数, 预期文字层字符数±容差, 库内对应文档)
# 「预期值」与 data/pdf_samples/README.md 的清单表逐字对应，两处必须一起改
TARGETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    (
        "jm_labor_contract_law.pdf",
        "http://www.jiangmen.gov.cn/attachment/0/287/287575/2982353.pdf",
        "中华人民共和国劳动合同法",
        27,
        12375,
        "documents#9（HTML）",
    ),
    (
        "jm_lc_implement_regulation.pdf",
        "http://www.jiangmen.gov.cn/attachment/0/287/287576/2982353.pdf",
        "中华人民共和国劳动合同法实施条例",
        10,
        4617,
        "documents#3（HTML）",
    ),
    (
        "jm_labor_law.pdf",
        "http://www.jiangmen.gov.cn/attachment/0/287/287574/2982353.pdf",
        "中华人民共和国劳动法",
        21,
        8952,
        "documents#10（HTML）",
    ),
    (
        "nc_labor_contract_general.pdf",
        "https://www.nc.gov.cn/ncszf/shbz/202507/65ce5e3bd5f546fe8f446c374221f6d7/"
        "files/%E5%8A%B3%E5%8A%A8%E5%90%88%E5%90%8C(%20%E9%80%9A%E7%94%A8)-20240417143159399005.pdf",
        "劳动合同（通用）示范文本（人社部编制版）",
        7,
        3591,
        "无",
    ),
)

# 文字层字符数的容差：同一份 PDF 用不同 PyMuPDF 版本抽取，个别字符会有差异
TEXT_CHARS_TOLERANCE = 0.02


def build_parser() -> argparse.ArgumentParser:
    """命令行参数（默认行为：缺哪份下哪份，然后全部校验）。"""
    parser = argparse.ArgumentParser(description="重新下载并校验 data/pdf_samples/ 的 PDF 样本")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="样本目录")
    parser.add_argument("--force", action="store_true", help="即使本地已存在也重新下载")
    parser.add_argument("--verify-only", action="store_true", help="只校验本地文件，完全不联网")
    parser.add_argument("--timeout", type=int, default=120, help="单个文件下载超时秒数")
    return parser


def download(url: str, dest: Path, timeout: int) -> str | None:
    """下载到 dest，返回错误描述（None = 成功）。失败只记录不抛，方便一次看完 4 份。"""
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        dest.write_bytes(data)
        return None
    except Exception as error:  # noqa: BLE001 - 网络/站点异常种类多，统一降级为"这份失败"
        return f"{type(error).__name__}: {error}"


def inspect(path: Path, fitz_module) -> tuple[int, int, str]:
    """用 PyMuPDF 抽取页数、文字层字符数与首页片段（用于人工确认没下错文件）。

    文字层口径：**逐页 `get_text()` 字符数直接相加，不做 strip** ——
    这样与 `data/pdf_samples/README.md` 的清单表、批次 22 报告里的
    "PyMuPDF 直取 12,375 字 vs MinerU 12,250 字" 完全是同一个数。
    （逐页 strip 会每页少 1 个换行，实测 27 页差 27 字，容易误判成"样本变了"。）
    """
    document = fitz_module.open(path)
    try:
        per_page = [len(page.get_text()) for page in document]
        return document.page_count, sum(per_page), document[0].get_text().strip()[:60]
    finally:
        document.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ 缺少 PyMuPDF，先安装：pip install PyMuPDF==1.28.0")
        return 2

    out_dir: Path = args.out_dir
    if not out_dir.exists():
        if args.verify_only:
            print(f"❌ 样本目录不存在：{out_dir}")
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"样本目录：{out_dir}")
    print(f"模式：{'只校验（不联网）' if args.verify_only else ('全部重下' if args.force else '缺哪份下哪份')}")
    print()

    failures: list[str] = []
    for filename, url, label, want_pages, want_chars, mapped in TARGETS:
        dest = out_dir / filename
        print(f"--- {label}")
        print(f"    文件 {filename}    库内对应：{mapped}")

        if dest.exists() and not args.force:
            print("    下载：跳过（本地已有，--force 可强制重下）")
        elif args.verify_only:
            print("    下载：跳过（只校验模式）")
        else:
            error = download(url, dest, args.timeout)
            if error:
                print(f"    ❌ 下载失败：{error}")
                failures.append(f"{filename}: 下载失败 {error}")
                print()
                continue
            print(f"    下载：OK（{dest.stat().st_size:,} B）")

        if not dest.exists():
            print("    ❌ 文件不存在，无法校验")
            failures.append(f"{filename}: 文件不存在")
            print()
            continue

        pages, chars, head = inspect(dest, fitz)
        page_ok = pages == want_pages
        low, high = want_chars * (1 - TEXT_CHARS_TOLERANCE), want_chars * (1 + TEXT_CHARS_TOLERANCE)
        chars_ok = low <= chars <= high
        print(f"    校验：页数 {pages}（预期 {want_pages}）{'✔' if page_ok else '✗'}    "
              f"文字层 {chars:,} 字 / 逐页不 strip（预期 {want_chars:,}）{'✔' if chars_ok else '✗'}")
        print(f"    首页片段：{head!r}")
        print(f"    来源：{url}")
        if not page_ok:
            failures.append(f"{filename}: 页数 {pages} != {want_pages}")
        if not chars_ok:
            failures.append(f"{filename}: 文字层 {chars} 字偏离预期 {want_chars}（>{TEXT_CHARS_TOLERANCE:.0%}）")
        print()

    print("=" * 78)
    if failures:
        print(f"❌ 有 {len(failures)} 项与预期不符：")
        for item in failures:
            print(f"  · {item}")
        print("  提示：站点改版/换附件时，需同时更新本脚本的 TARGETS 与 data/pdf_samples/README.md")
        return 1
    print(f"✅ {len(TARGETS)} 份样本全部校验通过（页数与文字层均与 README 清单一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
