# -*- coding: utf-8 -*-
"""MinerU 文档解析调用模块。

封装 MinerU CLI（mineru-kit）的解析流程，方便在 RAG 流程中一键调用：

1. 启动服务（可选）：``mineru server start``；
2. 转换文档：``mineru-kit parse <输入.pdf> -o <输出目录> --tier flash --format zip``；
3. 读取结果：``--format zip`` 会生成一个压缩包，内含
   ``markdown.md`` / ``middle_json.json`` / ``structured_content.json`` / ``images/``。
   本模块用标准库 ``zipfile`` + ``json`` 直接读取压缩包内容，**无需解压到硬盘**。

用法示例（详见同目录 README.md）：

    from mineru_parser import parse_pdf, read_zip

    result = parse_pdf("输入文件.pdf", output_dir="outputs", tier="flash")
    print(result.markdown)            # Markdown 正文
    print(result.middle_json)         # 中间格式 JSON（dict）
    print(result.structured_content)  # 结构化内容 JSON（dict）
    for name, data in result.images:  # 提取的图片 (文件名, 字节)
        ...

    # 直接读取已有压缩包（不调用 MinerU）
    result2 = read_zip("outputs/输入文件.zip")
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag2.mineru_parser")

# MinerU 可执行文件默认路径（独立 conda 环境 mineru，见 RAG_1/config.yaml）
DEFAULT_EXE = r"D:/an/envs/mineru/Scripts/mineru-kit.exe"
DEFAULT_SERVER_EXE = r"D:/an/envs/mineru/Scripts/mineru.exe"

# 识别为图片的文件后缀
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")


class ParseResult:
    """MinerU 解析结果（对应一个 ``--format zip`` 压缩包）。

    使用标准库 ``zipfile`` 在内存中读取压缩包内容，不落盘解压。
    ``markdown`` / ``middle_json`` / ``structured_content`` / ``images``
    首次访问时读取并缓存，后续访问零开销。
    """

    def __init__(self, zip_path: str | Path) -> None:
        """打开解析结果压缩包。

        参数：
            zip_path: MinerU ``--format zip`` 生成的压缩包路径。
        """
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"MinerU 结果压缩包不存在：{self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as z:
            self.names: list[str] = z.namelist()
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ 底层读取
    def read(self, name: str) -> bytes:
        """读取压缩包内某个文件的原始字节。"""
        with zipfile.ZipFile(self.zip_path, "r") as z:
            return z.read(name)

    def read_text(self, name: str) -> str:
        """读取压缩包内某个文件的文本（UTF-8，自动去除 BOM）。"""
        return self.read(name).decode("utf-8-sig")

    def list_files(self) -> list[str]:
        """返回压缩包内全部条目路径。"""
        return list(self.names)

    def extract(self, dest_dir: str | Path) -> Path:
        """把压缩包安全解压到指定目录（含路径穿越防护）。

        参数：
            dest_dir: 解压目标目录。

        返回：
            解压目录的 Path。
        """
        dest = Path(dest_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        for name in self.names:
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise RuntimeError(f"压缩包内存在非法路径，拒绝解压：{name}")
        with zipfile.ZipFile(self.zip_path, "r") as z:
            z.extractall(dest)
        logger.info("已解压 %d 个条目到 %s", len(self.names), dest)
        return dest

    # ------------------------------------------------------------------ 条目定位
    def _find(self, keywords: tuple[str, ...], suffix: str) -> str | None:
        """按关键词 + 后缀定位条目（不区分大小写，只比对文件名部分）。

        参数：
            keywords: 文件名须同时包含的关键词（小写）。
            suffix:   文件名须以该后缀结尾（如 ".md"、".json"）。

        返回：
            匹配的条目路径；找不到返回 None。
        """
        for name in self.names:
            base = name.rsplit("/", 1)[-1].lower()
            if base.endswith(suffix) and all(k in base for k in keywords):
                return name
        return None

    def _find_markdown(self) -> str | None:
        return self._find(("markdown",), ".md") or self._find((), ".md")

    def _find_middle_json(self) -> str | None:
        # MinerU 不同版本命名略有差异：middle_json.json / content_list.json
        for keywords in (("middle",), ("content_list",), ("content",), ()):
            name = self._find(keywords, ".json")
            if name:
                return name
        return None

    def _find_structured(self) -> str | None:
        for keywords in (("structured",), ("content",), ()):
            name = self._find(keywords, ".json")
            if name:
                return name
        return None

    # ------------------------------------------------------------------ 结果访问
    @property
    def markdown(self) -> str:
        """Markdown 正文（字符串）。"""
        if "markdown" not in self._cache:
            name = self._find_markdown()
            self._cache["markdown"] = self.read_text(name) if name else ""
        return self._cache["markdown"]

    @property
    def middle_json(self) -> dict | list | None:
        """中间格式 JSON（解析后的 dict/list；缺失时返回 None）。"""
        if "middle_json" not in self._cache:
            name = self._find_middle_json()
            self._cache["middle_json"] = json.loads(self.read_text(name)) if name else None
        return self._cache["middle_json"]

    @property
    def structured_content(self) -> dict | list | None:
        """结构化内容 JSON（解析后的 dict/list；缺失时返回 None）。"""
        if "structured_content" not in self._cache:
            name = self._find_structured()
            self._cache["structured_content"] = json.loads(self.read_text(name)) if name else None
        return self._cache["structured_content"]

    @property
    def images(self) -> list[tuple[str, bytes]]:
        """提取的图片列表，元素为 (压缩包内路径, 图片字节)。"""
        if "images" not in self._cache:
            result: list[tuple[str, bytes]] = []
            for name in self.names:
                if name.endswith("/"):
                    continue
                base = name.rsplit("/", 1)[-1].lower()
                if base.endswith(_IMAGE_EXTS):
                    result.append((name, self.read(name)))
            self._cache["images"] = result
        return self._cache["images"]

    def __repr__(self) -> str:  # pragma: no cover - 调试打印
        return f"ParseResult({self.zip_path}, 条目数={len(self.names)})"


class MineruParser:
    """MinerU CLI 调用器（默认懒初始化，无重资源）。"""

    def __init__(
        self,
        exe: str | None = None,
        server_exe: str | None = None,
        tier: str = "flash",
        pages: str | None = None,
        fmt: str = "zip",
    ) -> None:
        """初始化。

        参数：
            exe:        mineru-kit 可执行文件路径；缺省用默认路径，找不到再查 PATH。
            server_exe: mineru server 可执行文件路径；缺省用默认路径，找不到回退 "mineru"。
            tier:       解析档位（flash / fast / high，取决于 MinerU 版本）。
            pages:      页码范围（如 "1-10"、"all"）；None 表示解析全部。
            fmt:        输出格式，默认 "zip"。
        """
        self.exe = exe
        self.server_exe = server_exe
        self.tier = tier
        self.pages = pages
        self.fmt = fmt

    def _resolve_exe(self) -> str:
        """解析并校验 mineru-kit 可执行文件路径。"""
        exe = self.exe or DEFAULT_EXE
        if Path(exe).exists():
            return str(Path(exe))
        found = shutil.which(exe) or shutil.which("mineru-kit")
        if found:
            return found
        raise FileNotFoundError(
            f"未找到 MinerU 可执行文件：{exe}。请确认 mineru 环境已安装，"
            f"或通过 exe 参数指定 mineru-kit 的完整路径。"
        )

    # ------------------------------------------------------------------ 解析
    def parse(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
        tier: str | None = None,
        pages: str | None = None,
        fmt: str | None = None,
        force: bool = False,
        timeout: int | None = None,
    ) -> ParseResult:
        """调用 MinerU 解析 PDF，返回结果对象。

        参数：
            pdf_path:   源 PDF 路径。
            output_dir: 输出目录；缺省为源文件同级目录下 ``<stem>_mineru``。
            tier:       解析档位；缺省取构造参数 tier。
            pages:      页码范围；缺省取构造参数 pages。
            fmt:        输出格式；缺省取构造参数 fmt（zip）。
            force:      为 False 且输出目录已存在同名结果时，直接复用、跳过解析。
            timeout:    子进程超时（秒）；None 表示不限制。

        返回：
            ParseResult 对象。
        """
        pdf = Path(pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"源 PDF 不存在：{pdf}")
        out = Path(output_dir) if output_dir else pdf.parent / f"{pdf.stem}_mineru"
        out.mkdir(parents=True, exist_ok=True)
        exe = self._resolve_exe()
        tier = tier or self.tier
        pages = pages if pages is not None else self.pages
        fmt = fmt or self.fmt

        # 已有结果且未要求强制重跑 → 直接复用
        existing = self._locate_zip(pdf, out)
        if existing is not None and not force:
            logger.info("检测到已有解析结果，跳过解析：%s", existing)
            return ParseResult(existing)

        cmd = [exe, "parse", str(pdf), "-o", str(out), "--tier", tier, "--format", fmt]
        if pages:
            cmd += ["-p", str(pages)]
        logger.info("开始 MinerU 解析：%s（tier=%s, format=%s, pages=%s）",
                    pdf.name, tier, fmt, pages or "all")
        logger.debug("执行命令：%s", " ".join(cmd))

        # 这里并不直接 import MinerU（它装在独立的 mineru 环境里），
        # 而是像在命令行敲命令一样，用 subprocess 调用 mineru-kit.exe 这个外部程序。
        # MinerU 进度写入 stderr，合并捕获；即使成功也可能返回非零退出码。
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        merged = proc.stdout or ""

        result_zip = self._locate_zip(pdf, out)
        if result_zip is None:
            logger.error("MinerU 解析失败，返回码=%s", proc.returncode)
            logger.error(merged[-3000:])
            raise RuntimeError(f"MinerU 解析失败：{merged[-2000:]}")

        if proc.returncode != 0:
            logger.warning("MinerU 返回非零退出码 %s，但已生成结果 zip，按成功处理", proc.returncode)
        logger.info("解析完成：%s", result_zip)
        return ParseResult(result_zip)

    @staticmethod
    def _locate_zip(pdf: Path, out: Path) -> Path | None:
        """在输出目录中定位与源 PDF 对应的结果压缩包。"""
        zips = list(out.rglob("*.zip"))
        if not zips:
            return None
        stem = pdf.stem.lower()
        for z in zips:
            if z.stem.lower() == stem:
                return z
        return max(zips, key=lambda p: p.stat().st_mtime)

    # ------------------------------------------------------------------ 服务
    def start_server(self, *args: str) -> subprocess.Popen:
        """启动 MinerU server（``mineru server start``），返回子进程句柄。"""
        exe = self.server_exe or DEFAULT_SERVER_EXE
        if not Path(exe).exists():
            exe = shutil.which(exe) or "mineru"
        cmd = [exe, "server", "start", *args]
        logger.info("启动 MinerU server：%s", " ".join(cmd))
        return subprocess.Popen(cmd)


# ---------------------------------------------------------------------- 便捷函数
def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    tier: str = "flash",
    pages: str | None = None,
    exe: str | None = None,
    fmt: str = "zip",
    force: bool = False,
    timeout: int | None = None,
) -> ParseResult:
    """便捷函数：调用 MinerU 解析 PDF（等价于 ``MineruParser(...).parse(...)``）。"""
    return MineruParser(exe=exe, tier=tier, pages=pages, fmt=fmt).parse(
        pdf_path, output_dir=output_dir, force=force, timeout=timeout
    )


def read_zip(zip_path: str | Path) -> ParseResult:
    """便捷函数：直接读取已有 MinerU 结果压缩包（不调用 MinerU）。"""
    return ParseResult(zip_path)


def start_server(*args: str) -> subprocess.Popen:
    """便捷函数：启动 MinerU server。"""
    return MineruParser().start_server(*args)


__all__ = ["ParseResult", "MineruParser", "parse_pdf", "read_zip", "start_server"]


if __name__ == "__main__":  # pragma: no cover - 无需 MinerU 即可跑通的自测
    import tempfile

    # 构造一个模拟 MinerU 输出的 zip，验证 ParseResult 读取逻辑
    demo_zip = Path(tempfile.gettempdir()) / "mineru_demo_result.zip"
    with zipfile.ZipFile(demo_zip, "w") as z:
        z.writestr("markdown.md", "# 标题\n\n这是正文内容。")
        z.writestr("middle_json.json",
                   json.dumps({"blocks": [{"type": "text", "text": "示例"}]}, ensure_ascii=False))
        z.writestr("structured_content.json",
                   json.dumps({"title": "示例", "sections": []}, ensure_ascii=False))
        z.writestr("images/img1.png", b"\x89PNG fake-bytes")

    result = read_zip(demo_zip)
    print("文件列表:", result.list_files())
    print("markdown:", repr(result.markdown))
    print("middle_json:", result.middle_json)
    print("structured_content:", result.structured_content)
    print("images:", [(name, len(data)) for name, data in result.images])
