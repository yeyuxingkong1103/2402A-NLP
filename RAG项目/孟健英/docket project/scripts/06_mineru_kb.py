#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06_mineru_kb.py — MinerU 解析无文本层的扫描版 PDF 并入库知识库。

用法:
    python3 scripts/06_mineru_kb.py <扫描版.pdf> [--keep]

前提（一次性）:
    pip install --user 'mineru[core]'
    mineru config set parse_server.local.mode managed
    mineru config set parse_server.local.managed_tier basic
    mineru-kit models download --tier basic   # 引擎模型（~数百 MB）

流程: 确保 mineru server 在跑 → mineru parse 全页解析出 markdown
      → 按页重组为 [第N页] 标记文本 → 复用 split_pages/chunk_pages
      切块 → embed_and_insert 入 Milvus。
"""
import argparse  # 命令行参数解析：位置参数 pdf + 开关 --keep
import logging  # 日志：记录解析页数与入库条数
import re  # 正则：从 markdown 里识别分页标记
import shutil  # which() 定位可执行文件、copy2() 复制解析产物
import subprocess  # 起子进程调 mineru CLI：查/起 server、跑 parse
import sys  # 改 sys.path 让脚本直跑也能 import src；sys.exit 带消息退出
import tempfile  # 临时目录：解析产物默认用完即删
import time  # sleep 等待 server 完成启动
from pathlib import Path  # 路径处理

_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录：本文件在 scripts/ 下，上级即根
sys.path.insert(0, str(_ROOT))  # 项目根插到模块搜索路径最前：脚本直跑时也能 import src.*
from src.config import setup_logging  # noqa: E402
from src.ingestion import chunk_pages, embed_and_insert, split_pages  # noqa: E402

setup_logging()  # 初始化日志
logger = logging.getLogger("06_mineru")  # 用固定名字而非 __name__：日志里一眼认出是这个建库脚本

PAGE_MARK = re.compile(  # 预编译分页标记正则：同时兼容两种格式
    r"(?:\[第\s*(\d+)\s*页\]|<!--\s*page\s+(\d+)\s+of\s+\d+\s*-->)")  # mineru 4.x 输出 HTML 注释分页


def _bin(name: str) -> str:  # 定位 mineru 可执行文件的完整路径
    return shutil.which(name) or str(Path.home() / ".local" / "bin" / name)  # PATH 里找不到就回退到 pip --user 的默认安装位置 ~/.local/bin


def ensure_server() -> None:  # 确保 mineru 本地解析服务在运行
    """mineru parse 依赖本地 server，不在则拉起。"""
    st = subprocess.run([_bin("mineru"), "server", "status"],  # 查询 server 状态
                        capture_output=True, text=True)  # 捕获输出并按文本返回：不弹窗、不刷屏
    if "PID" in st.stdout:  # 输出里含 PID 说明服务已在跑
        return  # 已在运行，直接返回
    logger.info("mineru server 未运行，启动中…")
    subprocess.run([_bin("mineru"), "server", "start"],  # 后台拉起 server
                   capture_output=True, text=True)
    time.sleep(6)  # 等 6 秒让服务完成初始化：立刻发请求会被拒绝连接


def run_mineru(pdf: Path, out_md: Path) -> Path:  # 调 mineru 把扫描版 PDF 解析成 markdown
    """全页解析 PDF → markdown 文件路径。"""
    out_md.parent.mkdir(parents=True, exist_ok=True)  # 先建好输出目录：mineru 不会自动建多级目录
    r = subprocess.run(
        [_bin("mineru"), "parse", str(pdf), "--pages", "all",  # parse 子命令：--pages all 全页解析
         "--format", "markdown", "--force", "--wait", "7200",  # 输出 markdown；--force 覆盖旧产物；--wait 最多等 7200 秒（扫描版 OCR 很慢）
         "-o", str(out_md)],  # 指定输出文件路径
        capture_output=True, text=True)
    if r.returncode != 0:  # 非零退出码即解析失败
        raise RuntimeError((r.stderr or r.stdout).strip()[-600:])  # 只取末尾 600 字符报错：mineru 输出很长，全抛会刷屏
    if not out_md.exists():  # 退出码正常但没产出文件，同样算失败
        raise FileNotFoundError(f"解析未产出 {out_md}")
    return out_md


def pages_from_markdown(md_path: Path) -> list[tuple[int, str]]:  # 把 mineru 输出的 markdown 按分页标记重组
    """把带分页标记的 markdown 重组为 [(页码, 文本)]；无标记回退整篇 1 页。"""
    text = md_path.read_text(encoding="utf-8")  # 读入 markdown 全文
    pages: list[list] = []  # [页码, [行]]
    for line in text.splitlines():  # 逐行扫描
        m = PAGE_MARK.fullmatch(line.strip())  # 整行匹配分页标记：fullmatch 防止正文里局部提到标记被误判
        if m:  # 命中分页标记
            pages.append([int(m.group(1) or m.group(2)), []])  # 新开一页：两种标记各对应一个捕获组，哪个命中取哪个
            continue  # 标记行本身不进正文
        if line.strip():  # 跳过空行
            if not pages:  # 第一个标记之前就有内容（封面/前言），归入第 1 页
                pages.append([1, []])
            pages[-1][1].append(line.rstrip())  # 行归入当前页，去行尾空白
    return [(p, "\n".join(ls).strip()) for p, ls in pages if ls]  # 丢弃空页，把行列表拼成页文本返回


def main() -> None:  # 主流程：确保服务 → 解析 → 重组 → 切块 → 入库
    ap = argparse.ArgumentParser(description="MinerU 扫描版 PDF 入知识库")
    ap.add_argument("pdf", help="扫描版 PDF 路径")  # 位置参数：待入库的 PDF
    ap.add_argument("--keep", action="store_true", help="保留解析产物到 data/mineru_out/")  # 开关：默认产物随临时目录一起删除
    args = ap.parse_args()  # 解析命令行

    pdf = Path(args.pdf)
    if not pdf.exists():  # 文件不存在直接退出
        sys.exit(f"文件不存在：{pdf}")  # sys.exit 带消息：打印到 stderr 且退出码非 0

    with tempfile.TemporaryDirectory(prefix="mineru_kb_") as tmp:  # 临时目录存解析产物，with 结束自动清理
        out_md = Path(tmp) / "out" / "result.md"  # mineru 输出的 markdown 路径
        ensure_server()  # 先确保本地解析服务在跑
        run_mineru(pdf, out_md)  # 扫描版 PDF 没有文本层，PyMuPDF 提不出字，必须走 mineru 的 OCR+版面分析
        pages = pages_from_markdown(out_md)  # markdown 按分页标记重组为 [(页码, 文本)]
        total = sum(len(t) for _, t in pages)  # 总字数：评估解析质量的直观指标
        logger.info("解析出 %d 页 / %d 字", len(pages), total)
        if not pages:  # 一页都没解析出来，说明彻底失败
            sys.exit("未解析出任何文本，请确认 PDF 是否有效")
        full = "\n".join(f"[第{p}页]\n{t}" for p, t in pages)  # 拼成与 PyMuPDF 路线一致的 [第N页] 标记格式：下游切块代码完全复用
        chunks = chunk_pages(split_pages(full), source=pdf.name)  # 切页再切块：与文本版 PDF 共用同一条入库管线，保证库内块格式统一
        n = embed_and_insert(chunks)  # 用同一个 BGE-m3 模型向量化并写入 Milvus：入库与检索模型一致，向量空间才对得上
        logger.info("✅ 入库 %s 条（来源 %s，%d 页）", n, pdf.name, len(pages))
        if args.keep:  # 需要留存产物排查问题时加 --keep
            dst = _ROOT / "data" / "mineru_out" / pdf.stem  # 产物保留目录：按 PDF 名分文件夹
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_md, dst / "result.md")  # copy2 连文件元数据（时间戳）一起复制
            logger.info("解析产物保留在 %s", dst)


if __name__ == "__main__":  # 脚本入口
    main()  # 扫描版 PDF 的离线入库：与 03/04 殊途同归，最终都进同一个 Milvus 集合
