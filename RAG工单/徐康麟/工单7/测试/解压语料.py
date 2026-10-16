# -*- coding: utf-8 -*-
"""工单7：解压 CCF 竞赛语料（``ccf_competition.zip`` → ``研发/data/raw``）。

工单编号：人工智能NLP-RAG-功能测试及评估

背景（实测坑，务必保留此逻辑）：
    ``ccf_competition.zip`` 内的中文文件名是 **GBK 字节被 zipfile 当 cp437 解码** 的乱码形式，
    例如 ``2020-02-14__╞╜░▓╥°╨╨╣╔╖▌╙╨╧▐╣½╦╛__000001__...pdf``。
    必须 ``info.filename.encode("cp437").decode("gbk")`` 才能还原成
    ``2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf``。
    若用 ``ZipFile.extractall()`` 直接解，落盘文件名会是乱码，且 PDF 内部 /Title 与实际不符，
    下游「按公司名找证据」的分析代码会全部失效。

用法：
    python 测试/解压语料.py                          # 默认输入/ccf_competition.zip → 研发/data/raw
    python 测试/解压语料.py --zip 输入/ccf_competition.zip --out 研发/data/raw
    python 测试/解压语料.py --list                   # 只列清单，不解压

退出码：0 成功（含「已存在且哈希一致，跳过」）；1 输入文件缺失；2 参数或解压异常。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = REPO_ROOT / "输入" / "ccf_competition.zip"
DEFAULT_OUT = REPO_ROOT / "研发" / "data" / "raw"


def fix_name(raw: str) -> str:
    """把 zipfile 解出的乱码文件名还原为真实中文名（cp437 → gbk）；失败时原样返回。"""
    try:
        return raw.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    """算文件 sha256（用于「已解压且一致则跳过」的幂等判断）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"工单7：{WORK_ORDER} —— CCF 语料解压")
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help=f"压缩包路径（默认 {DEFAULT_ZIP}）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"PDF 落盘目录（默认 {DEFAULT_OUT}）")
    parser.add_argument("--list", action="store_true", help="只打印压缩包清单，不解压")
    parser.add_argument("--force", action="store_true", help="已存在也重新解压")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    print("=" * 78)
    print(f"工单7 语料解压｜{WORK_ORDER}")
    print(f"压缩包：{zip_path}")
    print(f"落盘  ：{out_dir}")
    print("=" * 78)

    if not zip_path.is_file():
        print(f"❌ 找不到压缩包：{zip_path}", file=sys.stderr)
        return 1

    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"❌ 压缩包无法打开：{exc}", file=sys.stderr)
        return 2

    members: list[tuple[zipfile.ZipInfo, str]] = []
    with archive:
        for info in archive.infolist():
            name = fix_name(info.filename)
            if not name.lower().endswith(".pdf"):
                continue
            members.append((info, name))
        members.sort(key=lambda pair: pair[1])

        if args.list:
            print(f"共 {len(members)} 份 PDF：")
            for info, name in members:
                print(f"  {info.file_size:>10d} B  {Path(name).name}")
            return 0

        if not members:
            print("❌ 压缩包里没有任何 PDF", file=sys.stderr)
            return 1

        out_dir.mkdir(parents=True, exist_ok=True)
        extracted = skipped = 0
        for info, name in members:
            target = out_dir / Path(name).name
            payload = archive.read(info)
            want = hashlib.sha256(payload).hexdigest()
            if target.is_file() and not args.force and sha256_of(target) == want:
                print(f"  = 跳过（已存在且一致）：{target.name}")
                skipped += 1
                continue
            target.write_bytes(payload)
            print(f"  ✅ 解压 {target.name}（{info.file_size} B）")
            extracted += 1

    total_pages = sum(1 for _ in out_dir.glob("*.pdf"))
    print("-" * 78)
    print(f"完成：新解压 {extracted} 份，跳过 {skipped} 份；{out_dir} 下现有 {total_pages} 份 PDF")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已中断]", file=sys.stderr)
        raise SystemExit(130)
