# -*- coding: utf-8 -*-
"""修复「同分类下重名」导致的数据丢失：把撞车的法规逐篇重新取回、按唯一名落盘。

来龙去脉（2026-09-16 实测）
--------------------------
抓取脚本早期的去重后缀用的是 ``bbbs[:8]``。这批 id 是「时间戳前缀 + 随机尾」结构，
**前 8 位大量相同**（``ff808081`` / ``ff808181`` 反复出现），于是"冲突后缀"自己又冲突了：

    出版管理条例（2020-11-29 版, bbbs=ff808081777d07c5...）-> 出版管理条例__ff808081.md
    出版管理条例（2016-02-06 版, bbbs=ff8080816f3e9784...）-> 出版管理条例__ff808081.md  ← 同名！

结果：9 组（18 篇）行政法规**每对只剩 1 个 md 与 1 个 docx**，另一篇正文丢失，
且两条 state 记录指向同一个文件（所以"文件缺失数=0"查不出来，要靠**重名检测**才发现）。

本脚本做的事
------------
1. 找出同 ``(category, name)`` 下有多条 ``ok`` 记录的分组；
2. 组内每篇都**重新取回正文**（docx -> pdf -> OFD 阅读器，逐级回退），
   用 ``<标题>__<完整 bbbs>`` 作唯一文件名（确定性 + 必然唯一）；
3. 两篇都写成功后才删除旧的撞车文件；
4. 更新 state 里的 ``name/text/raw``，并可 ``--verify`` 复核（重名 0 / 缺失 0）。

用法
----
    python scripts/fix_name_collisions.py --list        # 只看有哪些组
    python scripts/fix_name_collisions.py --dry-run     # 试第一组，不落盘
    python scripts/fix_name_collisions.py               # 全部修复
    python scripts/fix_name_collisions.py --verify      # 复核（修复后跑）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_npc_law import (  # noqa: E402
    BASE, RAW_DIR, TEXT_DIR, NpcClient, RateLimiter, build_markdown,
    docx_to_text, pdf_to_text, sanitize_filename, sniff_container,
)
from legal_rag.logging_setup import get_logger, setup_logging  # noqa: E402
from recover_via_ofd_reader import OfdReaderClient  # noqa: E402

logger = get_logger("scripts.fix_name_collisions")

STATE_PATH = ROOT / "index" / "npc_crawl_state.json"


def unique_name(title: str, bbbs: str) -> str:
    """唯一文件名：完整 bbbs 作后缀（确定性且必然唯一）。"""
    return f"{sanitize_filename(title)}__{bbbs}"


class CollisionFixer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        self.docs = self.state["docs"]
        limiter = RateLimiter(args.min_interval)
        self.client = NpcClient(timeout=args.timeout, attempts=args.attempts, limiter=limiter)
        self.ofd = OfdReaderClient(timeout=args.timeout, attempts=args.attempts,
                                   limiter=limiter)

    # ---------- 找组 ----------
    def groups(self) -> dict[tuple, list[dict]]:
        bucket: dict[tuple, list[dict]] = defaultdict(list)
        for record in self.docs.values():
            if record.get("status") == "ok":
                bucket[(record.get("category"), record.get("name"))].append(record)
        return {k: v for k, v in bucket.items() if len(v) > 1}

    # ---------- 取正文（docx -> pdf -> OFD 逐级回退） ----------
    def fetch_text(self, record: dict, target_name: str) -> tuple[str, str, str]:
        """返回 (正文, 落盘的原始件相对路径, 来源说明)。"""
        bbbs = str(record.get("bbbs") or "")
        title = record.get("title") or target_name
        category = record.get("category") or "未分类"
        blob = b""
        url = self.client.get_json("/law-search/download/pc",
                                   {"format": "docx", "bbbs": bbbs, "fileId": ""},
                                   operation="download-pc:docx",
                                   label=f"download-pc:docx:{bbbs[:8]}").get("data") or {}
        if url.get("url"):
            blob = self.client.request(url["url"], operation="file:docx",
                                       label=f"file:docx:{title[:20]}")
        kind = sniff_container(blob)

        if kind == "docx":
            raw_path = RAW_DIR / category / f"{target_name}.docx"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(blob)
            return docx_to_text(raw_path), str(raw_path.relative_to(ROOT)).replace("\\", "/"), "docx"

        # pdf 回退
        pdf = self.client.get_json("/law-search/download/pc",
                                   {"format": "pdf", "bbbs": bbbs, "fileId": ""},
                                   operation="download-pc:pdf",
                                   label=f"download-pc:pdf:{bbbs[:8]}").get("data") or {}
        if pdf.get("url"):
            pblob = self.client.request(pdf["url"], operation="file:pdf",
                                        label=f"file:pdf:{title[:20]}")
            if sniff_container(pblob) == "pdf":
                raw_path = RAW_DIR / category / f"{target_name}.pdf"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(pblob)
                body = pdf_to_text(raw_path)
                if len(body.strip()) >= 20:
                    return body, str(raw_path.relative_to(ROOT)).replace("\\", "/"), "pdf"

        # 最后走官方 OFD 阅读器（文本可靠，且不依赖下载格式）
        ofd_path = self.ofd.ofd_path(record)
        raw_file, wr = self.ofd.sign(ofd_path, bbbs=bbbs)
        body, pages = self.ofd.full_text(raw_file, wr, label=f"{category}/{title[:20]}")
        logger.info("  经 OFD 阅读器取回：%d 字符 / %d 页", len(body), pages)
        if blob and kind == "ole2":
            raw_path = RAW_DIR / category / f"{target_name}.doc"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(blob)
            return body, str(raw_path.relative_to(ROOT)).replace("\\", "/"), "ofd(+doc原件)"
        return body, "", "ofd"

    # ---------- 修一组 ----------
    def fix_group(self, key: tuple, records: list[dict]) -> tuple[int, int]:
        category, old_name = key
        logger.info("修复重名组 [%s] %s（%d 篇）", category, old_name, len(records))
        written: list[tuple[dict, Path]] = []
        ok = fail = 0
        for record in records:
            bbbs = str(record.get("bbbs") or "")
            title = record.get("title") or ""
            target = unique_name(title, bbbs)
            try:
                old_chars = record.get("chars") or 0
                body, raw_rel, source = self.fetch_text(record, target)
                if len(body.strip()) < 20:
                    raise RuntimeError(f"正文过短（{len(body.strip())} 字符）")
                # 版本一致性自检：取回的字数应与当初记录的字数接近，
                # 差太多说明可能取成了**同名另一个版本**，必须留痕告警（不静默）。
                if old_chars and abs(len(body) - old_chars) / old_chars > 0.15:
                    logger.warning("  字数偏差较大：本次 %d vs 记录 %d（%s）—— 请抽查是否取错版本",
                                   len(body), old_chars, title)
                markdown = build_markdown(
                    record, body, category=category,
                    source_url=f"{BASE}/detail?bbbs={bbbs}",
                    fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                text_path = TEXT_DIR / category / f"{target}.md"
                if self.args.dry_run:
                    logger.info("  [dry-run] %s -> %d 字符（来源 %s，原记录 %d）",
                                target, len(body), source, old_chars)
                else:
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_text(markdown, encoding="utf-8")
                    record.update({
                        "name": target, "chars": len(body), "content_source": source,
                        "chars_before_fix": old_chars,
                        "text": str(text_path.relative_to(ROOT)).replace("\\", "/"),
                        "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    if raw_rel:
                        record["raw"] = raw_rel
                    written.append((record, text_path))
                    logger.info("  已写：%s（%d 字符，来源 %s，原记录 %d 字符）",
                                target, len(body), source, old_chars)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - 单篇失败不阻断
                logger.exception("  修复失败：%s", target)
                record["fix_error"] = f"{type(exc).__name__}: {exc}"
                fail += 1

        # 全部成功才清理旧的撞车文件（失败就保留现场，别把唯一副本删掉）
        if not self.args.dry_run and fail == 0:
            for suffix in (".md",):
                old = TEXT_DIR / category / f"{old_name}{suffix}"
                if old.is_file():
                    old.unlink()
                    logger.info("  已删除旧撞车文件：%s", old.name)
            for suffix in (".docx", ".doc", ".pdf"):
                old = RAW_DIR / category / f"{old_name}{suffix}"
                if old.is_file():
                    old.unlink()
                    logger.info("  已删除旧撞车原始件：%s", old.name)
        return ok, fail

    def save(self) -> None:
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)

    # ---------- 复核 ----------
    def verify(self) -> int:
        ok_records = [r for r in self.docs.values() if r.get("status") == "ok"]
        bucket: dict[tuple, list[str]] = defaultdict(list)
        for r in ok_records:
            bucket[(r.get("category"), r.get("name"))].append(r.get("bbbs") or "")
        dup = {k: v for k, v in bucket.items() if len(v) > 1}
        missing = [r for r in ok_records
                   if not r.get("text") or not (ROOT / r["text"]).is_file()]
        print(f"ok 篇目        : {len(ok_records)}")
        print(f"重名组         : {len(dup)}（应为 0）")
        for k, v in list(dup.items())[:10]:
            print(f"   重名: {k} -> {v}")
        print(f"正文文件缺失   : {len(missing)}（应为 0）")
        for r in missing[:10]:
            print(f"   缺失: {r.get('category')} / {r.get('title')} -> {r.get('text')}")
        total = len(list((TEXT_DIR).rglob("*.md")))
        print(f"磁盘 md 总数   : {total}")
        return 0 if (not dup and not missing) else 1

    def run(self) -> int:
        groups = self.groups()
        print(f"重名组：{len(groups)} 组 / {sum(len(v) for v in groups.values())} 篇")
        if self.args.list:
            for (category, name), items in sorted(groups.items()):
                print(f"  [{category}] {name}  ({len(items)} 篇)")
            return 0
        if not groups:
            print("没有需要修复的重名组。")
            return self.verify() if self.args.verify else 0
        if self.args.limit:
            groups = dict(list(sorted(groups.items()))[: self.args.limit])
        total_ok = total_fail = 0
        for key, records in sorted(groups.items()):
            ok, fail = self.fix_group(key, records)
            total_ok += ok
            total_fail += fail
            if not self.args.dry_run:
                self.save()
        print(f"\n修复结束：成功 {total_ok} / 失败 {total_fail}")
        return 0 if total_fail == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="修复同分类重名导致的语料覆盖问题")
    parser.add_argument("--list", action="store_true", help="只列出重名组")
    parser.add_argument("--verify", action="store_true", help="复核重名/缺失（不修）")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="只取回并打印，不落盘")
    parser.add_argument("--limit", type=int, default=0, help="最多修多少组（0=全部）")
    parser.add_argument("--min-interval", type=float, default=0.5, dest="min_interval")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--attempts", type=int, default=3)
    return parser


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    fixer = CollisionFixer(args)
    if args.verify and not args.list:
        return fixer.verify()
    return fixer.run()


if __name__ == "__main__":
    sys.exit(main())
