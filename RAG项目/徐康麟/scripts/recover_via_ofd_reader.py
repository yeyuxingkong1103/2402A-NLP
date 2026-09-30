# -*- coding: utf-8 -*-
"""用官方阅读器的 OFD 接口**补回**那些解析不了的法规（跳过篇目恢复）。

背景（2026-09-16 实测）
----------------------
抓取时发现 55 篇法规下不动正文，两类原因：
  * ``pdf_no_text_layer``（28 篇）：回退到的 PDF 是**扫描件**，pypdf 抽 0 字符；
  * ``legacy_doc_no_pdf`` （27 篇）：服务端只给**老式 ``.doc``**（OLE2）且无 PDF 直链。

但这两类在 ``flfgDetails`` 里**都带 ``ossWordOfdPath``** —— 服务端用**原始 Word 文件**转好的
OFD（OFD 是国标版式文档，本质 ZIP+XML）。站点自己的阅读器就是读它，并且暴露了结构化文本接口：

    GET https://flkofd.npc.gov.cn/reader/info?file=<内网file参数>&_wr_*   -> 页数 count 等元数据
    GET https://flkofd.npc.gov.cn/reader/text?file=<同上>&_i=<页号>      -> 逐字符 JSON（带坐标）

于是**不需要 OCR、不需要 .doc 解析器、不需要装任何东西**：走 ``previewLink`` 拿签名，
再逐页取文本按坐标还原阅读顺序即可。

用法
----
    # 看有哪些待恢复
    python scripts/recover_via_ofd_reader.py --list

    # 先试一篇（不落盘，只打印前若干字）
    python scripts/recover_via_ofd_reader.py --limit 1 --dry-run

    # 试一篇并落盘
    python scripts/recover_via_ofd_reader.py --limit 1

    # 批量恢复全部（幂等：已生成 md 的会跳过）
    python scripts/recover_via_ofd_reader.py

    # 连那 2 篇网络失败的也一起试
    python scripts/recover_via_ofd_reader.py --include-fail

产物：``knowledge/lawyer/<分类>/<名称>.md``（与抓取脚本同一套元数据头），
并把 ``index/npc_crawl_state.json`` 里对应记录改成 ``ok``/``recovered_via_ofd_reader``。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_npc_law import (  # noqa: E402
    BASE, ROOT as CRAWL_ROOT, TEXT_DIR, NpcClient, RateLimiter,
    build_markdown, sanitize_filename,
)
from legal_rag.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger("scripts.recover_via_ofd_reader")

STATE_PATH = ROOT / "index" / "npc_crawl_state.json"
READER = "https://flkofd.npc.gov.cn"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

#: 需要恢复的跳过原因
RECOVER_REASONS = ("pdf_no_text_layer", "legacy_doc_no_pdf")


# --------------------------------------------------------------------------
# OFD 阅读器
# --------------------------------------------------------------------------

class OfdReaderClient:
    """官方 OFD 阅读器的文本接口（需要 previewLink 下发的 ``_wr_*`` 签名）。"""

    def __init__(self, *, timeout: float = 45.0, attempts: int = 3,
                 limiter: RateLimiter | None = None) -> None:
        self.timeout = timeout
        self.attempts = max(1, int(attempts))
        self.limiter = limiter or RateLimiter(0.0)
        # 复用抓取脚本的客户端（它带 Cookie 会话与 WAF 重定向处理）
        self.flk = NpcClient(timeout=timeout, attempts=attempts, limiter=self.limiter)

    # ---------- 拿签名 ----------
    def sign(self, ofd_path: str, *, bbbs: str = "") -> tuple[str, dict]:
        """``previewLink`` -> (内网 file 参数, {_wr_timestamp/_wr_app_id/_wr_sign})。"""
        data = self.flk.get_json("/law-search/amazonFile/previewLink", {"filePath": ofd_path},
                                 operation="previewLink",
                                 label=f"previewLink:{bbbs[:8] or ofd_path.split('/')[-1][:12]}")
        url = str((data.get("data") or {}).get("url") or "")
        if not url:
            raise RuntimeError(f"previewLink 未返回读者链接：{data}")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        raw_file = (query.get("file") or [""])[0]
        if not raw_file:
            raise RuntimeError(f"读者链接里没有 file 参数：{url}")
        wr = {key: (query.get(key) or [""])[0]
              for key in ("_wr_timestamp", "_wr_app_id", "_wr_sign")}
        return raw_file, wr

    # ---------- 调阅读器接口 ----------
    def _reader_get(self, endpoint: str, raw_file: str, wr: dict, *, extra: str = "",
                    label: str = "") -> bytes:
        # 阅读器要求 file 参数**二次编码**（实测：浏览器发的就是 %253A%252F...）
        encoded = urllib.parse.quote(urllib.parse.quote(raw_file, safe=""), safe="")
        url = (f"{READER}/reader/{endpoint}?file={encoded}"
               f"&_wr_timestamp={wr['_wr_timestamp']}&_wr_app_id={wr['_wr_app_id']}"
               f"&_wr_sign={wr['_wr_sign']}&_b=3.2.0&_v=1{extra}")
        headers = {"User-Agent": UA, "Accept": "*/*", "Referer": f"{READER}/reader"}
        last: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            self.limiter.wait()
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return resp.read()
            except Exception as exc:  # noqa: BLE001 - 逐页重试
                last = exc
                logger.warning("阅读器 %s 第 %d/%d 次失败（%s）：%s: %s",
                               endpoint, attempt, self.attempts, label,
                               type(exc).__name__, exc)
                if attempt < self.attempts:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"阅读器 {endpoint} 失败（{label}）：{type(last).__name__}: {last}")

    def page_count(self, raw_file: str, wr: dict, *, label: str = "") -> int:
        body = self._reader_get("info", raw_file, wr, label=label)
        try:
            info = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"/reader/info 不是 JSON（{label}）：{body[:200]!r}") from exc
        count = int(info.get("count") or 0)
        logger.info("阅读器元数据 %s：页数=%d 标题=%r 生成器=%r isOcr=%s",
                    label, count, info.get("Title"), info.get("Creator"), info.get("isOcr"))
        return count

    def page_text(self, raw_file: str, wr: dict, index: int, *, label: str = "") -> str:
        body = self._reader_get("text", raw_file, wr, extra=f"&_i={index}",
                                label=f"{label}#p{index}")
        try:
            page = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"/reader/text 第 {index} 页不是 JSON（{label}）") from exc
        return _flatten_page(page)

    def full_text(self, raw_file: str, wr: dict, *, label: str = "") -> tuple[str, int]:
        count = self.page_count(raw_file, wr, label=label)
        if count <= 0:
            raise RuntimeError(f"阅读器报告页数为 0（{label}）")
        pages = []
        for index in range(count):
            pages.append(self.page_text(raw_file, wr, index, label=label))
        return "\n".join(p for p in pages if p.strip()), count


def _flatten_page(page: dict) -> str:
    """逐字符 JSON -> 文本。按坐标 (上, 左) 排序，还原阅读顺序。"""
    def key(boundary):
        try:
            return (round(float(boundary[1]), 2), round(float(boundary[0]), 2))
        except Exception:  # noqa: BLE001 - 坐标缺失时保持原顺序
            return (0.0, 0.0)

    lines: list[str] = []
    areas = page.get("areas") or []
    for area in sorted(areas, key=lambda a: key(a.get("boundary") or (0, 0, 0, 0))):
        for line in sorted(area.get("lines") or [],
                           key=lambda l: key(l.get("boundary") or (0, 0, 0, 0))):
            text = "".join((ch.get("char") or "") for ch in (line.get("chars") or []))
            if text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 恢复流程
# --------------------------------------------------------------------------

class Recoverer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        self.client = OfdReaderClient(timeout=args.timeout, attempts=args.attempts,
                                      limiter=RateLimiter(args.min_interval))

    def targets(self) -> list[dict]:
        docs = self.state.get("docs", {})
        picked = [r for r in docs.values()
                  if r.get("status") == "skip" and r.get("reason") in RECOVER_REASONS]
        if self.args.include_fail:
            picked += [r for r in docs.values() if r.get("status") == "fail"]
        if self.args.only:
            picked = [r for r in picked if r.get("category") in self.args.only]
        return sorted(picked, key=lambda r: (r.get("category") or "", r.get("title") or ""))

    def ofd_path(self, record: dict) -> str:
        """取该篇的 Word 版 OFD 路径（缺失时退回 PDF 版 OFD）。"""
        data = self.client.flk.get_json("/law-search/search/flfgDetails",
                                        {"bbbs": record.get("bbbs")},
                                        operation="detail",
                                        label=f"detail:{(record.get('bbbs') or '')[:8]}")
        if not isinstance(data, dict) or data.get("code") != 200:
            raise RuntimeError(f"详情接口异常：{data}")
        oss = ((data.get("data") or {}).get("ossFile")) or {}
        for key in ("ossWordOfdPath", "ossPdfOfdPath"):     # Word 版优先（文本可用）
            if oss.get(key):
                return str(oss[key])
        raise RuntimeError(f"详情里没有 OFD 路径：{list(oss.keys())}")

    def name_of(self, record: dict) -> str:
        return str(record.get("name") or sanitize_filename(record.get("title") or "unnamed"))

    def recover_one(self, record: dict) -> dict:
        title = record.get("title") or ""
        category = record.get("category") or "未分类"
        label = f"{category}/{title[:28]}"
        logger.info("入口 recover_one(%s)", label)
        started = time.perf_counter()
        try:
            text_path = TEXT_DIR / category / f"{self.name_of(record)}.md"
            if text_path.is_file() and not self.args.force:
                logger.info("已存在正文，跳过：%s", text_path.name)
                record.update({"status": "ok", "reason": "recovered_via_ofd_reader",
                               "text": str(text_path.relative_to(ROOT)).replace("\\", "/")})
                return record

            ofd = self.ofd_path(record)
            raw_file, wr = self.client.sign(ofd, bbbs=record.get("bbbs") or "")
            body, pages = self.client.full_text(raw_file, wr, label=label)
            if len(body.strip()) < 20:
                raise RuntimeError(f"OFD 文本过短（{len(body.strip())} 字符），疑似无文本层")

            markdown = build_markdown(record, body, category=category,
                                      source_url=f"{BASE}/detail?bbbs={record.get('bbbs')}",
                                      fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            if self.args.dry_run:
                logger.info("[dry-run] 不落盘：%s -> %d 字符 / %d 页；开头：%r",
                            label, len(body), pages, body[:80])
            else:
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(markdown, encoding="utf-8")
                record.update({
                    "status": "ok", "reason": "recovered_via_ofd_reader",
                    "chars": len(body), "pages": pages, "ofd": ofd,
                    "text": str(text_path.relative_to(ROOT)).replace("\\", "/"),
                    "recovered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                logger.info("恢复成功：%s -> %d 字符 / %d 页", label, len(body), pages)
        except Exception as exc:  # noqa: BLE001 - 单篇失败不影响整批
            logger.exception("恢复失败：%s", label)
            record["recover_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["recover_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            logger.info("出口 recover_one(%s) -> %s（%.0fms）", label,
                        record.get("status"), record["recover_ms"])
        return record

    def save(self) -> None:
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)

    def run(self) -> int:
        targets = self.targets()
        logger.info("待恢复 %d 篇（原因 %s，include_fail=%s）",
                    len(targets), "/".join(RECOVER_REASONS), self.args.include_fail)
        if self.args.list or not targets:
            for r in targets[:200] if self.args.list else []:
                print(f"  [{r.get('category')}] {r.get('title')}  ({r.get('reason')})")
            return 0
        if self.args.limit:
            targets = targets[: self.args.limit]
        ok = fail = 0
        for index, record in enumerate(targets, 1):
            print(f"[{index}/{len(targets)}] {record.get('category')} :: "
                  f"{(record.get('title') or '')[:40]}", flush=True)
            self.recover_one(record)
            # dry-run 不写 reason，按"没有 recover_error"判成功，免得汇总数字自相矛盾
            recovered = (not record.get("recover_error")) if self.args.dry_run \
                else record.get("reason") == "recovered_via_ofd_reader"
            if recovered:
                ok += 1
            else:
                fail += 1
            if not self.args.dry_run and index % 5 == 0:
                self.save()
        if not self.args.dry_run:
            self.save()
        prefix = "[dry-run] " if self.args.dry_run else ""
        print(f"\n{prefix}恢复结束：成功 {ok} / 未恢复 {fail}")
        logger.info("恢复结束：成功 %d / 未恢复 %d", ok, fail)
        return 0 if fail == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用官方 OFD 阅读器接口恢复解析不了的法规")
    parser.add_argument("--list", action="store_true", help="只列出待恢复篇目")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少篇（0=全部，试跑用 1）")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="只取文本并打印，不落盘、不改状态")
    parser.add_argument("--force", action="store_true", help="即使 md 已存在也重做")
    parser.add_argument("--include-fail", action="store_true", dest="include_fail",
                        help="连 fail 状态的篇目一起试")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些分类")
    parser.add_argument("--min-interval", type=float, default=0.5, dest="min_interval",
                        help="全局最小请求间隔秒数（默认 0.5）")
    parser.add_argument("--timeout", type=float, default=45.0, help="单次请求超时秒数")
    parser.add_argument("--attempts", type=int, default=3, help="单次请求最大尝试次数")
    return parser


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    if not STATE_PATH.is_file():
        print(f"找不到抓取状态文件：{STATE_PATH}")
        return 1
    print(f"状态文件 : {STATE_PATH}")
    print(f"正文目录 : {TEXT_DIR}")
    print(f"阅读器   : {READER}")
    print("-" * 68)
    return Recoverer(args).run()


if __name__ == "__main__":
    sys.exit(main())
