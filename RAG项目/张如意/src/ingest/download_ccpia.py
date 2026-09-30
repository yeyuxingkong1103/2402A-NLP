# -*- coding: utf-8 -*-
"""从 ccpia.com.cn（中国农药工业协会）资讯页下载附件 PDF。

用法:
  python download_ccpia.py "<页面URL>" [--out DIR] [--all]

行为:
  抓取页面 -> 找 href 指向 .pdf/.doc/.docx/.xls/.xlsx 的附件链接 -> 用链接文字命名
  (sanitize 掉 Windows 非法字符) -> 下载并校验 %PDF- 头 + 页数 -> 追加 data/raw/_下载清单.csv

注意: 该站资讯页通常只挂「公告 PDF」，标准全文需邮件索取，页面上不会有链接。
"""
import argparse
import csv
import os
import re
import sys

import requests
from pypdf import PdfReader

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")   # 伪装成 Chrome，避免被站点按脚本拦截
EXT_RE = re.compile(r"\.(pdf|docx?|xlsx?)(?:$|[?#])", re.I)   # 匹配附件扩展名：.pdf/.doc/.docx/.xls/.xlsx（不区分大小写，可带 ?/# 查询串）


def safe(name, fallback):
    """清洗文件名：把 Windows 非法字符（\\/:*?"<>| 与控制符）替换为下划线，
    去掉首尾空格和点；清洗后为空则返回 fallback。"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", name).strip(" .")
    return name or fallback


def attachment_filename(url, label):
    """用链接文字做文件名；链接文字自带扩展名时不重复拼接。

    url：附件链接（用来取真实扩展名）；label：链接文字。
    返回「清洗后的文件名.扩展名」。
    """
    ext = EXT_RE.search(url).group(1).lower()
    stem = safe(label, os.path.basename(url)[:24])
    if stem.lower().endswith("." + ext):
        stem = stem[: -(len(ext) + 1)]
    return f"{stem}.{ext}"


def find_attachments(html, base):
    """返回 [(绝对URL, 链接文字)]，按链接文字去重。"""
    out, seen = [], set()
    for href, text in re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.S):
        if not EXT_RE.search(href):
            continue
        url = requests.compat.urljoin(base, href.replace("&amp;", "&"))
        if url in seen:
            continue
        seen.add(url)
        label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()
        out.append((url, label))
    return out


def main():
    """主流程：抓取资讯页 → 提取附件链接 → 逐个下载并校验（PDF 头 + 页数）
    → 把下载记录追加到 _下载清单_ccpia.csv。返回退出码：0 成功，1 无附件。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9", "Referer": args.url})
    r = s.get(args.url, timeout=60)
    r.encoding = r.apparent_encoding or "utf-8"
    title = re.search(r"<title>(.*?)</title>", r.text, re.S)
    print(f"页面: {r.status_code} {title.group(1).strip() if title else ''}", flush=True)

    items = find_attachments(r.text, args.url)
    if not items:
        print("未找到附件链接")
        return 1
    os.makedirs(args.out, exist_ok=True)
    # 与 GB 批次清单分开，避免覆盖 verify_pdfs.py 生成的 _下载清单.csv
    man = os.path.join(args.out, "_下载清单_ccpia.csv")
    rows = []
    if os.path.exists(man):
        with open(man, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    for url, label in items:
        ext = EXT_RE.search(url).group(1).lower()
        fn = os.path.join(args.out, attachment_filename(url, label))
        d = s.get(url, timeout=120)
        body = d.content
        # 校验文件头：PDF 必须以 %PDF- 开头，其余类型不校验；失败则不落盘
        ok = body[:5] == b"%PDF-" if ext == "pdf" else True
        pages, note = -1, ""
        if ok:
            with open(fn, "wb") as f:
                f.write(body)
            if ext == "pdf":
                try:
                    pages = len(PdfReader(fn).pages)
                except Exception as e:
                    note = f"解析失败 {type(e).__name__}"
        else:
            note = f"非 PDF 内容 ({body[:16]!r})"
        print(f"  [{'OK' if ok and pages != 0 else 'FAIL'}] {os.path.basename(fn)} "
              f"{len(body)//1024} KB {pages}页 {note}", flush=True)
        rows.append({"文件": os.path.basename(fn), "来源页面": args.url, "附件URL": url,
                     "大小KB": len(body) // 1024, "页数": pages, "校验": "OK" if ok else "FAIL"})

    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["文件", "来源页面", "附件URL", "大小KB", "页数", "校验"])
        w.writeheader()
        w.writerows(rows)
    print(f"清单: {man}（共 {len(rows)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
