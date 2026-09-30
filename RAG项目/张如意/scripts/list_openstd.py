# -*- coding: utf-8 -*-
"""List all rows of an ICS-filtered std_list_type page (all p.p1 types, paginated).

Usage:
  python list_openstd.py [ics] [--p1 1,2,3] [--search 关键词]

Prints `[采]` for 采标 rows (metadata-only, NOT downloadable) so you can count the
downloadable set BEFORE building a batch. Ends by printing the per-type totals.
Note: the site caps the list at one page for most ICS codes; this still pages through
until a short page arrives.

【模块说明（中文补充）】
  整体作用：抓取 openstd.samr.gov.cn（国家标准全文公开系统）按 ICS 分类过滤后的
  标准列表页，解析出每行（序号、标准号、状态标签、标准名称、id），并标记「采标」行。
  用途：在批量下载国标 PDF 之前，先清点该 ICS 分类下**可下载**（非采标）标准的数量。

  输入：命令行参数（无本地文件输入）——
    - ics      位置参数，ICS 国际分类号，默认 "65"（农业）；"13" 为环保/保健/安全
    - --p1     标准类型列表，逗号分隔：1=GB（强制），2=GB/T（推荐），3=GB/Z（指导），默认 "1,2,3"
    - --search 客户端关键字过滤（对标准名称做子串匹配）
  输出：
    - stdout：逐行打印标准清单（采标标记 / 类型 / 名称 / 标准号 id）
    - stderr：每个 p.p1 类型的统计汇总（总数、采标数、可下载数）
  ⚠️ 阅读提示：本文件的 HTML 解析正则依赖目标网站的页面结构；部分全局量
    （UA/LIST_ROW_RE/URL）若与实际运行环境不符，运行会抛 NameError，以实际代码为准。
"""
import argparse
import re
import sys

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")   # 伪装成 Chrome 浏览器的 User-Agent，避免被站点拦截

# 解析列表页 <tr> 行的正则，按顺序捕获 6 组：
#   1=序号  2=第 1 个 onclick 里的 32 位十六进制 id  3=第 1 个链接文本(标准号)
#   4=状态单元格 HTML(内含采标与否的 label-warning 标记)
#   5=第 2 个 onclick 里的 id  6=第 2 个链接文本(标准名称)
# re.S 让 "." 能匹配换行，兼容行内换行的 HTML
LIST_ROW_RE = re.compile(
    r"<tr>\s*<td>(\d+)</td>\s*<td[^>]*><a[^>]*onclick=\"showInfo\('([0-9A-F]{32})'\);\">([^<]+)</a></td>\s*<td>(.*?)</td>\s*<td class=\"mytxt\"[^>]*>.*?onclick=\"showInfo\('([0-9A-F]{32})'\);\">([^<]+)</a>",
    re.S)

# 列表页 URL 模板：{p1}=标准类型  {ics}=ICS 分类号  {page}=页码；pageSize 固定 50 条/页
URL = ("https://openstd.samr.gov.cn/bzgk/std/std_list_type"
       "?p.p1={p1}&p.p6={ics}&p.p90=circulation_date&p.p91=desc&pageSize=50&page={page}")


def main():
    """抓取并解析列表页，打印清单与统计。

    返回值：无显式返回（None）；结果全部打印到 stdout/stderr。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("ics", nargs="?", default="65", help="ICS code, 65=农业, 13=环保保健安全")
    ap.add_argument("--p1", default="1,2,3", help="comma list of p.p1 types: 1=GB 2=GB/T 3=GB/Z")
    ap.add_argument("--search", default="", help="在结果中筛选 keyword (client-side filter on name)")
    args = ap.parse_args()

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})  # 带浏览器 UA 与中文语言偏好

    for p1 in args.p1.split(","):                # 依次处理每种标准类型（GB / GB/T / GB/Z）
        count = cai = 0                          # count=该类型总行数  cai=采标（不可下载）行数
        for page in range(1, 6):                 # 最多翻 5 页；绝大多数 ICS 码只有一两页
            html = s.get(URL.format(p1=p1, ics=args.ics, page=page), timeout=60).text
            rows = LIST_ROW_RE.findall(html)     # 返回 [(序号,id,标准号,状态HTML,id,名称), ...]
            for row in rows:
                is_cai = "label-warning" in row[3]      # 状态列含 label-warning 样式即「采标」=仅有题录、不可下载
                hit = (not args.search) or (args.search in row[5])  # --search 对标准名称做子串过滤
                count += 1
                cai += is_cai
                if hit:
                    print(f"[{'采' if is_cai else ' '}] p1={p1} {row[2].strip():<20} "
                          f"{row[5].strip()[:52]:<54} {row[1]}")
            if len(rows) < 50:                   # 本页不足 50 条 = 已是最后一页，停止翻页
                break
        # 每种类型的汇总打到 stderr，避免与逐行清单混在一起
        print(f"--- p.p1={p1}: 共 {count} 条, 采标(不可下载) {cai}, 可下载(非采标) {count - cai}\n",
              file=sys.stderr)


if __name__ == "__main__":
    main()
