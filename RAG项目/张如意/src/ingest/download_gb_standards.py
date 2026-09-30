# -*- coding: utf-8 -*-
"""从全国标准信息公共服务平台(openstd.samr.gov.cn) 批量下载国家标准 PDF（按 ICS + 标准类型）。

用法:
  python download_gb_standards.py --ics 11 --p1 3 [--out data/raw] [--max N] [--delay 2.5]
  python download_gb_standards.py --url "<std_list_type 完整URL>" --exclude 实验动物

通道说明 (已在真实站点验证):
  列表页 -> 解析行 -> 跳过 采标(label-warning, 平台只给题录) -> 逐个打开
  newGbInfo?hcno=ID -> 点击 button.xz_btn -> 弹窗 showGb -> JS 自动跳 viewGb?hcno=ID。
  viewGb 由 window.location.href 导航加载, 所以 page.on("response") 拿不到 body;
  必须在 context 上 route("**/viewGb*") + route.fetch().body() 抓取 PDF 字节。

限流: 平台对高频请求会翻到验证码闸门, 表现为抓不到 PDF。重跑本脚本即可补齐(已存在的文件会跳过)。
"""
import argparse
import os
import re
import time

from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")   # 伪装成 Chrome 浏览器

LIST_URL = ("https://openstd.samr.gov.cn/bzgk/std/std_list_type"
            "?p.p1={p1}&p.p6={ics}&p.p90=circulation_date&p.p91=desc&pageSize=50&page=1")
PAGE_SIZE = 50
MAX_PAGES = 20          # 保险上限，防止翻页死循环


def page_url(base, n):
    """把列表 URL 里的 page=N 换成目标页；没有该参数则追加。"""
    if re.search(r"(?<=[?&])page=\d+", base):
        return re.sub(r"(?<=[?&])page=\d+", f"page={n}", base)
    return f"{base}{'&' if '?' in base else '?'}page={n}"

# p.p1: 1=强制性GB, 2=推荐性GB/T, 3=指导性GB/Z, 6=外文版
LIST_ROW_RE = re.compile(
    r"<tr>\s*<td>(\d+)</td>\s*<td[^>]*><a[^>]*onclick=\"showInfo\('([0-9A-F]{32})'\);\">([^<]+)</a></td>\s*<td>(.*?)</td>\s*<td class=\"mytxt\"[^>]*>.*?onclick=\"showInfo\('([0-9A-F]{32})'\);\">([^<]+)</a>",
    re.S)


def parse_list(html, exclude=()):
    """解析列表页 HTML，返回标准条目 dict 列表。

    每条含：hcno（详情页 ID）、stdno（标准号）、name（标准名）、
    cai（是否采标——采标只给题录，平台不提供 PDF 下载）、
    excluded（标准名是否命中排除关键词）。
    """
    items = []
    for _num, hcno, stdno, badge, _hcno2, name in LIST_ROW_RE.findall(html):
        stdno, name = stdno.strip(), name.strip()
        items.append({
            "hcno": hcno,
            "stdno": stdno,
            "name": name,
            "cai": "label-warning" in badge,          # 采标 -> 平台不提供下载
            "excluded": any(k in name for k in exclude),
        })
    return items


def safe_filename(stdno, name):
    """标准号 + 标准名称, 去掉 Windows 非法字符。"""
    return re.sub(r'[\\/:*?"<>|\r\n]', "_", f"{stdno} {name}").strip()


def main():
    """主流程：Playwright 打开列表页逐页解析 → 过滤采标/排除项 →
    逐条打开详情页、点下载按钮、经 route 拦截抓取 PDF 字节落盘。
    已存在（>20KB）的文件跳过；两条下载之间 sleep --delay 秒限流。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ics", default="65", help="ICS 代码, 65=农业, 11=医药卫生")
    ap.add_argument("--p1", default="3", help="标准类型: 1=GB 2=GB/T 3=GB/Z")
    ap.add_argument("--url", default="", help="直接给完整 std_list_type URL(优先于 --ics/--p1)")
    ap.add_argument("--exclude", default="", help="逗号分隔的排除关键词(命中标准名则跳过)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max", type=int, default=0, help="最多下载几条(0=全部)")
    ap.add_argument("--delay", type=float, default=3.0, help="两条之间的间隔秒数")
    args = ap.parse_args()

    url = args.url or LIST_URL.format(p1=args.p1, ics=args.ics)
    exclude = tuple(k.strip() for k in args.exclude.split(",") if k.strip())
    os.makedirs(args.out, exist_ok=True)

    with sync_playwright() as p:
        # 无头模式启动本机 Chrome（channel="chrome"），不走 Playwright 自带浏览器
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(accept_downloads=True, user_agent=UA, locale="zh-CN")
        page = ctx.new_page()

        allitems, seen = [], set()
        for n in range(1, MAX_PAGES + 1):
            page.goto(page_url(url, n), timeout=60000)
            page.wait_for_timeout(1200)   # 等待页面渲染完成再取 HTML
            rows = parse_list(page.content(), exclude)
            for r in rows:                              # 按 hcno 去重（翻页时可能重复）
                if r["hcno"] not in seen:
                    seen.add(r["hcno"])
                    allitems.append(r)
            if len(rows) < PAGE_SIZE:
                break   # 本页不足一整页 → 已到最后，停止翻页
        todo = [i for i in allitems if not i["cai"] and not i["excluded"]]
        print(f"列表共 {len(allitems)} 条 | 采标跳过 {sum(i['cai'] for i in allitems)} | "
              f"关键词跳过 {sum(i['excluded'] for i in allitems)} | 待下载 {len(todo)}", flush=True)

        ok = fail = 0
        for idx, it in enumerate(todo[: args.max or None], 1):
            fn = os.path.join(args.out, safe_filename(it["stdno"], it["name"]) + ".pdf")
            if os.path.exists(fn) and os.path.getsize(fn) > 20000:
                print(f"[{idx}] 已存在 {it['stdno']}", flush=True)
                ok += 1
                continue

            captured = []

            def route_handler(route):
                # 拦截 viewGb 响应：viewGb 由 JS 导航加载，page.on("response") 拿不到 body，
                # 必须在这里 route.fetch() 主动取字节
                resp = route.fetch()
                body = resp.body()
                # 校验 PDF 头，命中才落盘并记录大小（captured 非空即视为成功）
                if body[:5] == b"%PDF-":
                    with open(fn, "wb") as f:
                        f.write(body)
                    captured.append(len(body))
                route.continue_()

            ctx.route("**/viewGb*", route_handler)
            work = ctx.new_page()
            try:
                work.goto(f"https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno={it['hcno']}",
                          timeout=60000)
                work.wait_for_timeout(1000)
                btn = work.query_selector("button.xz_btn")
                if not btn:
                    print(f"[{idx}] {it['stdno']}: 无下载按钮(暂无全文)，跳过", flush=True)
                    fail += 1
                else:
                    btn.click()
                    for _ in range(20):                 # 最多等 20s
                        if captured:
                            break
                        work.wait_for_timeout(1000)
                    if captured:
                        print(f"[{idx}] OK {it['stdno']} {it['name'][:32]} -> "
                              f"{os.path.basename(fn)} ({captured[0] // 1024} KB)", flush=True)
                        ok += 1
                    else:
                        print(f"[{idx}] 失败 {it['stdno']}: 未抓到 PDF(可能被验证码闸门拦住)", flush=True)
                        fail += 1
            except Exception as e:
                print(f"[{idx}] 异常 {it['stdno']}: {type(e).__name__} {e}", flush=True)
                fail += 1
            finally:
                ctx.unroute("**/viewGb*", route_handler)
                work.close()
            time.sleep(args.delay)   # 限流：两条下载之间留间隔，防触发验证码闸门

        browser.close()
    print(f"完成: 成功 {ok} / 失败 {fail}", flush=True)


if __name__ == "__main__":
    main()
