# -*- coding: utf-8 -*-
"""全部数据源的抓取逻辑（浏览器驱动见 base.py，文档契约见 schema.py）。

本文件保留 4 个源：
  1) 中国临床试验（chictr）   —— 浏览器驱动，过阿里云 WAF
  2) 万方文献（wanfang）       —— 浏览器驱动，抓中文摘要
  3) 指南（guideline）         —— URL 列表驱动，下载 PDF 抽文本
  4) 处方药转 OTC 公告（otc_ann）—— 爬公告列表，下载附件(PDF/docx)原件

另有 2 部分已拆分到独立文件（本文件底部 re-export，对外接口不变）：
  - ``spider_nmpa.py``：NMPA 药品（批准文号 / OTC / 基药）
  - ``spider_yibao.py``：医保目录 + 结果合并（build_output）

约定：爬虫只「抓 + 解析 + 返回 dict」，不写文件；落盘 / 断点 / 失败记录
全部由 pipeline.py 统一处理。otc_ann 例外：附件下载在 pipeline.py 里
用 base.http_get 完成（浏览器只负责加载被瑞数保护的页面以拿到附件链接）。
"""
import hashlib
import json
import os
import re
import time
from urllib.parse import urljoin

from .base import BrowserSpider, get_logger
from .schema import build_doc, safe_id
# 拆分出去的实现，re-export 保证 `from .spiders import X` 继续可用
from .spider_nmpa import (NMPASpider, derive_dosage, load_name_list,
                          match_otc_name, normalize_name, parse_approval)
from .spider_yibao import YibaoCatalog, build_output


# =====================================================================
# 3) 中国临床试验注册中心（chictr）
# =====================================================================
SEARCH_URL = "https://www.chictr.org.cn/searchproj.html"


DETAIL_URL = "https://www.chictr.org.cn/showproj.html?proj=%s"


KEY_FIELDS = ["注册号", "注册题目", "注册时间", "注册号状态", "研究类型",
              "研究设计", "干预措施", "入选标准", "研究目的", "申请单位", "研究负责人"]


_LABEL_END = re.compile(r"[：:]$")


def resolve_term_cn(line):
    """检索词取 tab 左侧（中文名）；无 tab 原样返回。"""
    return line.split("\t")[0].strip()


def extract_fields(text):
    """从详情页全文提取关键字段：`标签：\n值` 结构，跳过连续标签行。"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    out = {}
    for i, l in enumerate(lines):
        if _LABEL_END.search(l):
            label = _LABEL_END.sub("", l).strip()
            if label in KEY_FIELDS:
                j = i + 1
                while j < len(lines) and _LABEL_END.search(lines[j]):
                    j += 1
                if j < len(lines):
                    out[label] = lines[j]
    return out


class ChiCTRSpider(BrowserSpider):
    """chictr 爬虫：搜索页表格 -> 详情页方案书，逐条 yield 统一 schema doc。"""

    def crawl_term(self, term, max_docs, done):
        seen, page_no = set(), 1
        while len(seen) < max_docs and page_no <= 10:
            rows = self._search_rows(term, page_no)
            if not rows:
                break
            fresh = 0
            for regno, title, unit, stype, detail in rows:
                if not regno or regno in seen:
                    continue
                seen.add(regno)
                doc_id = "chictr:" + regno
                if safe_id(doc_id) in done:   # 已落盘则跳过，避免重复抓详情
                    seen.add(regno)
                    continue
                if not detail:
                    continue
                doc = self._crawl_detail(detail, regno)
                if not doc:
                    continue
                yield doc
                fresh += 1
                if len(seen) >= max_docs:
                    break
            if fresh == 0 and len(seen) == 0:   # 连续无新数据，防死循环
                break
            page_no += 1

    def _search_rows(self, term, page_no):
        self.navigate(SEARCH_URL + "?page=%d&title=%s&btngo=btn" % (page_no, term), wait=6)
        tables = self.page.eles("tag:table")
        if not tables:
            return []
        rows = []
        for tr in tables[0].eles("tag:tr")[1:]:
            cells = [c.text.strip() for c in tr.eles("tag:td")]
            if len(cells) < 4:
                continue
            regno = cells[1].strip()
            title_unit = cells[2].strip().split("\n")
            title = title_unit[0].strip()
            unit = title_unit[1].strip() if len(title_unit) > 1 else ""
            links = [a.attr("href") for a in tr.eles("tag:a") if a.attr("href")]
            detail = next((l for l in links if "showproj" in l), "")
            rows.append((regno, title, unit, cells[3].strip(), detail))
        return rows

    def _crawl_detail(self, detail_url, regno):
        self.navigate(detail_url, wait=self.interval + 3)
        text = ""
        for t in self.page.eles("tag:table")[:3]:
            text += (t.text or "") + "\n"
        if len(text) < 200:
            return None
        fields = extract_fields(text)
        return build_doc(
            "chictr", regno, detail_url, fields.get("注册题目", regno),
            fields.get("注册时间", ""),
            {k: v for k, v in fields.items() if v}, text, [],
            doc_id="chictr:" + regno)


# =====================================================================
# 4) 万方中文文献（wanfang）
# =====================================================================
EXTRACT_JS = """
var out = [];
document.querySelectorAll('.normal-list').forEach(function(item){
  var link = '';
  var a = item.querySelector('a[href*="detail"]') || item.querySelector('a[href*="periodical"]') || item.querySelector('a[href^="http"]');
  if (a) link = a.href;
  var t = item.querySelector('.title') || item.querySelector('.title-text') || item.querySelector('strong');
  out.push({ title: (t ? t.textContent : '').trim().slice(0,300), link: link,
             text: item.innerText.trim().slice(0,3000) });
});
return JSON.stringify(out);
"""


WANFANG_URL = "https://s.wanfangdata.com.cn/paper?q=%s"


def term_id(term):
    """doc_id 用标题哈希（万方无稳定原生 ID）"""
    return "wf_" + hashlib.md5(term.encode("utf-8")).hexdigest()[:12]


def parse_item(item):
    """从抓取的 item dict 解析出 标题 / 期刊 / 年份 / 摘要"""
    lines = [l.strip() for l in item["text"].split("\n") if l.strip()]
    title = (item["title"] or (lines[0] if lines else "")).strip()
    meta, abstract = "", ""
    for l in lines:
        if l.startswith("摘要") or l.startswith("Abstract"):
            abstract = l[3:].lstrip("：: ").strip()
        elif l.startswith("[") and not meta:
            meta = l
    year = ""
    for part in meta.split():
        if part[:4].isdigit() and "年" in part:
            year = part[:4]
            break
    return title, meta, year, abstract


class WanfangSpider(BrowserSpider):
    """万方爬虫：搜索结果 SPA 渲染后抓 DOM，逐条 yield 统一 schema doc。"""

    def crawl_term(self, term, max_docs, done):
        seen, page_no = set(), 1
        while len(seen) < max_docs and page_no <= 50:
            items = self._search_items(term, page_no)
            if not items:
                break
            fresh = 0
            for it in items:
                title, meta, year, abstract = parse_item(it)
                if not title or not abstract:
                    continue
                if not any("一" <= c <= "鿿" for c in abstract):  # 只收中文
                    continue
                doc_id = "wanfang:" + term_id(title)
                if doc_id in seen or safe_id(doc_id) in done:
                    continue
                seen.add(doc_id)
                yield build_doc(
                    "wanfang", term_id(title),
                    it.get("link") or (WANFANG_URL % term), title, year,
                    {"摘要": abstract, "期刊/作者": meta}, abstract, [],
                    doc_id=doc_id)
                fresh += 1
                if len(seen) >= max_docs:
                    break
            if fresh == 0 or len(seen) >= max_docs:
                break
            time.sleep(self.interval)
            page_no += 1

    def _search_items(self, term, page_no):
        url = WANFANG_URL % term
        if page_no > 1:
            url += "&p=%d" % page_no
        self.navigate(url, wait=10)
        raw = self.page.run_js(EXTRACT_JS)
        try:
            items = json.loads(raw)
        except Exception:
            return None
        if len(items) < 10:        # 页面渲染慢：首抓不足 10 条补一次
            time.sleep(6)
            try:
                items = json.loads(self.page.run_js(EXTRACT_JS))
            except Exception:
                pass
        return items


# =====================================================================
# 5) 指南 PDF（guideline）
# =====================================================================
def extract_pdf(path):
    """用 PyMuPDF 抽取 PDF 全文；失败返回空串。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        get_logger().error("需要 PyMuPDF: pip install PyMuPDF")
        return ""
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:  # noqa: BLE001
        get_logger().error("PDF 打开失败 %s: %s", path, e)
        return ""


# =====================================================================
# 6) 处方药转换为非处方药公告（otc_ann）
# =====================================================================
# NMPA 的「转换为非处方药」公告频道。列表页与公告详情页都被瑞数(412)保护，
# 必须用真浏览器加载才能拿到内容；但附件(.docx/.doc/.pdf)是静态文件，可直接 http 下载。
LISTING_URL = "https://www.nmpa.gov.cn/yaopin/ypggtg/index.html"  # 2026-09 改版后列表页迁至此，旧路径 302 回首页


_LINK_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.S | re.I)


_ATT_RE = re.compile(r'\.(docx?|pdf|wps)$', re.I)


_DATE_RE = re.compile(r'发布时间[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)', re.I)


_TITLE_RE = re.compile(r'<title>(.*?)</title>', re.S | re.I)


def ann_id_from_url(url):
    """从公告 URL 取出数字 id 作文件夹名，如 20260701172735104。"""
    m = re.search(r'/(\d{14,})\.html', url)
    return m.group(1) if m else hashlib.md5(url.encode()).hexdigest()[:16]


def parse_listing(html, base=LISTING_URL):
    """从列表页 HTML 提取「转换为非处方药」公告链接 -> [(title, url)]。"""
    out, seen = [], set()
    for href, text in _LINK_RE.findall(html or ""):
        t = re.sub(r'<[^>]+>', '', text).strip()
        if '转换为非处方药' not in t and '转换为非处方药的公告' not in t:
            continue
        if href.lower().startswith('javascript') or href == '#':
            continue
        url = urljoin(base, href)
        if url in seen:
            continue
        seen.add(url)
        out.append((t, url))
    return out


def parse_ann_page(html, base=LISTING_URL):
    """从公告详情页 HTML 提取 (标题, 发布日期, [(附件名, 附件URL)])。"""
    html = html or ""
    title = ''
    m = _TITLE_RE.search(html)
    if m:
        title = re.sub(r'<[^>]+>', '', m.group(1)).strip()
    date = ''
    dm = _DATE_RE.search(html)
    if dm:
        date = dm.group(1).replace('年', '-').replace('月', '-').replace('日', '').strip('-')
    atts = []
    for href, text in _LINK_RE.findall(html):
        if not _ATT_RE.search(href):
            continue
        name = re.sub(r'<[^>]+>', '', text).strip() or os.path.basename(href)
        atts.append((name, urljoin(base, href)))
    return title, date, atts


class OtcAnnSpider(BrowserSpider):
    """爬 NMPA「转换为非处方药」公告：浏览器只用于加载列表/详情页拿链接，
    附件下载交给 cli.py 用 http_get（静态文件，直下即可）。"""

    def list_announcements(self, max_pages=3):
        self.page.get(LISTING_URL)
        time.sleep(3)
        results, seen = [], set()
        for _ in range(max_pages):
            for title, url in parse_listing(self.page.html):
                if url not in seen:
                    seen.add(url)
                    results.append((title, url))
            nxt = self._next_page_link()
            if not nxt:
                break
            self.page.get(nxt)
            time.sleep(3)
        return results

    def _next_page_link(self):
        try:
            a = self.page.ele('xpath://a[contains(text(),"下一页")]', timeout=2)
            if a:
                href = a.attr('href')
                if href and href.lower() not in ('#', 'javascript:;'):
                    return urljoin(LISTING_URL, href)
        except Exception:
            pass
        return None

    def ann_page_attachments(self, url):
        self.page.get(url)
        time.sleep(3)
        return parse_ann_page(self.page.html)

__all__ = [
    "ChiCTRSpider", "WanfangSpider", "OtcAnnSpider", "NMPASpider", "YibaoCatalog",
    "build_output", "extract_pdf", "load_name_list", "resolve_term_cn",
    "ann_id_from_url", "parse_approval", "derive_dosage",
    "normalize_name", "match_otc_name",
]

