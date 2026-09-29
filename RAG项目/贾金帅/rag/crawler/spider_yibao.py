# -*- coding: utf-8 -*-
"""医保目录（通用名 -> 甲乙类）与 NMPA 结果合并。

自 ``spiders.py`` 拆出。目录下载/解析依赖 ``base`` 与 ``schema``；
批准文号、剂型工具复用 ``spider_nmpa``。
"""
import os

from .base import get_logger, now_iso
from .schema import PATHS
from .spider_nmpa import derive_dosage, parse_approval


# 医保目录官方下载入口（通知页 -> 页内附件）。医保目录每年发新版，URL 会变：
# 下载失败不会静默，会明确告警并提示手动放置文件的路径（见 YibaoCatalog.ensure）。
YIBAO_ARTICLES = [
    "https://www.nhsa.gov.cn/art/2024/11/28/art_104_14886.html",
]


def build_output(crawl_results, yibao_lookup=None):
    """把药品爬取结果合并为用户指定格式的 JSON 数组（每条批准文号记录一项）

    抓不到任何批准文号的药品直接跳过、不产出条目。以前会补一个空 dict 凑成
    一条「全空记录」混进结果（实测 26 条），下游分不清「抓失败」和「确实没有」；
    现在改为跳过，失败药名由调用方另记 nmpa_failed.jsonl。
    """
    out = []
    for cr in crawl_results:
        info, records, keyword = cr["info"], cr["records"], cr["keyword"]
        if not records:
            continue
        for rec in records:
            approval = rec.get("批准文号", "") or info.get("批准文号", "")
            item = {
                "药品通用名": keyword,
                "物质来源": parse_approval(approval),
                "监管分类": rec.get("监管分类") or info.get("监管分类", ""),
                "监管分类依据": rec.get("监管分类依据") or info.get("监管分类依据", ""),
                "药理分类": info.get("药理分类", ""),
                "剂型": derive_dosage(rec.get("产品名称", "")),
                "医保类别": YibaoCatalog.fuzzy_get(yibao_lookup or {}, keyword),
                "国药准字": approval,
                "source_url": "https://www.nmpa.gov.cn/datasearch/",
                "crawled_at": now_iso(),
            }
            if rec.get("产品名称"):
                item["产品名称"] = rec["产品名称"]
            if rec.get("生产单位"):
                item["生产企业"] = rec["生产单位"]
            if rec.get("本位码"):
                item["药品本位码"] = rec["本位码"]
            out.append(item)
    return out


# =====================================================================
# 医保目录（通用名 -> 甲乙类）
# =====================================================================
def _looks_like_catalog(content):
    """粗校验下载内容是不是真的 PDF/Excel，防下到 HTML 错误页冒充目录文件。"""
    if content[:4] == b"%PDF":
        return True
    if content[:2] == b"PK":                                    # xlsx 本质是 zip
        return True
    if content[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":      # 老式 xls（OLE2）
        return True
    return False


class YibaoCatalog:
    """医保目录：下载官方文件（Excel/PDF）并解析 通用名 -> 甲乙类。"""

    @staticmethod
    def ensure(path=None, auto_download=True):
        """拿到可用的医保目录 lookup：本地有就解析，没有就先尝试下载。

        全失败时返回空 dict 并**明确告警**，不再静默留空字段 —— 医保类别空着
        会让整列失效（实测 4293 条全空就是因为这个方法以前根本没人调用）。
        """
        logger = get_logger()
        path = path or PATHS["yibao_file"]
        if os.path.exists(path):
            try:
                lookup = YibaoCatalog.load(path)
                logger.info("医保目录已加载：%s（%d 个品种）", path, len(lookup))
                return lookup
            except Exception as e:  # noqa: BLE001
                logger.error("医保目录解析失败（%s）：%s", path, e)
        if auto_download:
            try:
                YibaoCatalog.download(path)
                lookup = YibaoCatalog.load(path)
                logger.info("医保目录下载并解析成功：%d 个品种", len(lookup))
                return lookup
            except Exception as e:  # noqa: BLE001
                logger.error("医保目录自动下载失败：%s", e)
        logger.warning(
            "医保类别字段将全部留空。请到 国家医疗保障局官网 下载《国家基本医疗"
            "保险药品目录》官方文件（xlsx 或 pdf），放到 %s 后重跑。", path)
        return {}

    @staticmethod
    def download(save_to=None):
        """从医保局通知页抓目录附件。下到什么存什么，写完校验文件头。"""
        import requests
        save_to = save_to or PATHS["yibao_file"]
        logger = get_logger()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        s = requests.Session()
        s.headers.update(headers)
        for page_url in YIBAO_ARTICLES:
            try:
                r = s.get(page_url, timeout=20)
                r.encoding = "utf-8"
                if r.status_code != 200:
                    continue
                # 放宽：不再要求 URL 里含 download 字样，抓所有 xlsx/xls/pdf 链接
                links = re.findall(r'href="([^"]+)"', r.text)
                dl = [l for l in links
                      if re.search(r'\.(pdf|xlsx|xls)(\?|$)', l, re.I)]
                for link in dl:
                    if not link.startswith("http"):
                        link = "https://www.nhsa.gov.cn" + link
                    f = s.get(link, timeout=120)
                    if f.status_code != 200 or len(f.content) < 10000:
                        continue
                    if not _looks_like_catalog(f.content):
                        logger.debug("跳过非目录文件：%s", link)
                        continue
                    os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
                    with open(save_to, "wb") as fp:
                        fp.write(f.content)
                    logger.info("医保目录已下载：%s（%d KB）", save_to, len(f.content) // 1024)
                    return save_to
            except Exception as e:  # noqa: BLE001
                logger.debug("医保目录下载尝试失败 %s：%s", page_url, e)
                continue
        raise RuntimeError("医保目录自动下载失败（已尝试 %d 个入口）" % len(YIBAO_ARTICLES))

    @staticmethod
    def parse_excel(path):
        import pandas as pd
        lookup = {}
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            if "说明" in sheet or "凡例" in sheet:
                continue
            df = pd.read_excel(xl, sheet_name=sheet, header=None)
            header_row = None
            for idx in range(min(10, len(df))):
                row = " ".join(str(x) for x in df.iloc[idx].tolist())
                if "通用名" in row and ("甲乙" in row or "类别" in row):
                    header_row = idx
                    break
            if header_row is None:
                continue
            df.columns = [str(c).strip() for c in df.iloc[header_row]]
            df = df.iloc[header_row + 1:]
            name_col = next((c for c in df.columns if "通用名" in c or "药品名称" in c), None)
            jia_col = next((c for c in df.columns if "甲乙" in c or "类别" in c), None)
            if not name_col:
                continue
            for _, row in df.iterrows():
                name = str(row.get(name_col, "") or "").strip()
                if not name or name == "nan":
                    continue
                jia = str(row.get(jia_col, "") or "")
                if "甲" in jia:
                    lookup.setdefault(name, "甲类")
                elif "乙" in jia:
                    lookup.setdefault(name, "乙类")
        return lookup

    @staticmethod
    def parse_pdf(path):
        try:
            import pymupdf
        except ImportError:
            raise RuntimeError("解析 PDF 需要 PyMuPDF：pip install PyMuPDF")
        known_forms = ["口服常释剂型", "缓释控释剂型", "口服液体剂", "颗粒剂", "注射剂",
                       "栓剂", "乳膏剂", "贴膏剂", "巴布膏剂", "凝胶贴膏剂", "贴剂",
                       "肠溶胶囊", "软胶囊", "滴眼剂", "气雾剂", "吸入剂", "喷雾剂",
                       "口服溶液剂", "糖浆剂", "散剂", "胶囊剂", "片剂", "丸剂",
                       "滴丸", "凝胶剂", "软膏剂", "膏剂", "酊剂", "洗剂", "搽剂",
                       "植入剂", "粉针剂", "冻干粉针", "膜剂", "口服混悬剂",
                       "干混悬剂", "泡腾片", "咀嚼片", "含片", "分散片", "溶液剂",
                       "粉雾剂", "口腔崩解片", "外用溶液剂", "涂剂", "灌肠剂",
                       "硬膏剂", "眼膏剂", "锭剂", "缓释颗粒", "贴片", "橡皮膏",
                       "巴布膏", "口腔贴片", "滴鼻剂", "滴耳剂", "洗眼剂"]
        lookup = {}
        doc = pymupdf.open(path)
        for pno in range(doc.page_count):
            cur = None
            for w in doc[pno].get_text("words"):
                x0, text = w[0], w[4]
                if x0 < 205:
                    continue
                t = text.strip()
                if t in ("甲", "乙"):
                    if cur and cur["name"]:
                        name = "".join(cur["name"]).strip("()（） 　")
                        lookup.setdefault(name, "甲类" if cur["jia"] == "甲" else "乙类")
                    cur = {"name": [], "jia": t}
                elif t in ("第", "页") or t.startswith("★") or t.startswith("＊") \
                        or re.fullmatch(r"\d+", t) or t in known_forms:
                    continue
                elif cur:
                    cur["name"].append(t)
            if cur and cur["name"]:
                name = "".join(cur["name"]).strip("()（） 　")
                lookup.setdefault(name, "甲类" if cur["jia"] == "甲" else "乙类")
        doc.close()
        return lookup

    @staticmethod
    def fuzzy_get(lookup, keyword):
        """先精确匹配，再按包含关系模糊匹配（如 连花清瘟 -> 连花清瘟胶囊）"""
        if keyword in lookup:
            return lookup[keyword]
        hits = [v for k, v in lookup.items() if keyword in k or k in keyword]
        if hits:
            return "甲类" if "甲类" in hits else hits[0]
        return ""

    @staticmethod
    def load(path):
        if path.lower().endswith((".xlsx", ".xls")):
            return YibaoCatalog.parse_excel(path)
        if path.lower().endswith(".pdf"):
            return YibaoCatalog.parse_pdf(path)
        raise RuntimeError("不支持的医保目录文件类型（支持 xlsx/xls/pdf）")
