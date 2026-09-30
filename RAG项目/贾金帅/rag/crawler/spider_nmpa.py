# -*- coding: utf-8 -*-
"""NMPA 药品数据源：批准文号 / OTC 监管分类 / 基药药理分类。

自 ``spiders.py`` 拆出（单一数据源一个文件，便于定位）。
通用名归一化与 OTC 目录比对（``normalize_name`` / ``match_otc_name``）
也放在这里，因为只有 NMPA 源在用它。
"""
import re

from .base import BrowserSpider, get_logger


# =====================================================================
# 1) NMPA 药品
# =====================================================================
# NMPA 数据查询各数据库的 itemId（2026-08 验证有效，改版需重新抓包确认）
ITEM_MAIN = "ff80808183cad75001840881f848179f"        # 境内生产药品


ITEM_FOREIGN = "ff80808183cad7500184088665711800"     # 境外生产药品


ITEM_OTC_CHEM = "ff808081845be229018464992d1d0264"   # 非处方药化学药品目录


ITEM_OTC_TCM = "ff808081845be229018464989dc90242"    # 非处方药中药目录


ITEM_JIBEN = "2c9ba384759c957701759cc91ecf029e"      # 国家基本药物（2018 年版）


# 每个通用名最多保留几条批准文号（默认 1 = 一个药对应一个厂家）。
# 降数据量（避免一个药挂几十条）+ 降瑞数请求压力（只翻 1 页即可）。
# 想保留更多厂家时改大这个数即可。
MAX_RECORDS_PER_DRUG = 1


# 批准文号首字母 -> 物质来源
APPROVAL_MAP = {
    "H": "化学药", "Z": "中药", "S": "生物制品",
    "J": "进口分装药品", "F": "药用辅料", "B": "保健药品",
}


# 产品名后缀 -> 剂型
DOSAGE_MAP = [
    ("缓释片", "缓释片剂"), ("缓释胶囊", "缓释胶囊剂"), ("控释片", "控释片剂"),
    ("肠溶片", "肠溶片剂"), ("肠溶胶囊", "肠溶胶囊剂"), ("泡腾片", "泡腾片剂"),
    ("分散片", "分散片剂"), ("咀嚼片", "咀嚼片剂"), ("含片", "含片剂"),
    ("软胶囊", "软胶囊剂"), ("滴丸", "滴丸剂"), ("颗粒", "颗粒剂"),
    ("口服溶液", "口服溶液剂"), ("口服液", "口服液"), ("混悬液", "混悬剂"),
    ("干混悬剂", "干混悬剂"), ("注射液", "注射剂"), ("注射用", "注射剂"),
    ("冻干粉针", "注射剂"), ("粉针", "注射剂"), ("软膏", "软膏剂"),
    ("乳膏", "乳膏剂"), ("凝胶", "凝胶剂"), ("栓剂", "栓剂"), ("栓", "栓剂"),
    ("喷雾", "喷雾剂"), ("气雾", "气雾剂"), ("吸入", "吸入剂"), ("贴剂", "贴剂"),
    ("贴膏", "贴膏剂"), ("贴", "贴剂"), ("滴眼", "滴眼剂"), ("滴鼻", "滴鼻剂"), ("滴耳", "滴耳剂"),
    ("滴剂", "滴剂"), ("糖浆", "糖浆剂"), ("散剂", "散剂"), ("散", "散剂"),
    ("酊", "酊剂"), ("搽剂", "搽剂"), ("丸", "丸剂"),
    ("片", "片剂"), ("胶囊", "胶囊剂"),
]


# 基药目录查不到时的药理分类兜底（可自行扩充）
DEFAULT_PHARMACOLOGY = {
    "布洛芬": "解热镇痛抗炎药", "对乙酰氨基酚": "解热镇痛抗炎药", "阿司匹林": "解热镇痛抗炎药",
    "阿莫西林": "抗生素-青霉素类", "头孢克肟": "抗生素-头孢菌素类",
    "奥美拉唑": "质子泵抑制剂", "二甲双胍": "降糖药-双胍类", "氨氯地平": "降压药-钙通道阻滞剂",
}


def parse_approval(approval_no):
    """批准文号 -> 物质来源。如 国药准字H20123196 -> 化学药"""
    m = re.search(r"国药准字\s*([HZSJFB])", str(approval_no or ""))
    return APPROVAL_MAP.get(m.group(1), "未知") if m else ""


def derive_dosage(product_name):
    """从产品名称推导剂型（最长匹配优先，如『缓释片』先于『片』）"""
    name = str(product_name or "")
    for kw, form in DOSAGE_MAP:
        if kw in name:
            return form
    return ""


# 规格/括号正则：把「布洛芬片(0.1g)」「对乙酰氨基酚片 0.5g」归一成品种名再比对
_PAREN_RE = re.compile(r"[\(（][^)）]*[)）]")


_SPEC_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|ug|iu|单位)", re.I)


def normalize_name(name):
    """品种名归一化：去括号内容与规格数字，便于跨来源比对该不该算同一个品种。"""
    s = str(name or "").strip()
    s = _PAREN_RE.sub("", s)
    s = _SPEC_RE.sub("", s)
    return s.strip()


def match_otc_name(product_name, otc_names):
    """产品名称是否命中 OTC 目录里的某个品种名。

    只做「归一化后精确相等」，刻意不做前缀/包含模糊匹配：
    复方品种（如 布洛芬伪麻黄碱片）不是同一个 OTC 品种，模糊匹配会把整药
    连同它的处方药规格一起算成 OTC —— 这正是旧逻辑判出 81.8% 非处方药的主因。
    """
    target = normalize_name(product_name)
    if not target:
        return False
    return any(target == normalize_name(n) for n in otc_names)


class NMPASpider(BrowserSpider):
    """境内生产药品 爬虫（继承浏览器基类）。"""

    # ---------- 开跑前自检 ----------
    def check_health(self, probe="布洛芬", retries=1):
        """会话与 itemId 健康检查：用探针药查一次列表接口，通了再开跑。

        NMPA 改版会让顶部那串硬编码 ITEM_* 全部失效，症状是「每个查询都返回
        None」。旧代码会静默吞掉，一路产出空 records，最后攒出一堆空记录，人还
        以为抓成功了。这里先探一次，失效直接抛错，别让人白跑几小时。
        """
        logger = get_logger()
        for attempt in range(retries + 1):
            r = self.api_query(probe, ITEM_MAIN, page_num=1)
            if r is not None:
                total = r.get("total", 0)
                if not total:
                    logger.warning("探针「%s」查到 0 条：itemId 可能已变，结果需人工核对", probe)
                else:
                    logger.info("NMPA 接口自检通过（探针「%s」命中 %d 条）", probe, total)
                return total
            if attempt < retries:      # 可能是会话掉了，重载首页再试一次
                logger.info("探针查询无响应，重载首页重试...")
                try:
                    self.load_home()
                except Exception as e:  # noqa: BLE001
                    logger.debug("重载首页失败：%s", e)
        raise RuntimeError(
            "NMPA 列表接口无响应（探针「%s」查询失败）。常见原因：\n"
            "  1) 官网改版，spider_nmpa.py 顶部 ITEM_* 的硬编码 itemId 已失效，需重新抓包确认；\n"
            "  2) 瑞数挑战未通过 —— 确认 EDGE_PATH 指向的浏览器能正常打开 NMPA 首页；\n"
            "  3) 网络不通或代理异常。\n"
            "不要忽略这个错误：继续跑只会产出大量空记录。" % probe)

    def verify_session(self, probe="布洛芬"):
        """会话是否还活着：用已知有数据的探针药查一次。返回 True/False（不抛错）。

        用于 run 层判断「某药 0 记录」到底是『真没有』还是『会话刚过期』：
        探针也查不到 = 会话死了，应刷新后重试；探针能查到 = 该药确实无记录。
        """
        try:
            total = self.check_health(probe=probe, retries=0)
            return total is not None
        except Exception:  # noqa: BLE001
            return False

    def _query(self, keyword, item_id, page_num=1, page_size=20, tries=2):
        """带会话自愈的 pajax 列表查询：返回 None 多半是会话过期（瑞数令牌失效），
        自动刷新会话重试，最多 tries 次。

        关键区分：返回有效空结果（total=0、list=[]）属『该库确实没这药』，立即返回、
        不重试；只有 pajax 调用本身失败（返回 None）才判定为会话问题并重试。
        """
        logger = get_logger()
        for attempt in range(tries + 1):
            r = self.api_query(keyword, item_id, page_num, page_size)
            if r is not None:
                return r
            if attempt < tries:
                logger.warning("[%s] 查询无响应（会话可能过期），刷新会话重试 %d/%d",
                               keyword, attempt + 1, tries)
                self._refresh_session()
        return None

    # ---------- 药品批准文号记录 ----------
    def fetch_main(self, keyword, max_records=MAX_RECORDS_PER_DRUG):
        """翻页抓批准文号记录，按批准文号去重；每通用名最多保留 max_records 条
        （默认 1 = 一个药对应一个厂家）。达到上限即停翻页，省瑞数请求。
        返回 [{'批准文号','产品名称','生产单位','本位码'}, ...]"""
        records, seen, total = [], set(), None
        for page_no in range(1, self.max_pages + 1):
            r = self._query(keyword, ITEM_MAIN, page_num=page_no)
            if not r:
                break
            total = r.get("total", total)
            for rec in r.get("list") or []:
                key = rec.get("f0") or ""
                if key and key not in seen:
                    seen.add(key)
                    records.append({
                        "批准文号": rec.get("f0", ""),
                        "产品名称": rec.get("f1", ""),
                        "生产单位": rec.get("f2", ""),
                        "本位码": rec.get("f3", ""),
                    })
                    if len(records) >= max_records:
                        break
            if total and len(records) >= total:
                break
            if len(records) >= max_records:
                break
        return records

    def fetch_otc_names(self, keyword, item_id, max_pages=10):
        """非处方药目录：翻页拉全后返回品种名列表。

        原来只查第 1 页 20 条 —— 热门通用名的 OTC 品种远不止 20 条，漏掉的那些
        会被当成「不在 OTC 目录里」，把整药误判成处方药。这里改成翻页拉全。
        """
        names, total = [], None
        for page_no in range(1, max_pages + 1):
            r = self._query(keyword, item_id, page_num=page_no)
            if not r:
                break
            total = r.get("total", total)
            for rec in r.get("list") or []:
                n = rec.get("f0", "")
                if n:
                    names.append(n)
            if total and len(names) >= total:
                break
        return names

    def fetch_jiben_category(self, keyword):
        """国家基本药物目录：查第 1 页，返回（三级目录=药理分类, 品种名称）。"""
        r = self._query(keyword, ITEM_JIBEN, page_num=1)
        if not r:
            return "", ""
        for rec in r.get("list") or []:
            cat = rec.get("f2") or rec.get("f1", "")  # 三级目录优先，无则用二级
            name = rec.get("f3", "")
            if cat and name and (keyword in name or name in keyword):
                return cat, name
        return "", ""

    def crawl_drug(self, keyword):
        """抓取一个药品：批准文号记录 + 逐条监管分类 + 基药药理分类。

        监管分类下沉到「每条批准文号」判定：拿该条的产品名去 OTC 目录做品种名
        精确匹配，命中=非处方药，未命中=处方药。整药层面再按命中比例汇总成
        非处方药 / 处方药 / 双跨，并把依据写进 info['监管分类依据']，让下游能
        区分「目录精确命中」与「未命中推断」两种可信度。

        旧逻辑是按「剂型集合包含关系」整体判定的（只要 OTC 目录里有一个同剂型
        品种，整药全部批准文号都算 OTC），粒度太粗，判出 81.8% 非处方药，已废弃。
        """
        logger = get_logger()
        logger.info("[%s] 开始...", keyword)
        result = {"keyword": keyword, "records": [], "info": {}}

        main_records = self.fetch_main(keyword)
        if main_records:
            result["info"]["批准文号"] = main_records[0]["批准文号"]
            result["info"]["产品名称"] = main_records[0]["产品名称"]
            result["info"]["生产单位"] = main_records[0]["生产单位"]

        # OTC 目录（翻页拉全）-> 逐条批准文号判定
        otc_names = self.fetch_otc_names(keyword, ITEM_OTC_CHEM) + \
                    self.fetch_otc_names(keyword, ITEM_OTC_TCM)
        hit = 0
        for rec in main_records:
            if match_otc_name(rec.get("产品名称", ""), otc_names):
                rec["监管分类"] = "非处方药"
                rec["监管分类依据"] = "OTC目录品种名命中"
                hit += 1
            else:
                rec["监管分类"] = "处方药"
                rec["监管分类依据"] = "OTC目录无此品种（按处方药计）"
        result["records"] = main_records

        # 整药层面汇总
        # 注：MAX_RECORDS_PER_DRUG=1 时每药只保留 1 条批准文号记录，监管分类只可能落为
        # 「处方药」或「非处方药」；下面 双跨 分支只在保留 >1 条记录（即一个药多个厂家）时
        # 才可能出现。改大 MAX_RECORDS_PER_DRUG 即可恢复多厂家 + 双跨判定，无需动这里。
        if not main_records:
            result["info"]["监管分类"] = ""
            result["info"]["监管分类依据"] = "未抓到批准文号记录"
        elif hit == 0:
            result["info"]["监管分类"] = "处方药"
            result["info"]["监管分类依据"] = "OTC目录无命中（推断）"
            if otc_names:
                # 目录有货却一条没匹配上，多半是字段位置变了（改版）或名称写法差异，
                # 打出来方便核对，别让它悄悄变成「全是处方药」。
                logger.debug("[%s] OTC 目录有 %d 条但无一命中，样本：%s",
                             keyword, len(otc_names), otc_names[:3])
        elif hit == len(main_records):
            result["info"]["监管分类"] = "非处方药"
            result["info"]["监管分类依据"] = "OTC目录命中全部 %d 条" % hit
        else:
            result["info"]["监管分类"] = "双跨"
            result["info"]["监管分类依据"] = "OTC目录命中 %d/%d 条" % (hit, len(main_records))

        # 基药目录 -> 药理分类（三级目录）
        cat, _ = self.fetch_jiben_category(keyword)
        if not cat:
            cat = DEFAULT_PHARMACOLOGY.get(keyword, "")
        result["info"]["药理分类"] = cat

        logger.info("[%s] 完成：%d 条批准文号记录，%s",
                    keyword, len(main_records), result["info"]["监管分类依据"])
        return result


def load_name_list(path):
    """读药品名单文件（每行一个通用名），返回去空行后的列表"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
