# -*- coding: utf-8 -*-
"""国家法律法规数据库抓取（**律师角色**知识，离线链路的数据准备环节）。

数据源：国家法律法规数据库 https://flk.npc.gov.cn/

接口逆向结论（2026-09 实测，逐条验证过）
----------------------------------------
* 分类树/计数：``GET  /law-search/search/enumData``、``GET /law-search/index/aggregateData``
* 列表：      ``POST /law-search/search/list``（JSON：searchContent/searchType/searchRange/
              flfgCodeId/orderByParam/pageNum/pageSize；缺 ``searchRange`` 会 500）
* 详情：      ``GET  /law-search/search/flfgDetails?bbbs=<bbbs>``（取 fileId 兜底）
* **下载直链**：``GET /law-search/download/pc?format=docx&bbbs=<bbbs>&fileId=<fileId>``
  -> ``data.url`` 是 OBS 签名直链（**1 小时有效**，鉴权在签名里，无需 Cookie）；
  ``format=pdf`` 只在确有 PDF 源文件时给 url，多数法规没有 -> 以 docx 为准。

链路：分类翻页列表 -> download/pc 取直链 -> 下载 .docx -> 抽取正文 -> 落盘 Markdown。

产物
----
* 原始件：``data/npc_lawyer/raw/<类别>/<名称>.docx``
* 正文：  ``knowledge/lawyer/<类别>/<名称>.md``（入库：``--role-id lawyer``）
* 进度：  ``index/npc_crawl_state.json``
  （**刻意放在 knowledge/ 之外**：``build_index`` 会 rglob 整个目录，
  状态文件落在里面会被当成语料入进向量库）

用法
----
    # 只列目录不下载（核对每个分类的篇数与字段）
    python scripts/crawl_npc_law.py --list-only

    # 小样验证（只抓宪法，限 2 篇）
    python scripts/crawl_npc_law.py --only 宪法 --limit 2 --rebuild-state

    # 全量（5 类，约 1471 篇）；默认礼貌限速 + 断点续跑
    python scripts/crawl_npc_law.py --workers 4 --min-interval 0.3

    # 只重试上一轮失败的篇目
    python scripts/crawl_npc_law.py --retry-failed

设计要点
--------
* **礼貌抓取**：全局最小请求间隔（``--min-interval``）+ 有界并发 + 指数退避重试；
* **断点续跑**：每篇结果写状态文件，已 ``ok`` 的篇目直接跳过（除非 ``--force``）；
* **单篇失败不影响整体**：逐篇 try/except，失败原因（含异常类型与消息）写进状态文件；
* **全程日志**：函数入口/出口、每次请求的数据出入口（``log_io``）、每篇的成败与耗时。
"""
from __future__ import annotations

import argparse
import html
import http.cookiejar
import json
import random
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.logging_setup import get_logger, setup_logging  # noqa: E402
from legal_rag.observability import log_io  # noqa: E402

logger = get_logger("scripts.crawl_npc_law")

BASE = "https://flk.npc.gov.cn"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "npc_lawyer" / "raw"
TEXT_DIR = ROOT / "knowledge" / "lawyer"
STATE_PATH = ROOT / "index" / "npc_crawl_state.json"

#: 律师角色所需的国家级法规。地方法规（codeId 221，15726 篇）明确排除在外。
#: code_ids 取 enumData 里该分类的 ``codeIdList``（含子分类），保证不漏。
CATEGORIES: list[dict] = [
    {"name": "宪法", "code_ids": [100]},
    {"name": "法律", "code_ids": [101, 102, 110, 120, 130, 140, 150, 155,
                                  160, 170, 180, 190, 195, 200]},
    {"name": "行政法规", "code_ids": [201, 210, 215]},
    {"name": "监察法规", "code_ids": [220]},
    {"name": "司法解释", "code_ids": [311, 320, 330, 340, 350]},
]

#: enumData 里 flfgCodeId -> 人类可读的叶子分类名（写进正文头部，便于溯源）
CODE_NAMES = {
    100: "宪法", 101: "法律", 102: "法律", 110: "宪法相关法", 120: "民法商法",
    130: "行政法", 140: "经济法", 150: "社会法", 155: "生态环境法", 160: "刑法",
    170: "诉讼与非诉讼程序法", 180: "法律解释", 190: "有关法律问题和重大问题的决定",
    195: "修正案", 200: "修改、废止的决定", 201: "行政法规", 210: "行政法规",
    215: "行政法规 修改、废止的决定", 220: "监察法规", 311: "司法解释",
    320: "高法司法解释", 330: "高检司法解释", 340: "联合发布司法解释",
    350: "司法解释 修改、废止的决定",
}

_FILENAME_BAD = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def sanitize_filename(name: str, *, max_len: int = 100) -> str:
    """把法规标题变成安全的文件名（Windows/Linux 通用）。"""
    try:
        cleaned = _FILENAME_BAD.sub("_", str(name or "")).replace("\u3000", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(". ")
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len].rstrip(". ")
        return cleaned or "unnamed"
    except Exception:  # noqa: BLE001 - 文件名清洗绝不炸整条链路
        logger.exception("sanitize_filename 失败，退化为 unnamed: %r", name)
        return "unnamed"


class RateLimiter:
    """全局最小请求间隔：把并发请求的**发起时刻**按 min_interval 摊开。"""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> float:
        """阻塞到下一个可用时刻，返回实际睡眠秒数。"""
        if self.min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)
        return sleep_for


class WafAwareRedirectHandler(urllib.request.HTTPRedirectHandler):
    """跟随 307/308 并把请求体一起带过去。

    该站前置了 ``CWAP-waf``：挑战期会对**同一路径**回 302/307 并下发
    ``wzws_cid`` Cookie，客户端必须存下 Cookie 再打一次同一 URL 才能通过。
    urllib 默认对「307 + POST」直接抛 HTTPError（不肯重发请求体），
    于是 POST 全部失败、表现为「接口突然挂了」。这里显式允许重发。
    """

    max_repeats = 6

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        method = req.get_method()
        logger.warning("HTTP %d 重定向：%s -> %s（保留 %s 与请求体；WAF 挑战期属正常）",
                       code, req.full_url, newurl, method)
        if code in (307, 308) and method not in ("GET", "HEAD"):
            return urllib.request.Request(
                newurl, data=req.data, headers=dict(req.header_items()), method=method)
        try:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        except Exception:  # noqa: BLE001 - 3xx 组合没被默认逻辑覆盖时，按原方法重发
            logger.warning("默认重定向处理不支持该组合（%d + %s），按原方法重发：%s",
                           code, method, newurl)
            return urllib.request.Request(
                newurl, data=req.data, headers=dict(req.header_items()), method=method)


# --------------------------------------------------------------------------
# HTTP 客户端
# --------------------------------------------------------------------------

class NpcClient:
    """flk.npc.gov.cn 的轻量客户端：Cookie 会话 + 重试 + 限速 + 数据出入口日志。"""

    def __init__(self, *, timeout: float = 40.0, attempts: int = 3,
                 limiter: RateLimiter | None = None) -> None:
        self.timeout = timeout
        self.attempts = max(1, int(attempts))
        self.limiter = limiter or RateLimiter(0.0)
        self._ctx = ssl.create_default_context()
        # 该站历史上有过证书链过期（wb.flk.npc.gov.cn），直链域名也不同；
        # 这里不做证书强校验，只用于抓公开法规正文。
        self._ctx.check_hostname = False
        self._ctx.verify_mode = ssl.CERT_NONE
        self._headers = {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Referer": BASE + "/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        #: WAF（CWAP-waf）挑战靠 Cookie 通过，必须全程保持会话
        self.cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies),
            urllib.request.HTTPSHandler(context=self._ctx),
            WafAwareRedirectHandler(),
        )

    # ---------- 底层 ----------
    def _open(self, url: str, *, data: bytes | None = None,
              headers: dict | None = None) -> bytes:
        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        request = urllib.request.Request(url, data=data, headers=merged)
        with self._opener.open(request, timeout=self.timeout) as resp:
            self._note_waf(resp.headers, url)
            return resp.read()

    @staticmethod
    def _note_waf(headers, url: str) -> None:
        """把 WAF 的追踪头记进日志，便于事后定位「被封/被挑战」的时间点。"""
        try:
            ray = headers.get("WZWS-RAY") if headers is not None else None
            if ray:
                logger.debug("WAF 放行（WZWS-RAY=%s）：%s", ray, url)
        except Exception:  # noqa: BLE001 - 日志辅助逻辑绝不抛
            logger.debug("读取 WZWS-RAY 失败", exc_info=True)

    def request(self, url: str, *, data: bytes | None = None, headers: dict | None = None,
                operation: str = "http", label: str | None = None) -> bytes:
        """带重试的请求。重试的是**网络层**失败，业务层失败由调用方判断。

        ``label`` 只用于日志：签名直链动辄上千字符，直接打进日志会把
        「按日志快速定位问题」变成翻垃圾堆，因此日志里只留短标签与路径。
        """
        shown = label or urllib.parse.urlsplit(url).path or url
        logger.debug("入口 NpcClient.request(%s, operation=%s, payload=%s 字节)",
                     shown, operation, len(data or b""))
        last_exc: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            self.limiter.wait()
            try:
                with log_io(shown, "flk.npc.gov.cn", operation=operation,
                            size=len(data or b""), meta={"attempt": attempt}) as io:
                    body = self._open(url, data=data, headers=headers)
                    io["size"] = len(body)
                logger.debug("出口 NpcClient.request -> %d 字节（第 %d 次尝试）",
                             len(body), attempt)
                return body
            except Exception as exc:  # noqa: BLE001 - 统一重试所有网络/HTTP 异常
                last_exc = exc
                logger.warning("请求失败（第 %d/%d 次）：%s :: %s: %s",
                               attempt, self.attempts, shown, type(exc).__name__, exc)
                if attempt < self.attempts:
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.0, 0.4)
                    time.sleep(backoff)
        logger.error("请求最终失败：%s（共 %d 次）", shown, self.attempts)
        raise RuntimeError(f"请求失败 {shown}: {type(last_exc).__name__}: {last_exc}") from last_exc

    def warmup(self) -> bool:
        """先取一次首页与分类接口，把 WAF 挑战的 Cookie 拿到手再开始批量抓取。"""
        logger.info("入口 NpcClient.warmup（预热 WAF 会话）")
        ok = False
        for label, path in (("warmup:index", "/index"),
                            ("warmup:aggregateData", "/law-search/index/aggregateData")):
            try:
                body = self.request(BASE + path, operation="warmup", label=label)
                logger.info("预热 %s 成功（%d 字节），当前 Cookie %d 个",
                            label, len(body), len(self.cookies))
                ok = True
            except Exception as exc:  # noqa: BLE001 - 预热失败不阻断，后面照样重试
                logger.warning("预热 %s 失败：%s: %s", label, type(exc).__name__, exc)
        logger.info("出口 NpcClient.warmup -> %s", ok)
        return ok

    def get_json(self, path: str, params: dict | None = None, **kw) -> dict:
        url = path if path.startswith("http") else BASE + path
        if params:
            url = url + "?" + urllib.parse.urlencode(params, doseq=True)
        raw = self.request(url, **kw)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.error("响应不是合法 JSON：%s :: %r", url, raw[:200])
            raise RuntimeError(f"响应不是合法 JSON: {url}") from exc

    def post_json(self, path: str, payload: dict, **kw) -> dict:
        url = path if path.startswith("http") else BASE + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        raw = self.request(url, data=body,
                           headers={"Content-Type": "application/json;charset=utf-8"}, **kw)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.error("响应不是合法 JSON：%s :: %r", url, raw[:200])
            raise RuntimeError(f"响应不是合法 JSON: {url}") from exc



# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------

def _docx_xml_to_text(xml: str) -> str:
    """word/document.xml -> 纯文本（段落换行、单元格制表、去标签、还原实体）。"""
    xml = re.sub(r"<w:tab\b[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br\b[^>]*/>", "\n", xml)
    xml = re.sub(r"</w:tc>", "\t", xml)
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    text = html.unescape(xml)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line.strip())


def docx_to_text(path: Path) -> str:
    """抽取 .docx 正文（纯标准库：zipfile + 正则）。"""
    logger.debug("入口 docx_to_text(%s)", path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "word/document.xml" not in names:
                raise ValueError(f"不是标准 docx（缺 word/document.xml）：{names[:5]}")
            xml = archive.read("word/document.xml").decode("utf-8", "replace")
        text = _docx_xml_to_text(xml)
        logger.debug("出口 docx_to_text -> %d 字符", len(text))
        return text
    except Exception:
        logger.exception("docx_to_text 失败：%s", path)
        raise


def pdf_to_text(path: Path) -> str:
    """抽取 .pdf 正文（可选依赖 pypdf；没有就明确报错，不静默返回空）。"""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("解析 PDF 需要 pypdf（pip install pypdf）") from exc
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - 单页失败不影响整篇
            logger.warning("PDF 第 %d 页抽取失败：%s", index, path)
            pages.append("")
    return "\n\n".join(pages)


def sniff_container(blob: bytes) -> str:
    """按文件头判断真实容器类型（**不信 `format=docx` 参数的承诺**）。

    实测有文档在 `format=docx` 下返回的是**老式 `.doc`**（OLE2 复合文档，
    magic `D0 CF 11 E0`），不是 zip 版 docx —— 真机跑才暴露的那类问题。
    """
    if not blob:
        return ""
    if blob[:2] == b"PK":
        return "docx"
    if blob[:4] == b"%PDF":
        return "pdf"
    if blob[:4] == b"\xd0\xcf\x11\xe0":
        return "ole2"
    return "unknown"


#: 容器类型的可读名（日志用）
CONTAINER_LABEL = {"docx": "zip 版 .docx", "pdf": "PDF",
                   "ole2": "老式 .doc（OLE2）", "unknown": "未知格式"}


def build_markdown(row: dict, body: str, *, category: str, source_url: str,
                   fetched_at: str) -> str:
    """把元数据 + 正文拼成 Markdown（元数据留在正文里，检索时可见、引用时可溯源）。"""
    title = (row.get("title") or "").strip()
    code = row.get("flfgCodeId")
    leaf = CODE_NAMES.get(code, "")
    meta = [
        ("类别", f"{category} / {leaf}" if leaf and leaf != category else category),
        ("法律性质", row.get("flxz")),
        ("制定机关", row.get("zdjgName")),
        ("公布日期", row.get("gbrq")),
        ("施行日期", row.get("sxrq")),
        ("时效性", row.get("sxx")),
        ("法规编号", row.get("bbbs")),
        ("数据来源", "国家法律法规数据库（全国人大常委会）"),
        ("来源链接", source_url),
        ("抓取时间", fetched_at),
        ("归属角色", "lawyer（律师）"),
    ]
    head = [f"# {title}", ""]
    for key, value in meta:
        if value not in (None, "", []):
            head.append(f"- {key}：{value}")
    head += ["", "---", ""]
    return "\n".join(head) + body.strip() + "\n"


# --------------------------------------------------------------------------
# 抓取器
# --------------------------------------------------------------------------

class LawCrawler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = NpcClient(timeout=args.timeout, attempts=args.attempts,
                                limiter=RateLimiter(args.min_interval))
        self.state_path = Path(args.state)
        self.state: dict = {"version": 1, "docs": {}, "categories": {}}
        self._lock = threading.Lock()
        self._since_save = 0
        self._used_names: dict[str, str] = {}

    # ---------- 状态 ----------
    def load_state(self) -> None:
        logger.info("入口 LawCrawler.load_state(%s)", self.state_path)
        if self.args.rebuild_state:
            logger.warning("--rebuild-state：忽略既有进度，全量重抓")
            return
        if not self.state_path.is_file():
            logger.info("状态文件不存在，按全新抓取处理")
            return
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state.setdefault("docs", {})
            self.state.setdefault("categories", {})
            for bbbs, record in self.state["docs"].items():
                name = record.get("name")
                if name and record.get("status") == "ok":
                    self._used_names[name] = bbbs
            logger.info("状态已载入：%d 篇记录（成功 %d / 失败 %d）",
                        len(self.state["docs"]),
                        sum(1 for r in self.state["docs"].values() if r.get("status") == "ok"),
                        sum(1 for r in self.state["docs"].values() if r.get("status") == "fail"))
        except Exception:  # noqa: BLE001 - 坏状态文件不该阻断抓取
            logger.exception("状态文件损坏，按全新抓取处理：%s", self.state_path)
            self.state = {"version": 1, "docs": {}, "categories": {}}

    def save_state(self, *, force: bool = False) -> None:
        with self._lock:
            self._since_save += 1
            if not force and self._since_save < max(1, self.args.save_every):
                return
            self._since_save = 0
            snapshot = dict(self.state)
        snapshot["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception:  # noqa: BLE001
            logger.exception("写状态文件失败：%s", self.state_path)

    # ---------- 列表 ----------
    def list_category(self, category: dict) -> list[dict]:
        """翻页列出某分类下的全部法规（返回列表行）。"""
        name, code_ids = category["name"], category["code_ids"]
        logger.info("入口 list_category(%s, code_ids=%s)", name, code_ids)
        rows: list[dict] = []
        page = 1
        total = None
        while True:
            payload = {
                "searchContent": "", "searchType": 1, "searchRange": 1,
                "flfgCodeId": list(code_ids), "zdjgCodeId": [], "gbrqYear": [],
                "orderByParam": {"order": "-1", "sort": ""},
                "pageNum": page, "pageSize": self.args.page_size,
            }
            data = self.client.post_json("/law-search/search/list", payload,
                                         operation=f"list:{name}:p{page}",
                                         label=f"list:{name}:p{page}")
            if not isinstance(data, dict) or data.get("code") != 200:
                raise RuntimeError(f"列表接口返回异常（{name} 第 {page} 页）：{data}")
            batch = data.get("rows") or []
            if total is None:
                total = int(data.get("total") or 0)
                logger.info("分类 %s：接口报告 total=%d", name, total)
            rows.extend(batch)
            logger.info("分类 %s：第 %d 页取回 %d 条（累计 %d/%s）",
                        name, page, len(batch), len(rows), total)
            if self.args.limit and len(rows) >= self.args.limit:
                rows = rows[: self.args.limit]
                break
            if len(batch) < self.args.page_size or (total and len(rows) >= total):
                break
            page += 1
        with self._lock:
            self.state["categories"][name] = {"total": total, "listed": len(rows),
                                              "code_ids": list(code_ids),
                                              "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        logger.info("出口 list_category(%s) -> %d 条", name, len(rows))
        return rows

    # ---------- 单篇 ----------
    def detail(self, bbbs: str) -> dict:
        data = self.client.get_json("/law-search/search/flfgDetails", {"bbbs": bbbs},
                                    operation="detail", label=f"detail:{bbbs[:8]}")
        if not isinstance(data, dict) or data.get("code") != 200:
            raise RuntimeError(f"详情接口返回异常：{data}")
        return data.get("data") or {}

    def download_url(self, bbbs: str, file_id: str, fmt: str) -> str:
        data = self.client.get_json("/law-search/download/pc",
                                    {"format": fmt, "bbbs": bbbs, "fileId": file_id or ""},
                                    operation=f"download-pc:{fmt}",
                                    label=f"download-pc:{fmt}:{bbbs[:8]}")
        if not isinstance(data, dict) or data.get("code") != 200:
            raise RuntimeError(f"下载接口返回异常：{data}")
        return str((data.get("data") or {}).get("url") or "")

    def _fetch_file(self, url: str, fmt: str, title: str) -> bytes:
        """下载文件本体（签名直链）。"""
        return self.client.request(url, operation=f"file:{fmt}",
                                   label=f"file:{fmt}:{title[:24]}")

    def _detail_file_id(self, bbbs: str, title: str, record: dict) -> str:
        """取详情里的 ``fileId``：有些条目必须带上它才给下载直链。失败不致命。"""
        try:
            detail = self.detail(bbbs)
            file_id = str(detail.get("fileId") or "")
            if file_id:
                record["fileId"] = file_id
            return file_id
        except Exception as exc:  # noqa: BLE001 - 兜底信息拿不到就继续走原路
            logger.warning("取详情兜底失败（%s）：%s: %s", title, type(exc).__name__, exc)
            return ""

    def _save_raw(self, blob: bytes, category: dict, record: dict, *, ext: str) -> Path:
        """原始件落盘（**后缀按真实容器类型**，避免把 .doc 存成 .docx）。"""
        raw_path = RAW_DIR / category["name"] / f"{record['name']}.{ext}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(blob)
        record["raw"] = str(raw_path.relative_to(ROOT)).replace("\\", "/")
        return raw_path

    def fetch_one(self, row: dict, category: dict) -> dict:
        """抓取一篇：取直链 -> 下载 -> 抽取正文 -> 落盘。返回状态记录。"""
        bbbs = str(row.get("bbbs") or "")
        title = (row.get("title") or bbbs).strip()
        logger.info("入口 fetch_one(%s :: %s)", category["name"], title)
        record: dict = {
            "status": "fail", "category": category["name"], "title": title,
            "bbbs": bbbs, "gbrq": row.get("gbrq"), "sxrq": row.get("sxrq"),
            "sxx": row.get("sxx"), "zdjgName": row.get("zdjgName"),
            "flxz": row.get("flxz"), "flfgCodeId": row.get("flfgCodeId"),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        started = time.perf_counter()
        try:
            file_id = ""
            # ---- 1) 先要 docx 直链 ----
            url = self.download_url(bbbs, file_id, "docx")
            if not url:
                # 少数法规必须带 fileId 才给直链：从详情兜底取一次
                file_id = self._detail_file_id(bbbs, title, record)
                url = self.download_url(bbbs, file_id, "docx")

            blob = self._fetch_file(url, "docx", title) if url else b""
            kind = sniff_container(blob)

            # ---- 2) 不是 zip 版 docx 就回退 pdf ----
            # 注意：`format=docx` 并不保证真给 docx（实测有老式 .doc），
            # 所以判断依据是**文件头**，不是接口参数。
            if kind != "docx":
                if blob:
                    logger.warning("%s：docx 直链给的是 %s（magic=%s，%d 字节），回退 pdf",
                                   title, CONTAINER_LABEL.get(kind, "未知"),
                                   blob[:4].hex(), len(blob))
                    if kind == "ole2":
                        # 老式 .doc 原件按**真实后缀**留档：将来接 OCR / 老式解析器可直接复用，
                        # 也避免"把 .doc 存成 .docx"这种名不符实。
                        record["name"] = self._unique_name(title, bbbs)
                        self._save_raw(blob, category, record, ext="doc")
                        record["raw_doc"] = record.pop("raw")
                if not file_id:
                    file_id = self._detail_file_id(bbbs, title, record)
                pdf_url = self.download_url(bbbs, file_id, "pdf")
                if pdf_url:
                    blob = self._fetch_file(pdf_url, "pdf", title)
                    kind = sniff_container(blob)
                    record["fallback"] = "pdf"
                elif kind == "ole2":
                    # 老式 .doc 且无 pdf 可退：抽取需要额外依赖，明确记跳过（不静默丢），
                    # 但**原始件按真实后缀留档**，将来换解析器可以直接复用。
                    record["name"] = self._unique_name(title, bbbs)
                    record["bytes"] = len(blob)          # 记录体积：排查/排优先级都要用
                    record["container"] = kind
                    self._save_raw(blob, category, record, ext="doc")
                    record["status"] = "skip"
                    record["reason"] = "legacy_doc_no_pdf"
                    logger.warning("%s：只有老式 .doc 且无 pdf 直链，原始件已存 .doc，跳过", title)
                    return record
                elif not blob:
                    record["status"] = "skip"
                    record["reason"] = "no_url"
                    logger.warning("无可用下载直链，跳过：%s", title)
                    return record
                else:
                    raise ValueError(f"下载内容不是可解析格式（magic={blob[:4]!r}，"
                                     f"{len(blob)} 字节）")

            record["bytes"] = len(blob)
            record["format"] = "pdf" if kind == "pdf" else "docx"
            record["container"] = kind
            record["name"] = self._unique_name(title, bbbs)
            raw_path = self._save_raw(blob, category, record, ext=record["format"])

            body = pdf_to_text(raw_path) if record["format"] == "pdf" else docx_to_text(raw_path)
            if len(body.strip()) < 20:
                if record["format"] == "pdf":
                    # 扫描件（无文本层）：**不是"重试就能好"的错误**，而是需要 OCR 的另一类能力，
                    # 所以记 skip 并写明原因（原始件已留档），不污染"失败"清单。
                    record["status"] = "skip"
                    record["reason"] = "pdf_no_text_layer"
                    logger.warning("%s：PDF 无文本层（疑扫描件），原件已留档并跳过；需 OCR 才能入库",
                                   title)
                    return record
                raise ValueError(f"正文抽取结果过短（{len(body.strip())} 字符），疑似空文档")
            record["chars"] = len(body)

            markdown = build_markdown(
                row, body, category=category["name"],
                source_url=f"{BASE}/detail?bbbs={bbbs}",
                fetched_at=record["at"])
            text_path = TEXT_DIR / category["name"] / f"{record['name']}.md"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(markdown, encoding="utf-8")
            record["text"] = str(text_path.relative_to(ROOT)).replace("\\", "/")
            record["status"] = "ok"
            record.pop("reason", None)
        except Exception as exc:  # noqa: BLE001 - 单篇失败绝不影响整体
            record["reason"] = f"{type(exc).__name__}: {exc}"
            logger.exception("抓取失败：%s :: %s", title, bbbs)
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            with self._lock:
                self.state["docs"][bbbs] = record
            self.save_state()
            logger.info("出口 fetch_one(%s) -> %s（%.1fms）", title, record["status"],
                        record["elapsed_ms"])
        return record

    def _unique_name(self, title: str, bbbs: str) -> str:
        """文件名冲突处理：重名时补**完整 bbbs**（主键，全局唯一），保证 manifest/doc_id 不撞车。

        ⚠️ 别用 ``bbbs[:8]``：这批 id 是「时间戳前缀 + 随机尾」结构，
        前 8 位大量相同（实测 ``ff808081`` / ``ff808181`` 反复出现），
        于是"冲突后缀"自己也会冲突 —— 曾把 9 组同名的不同版本法规写成同一个文件
        （见 ``scripts/fix_name_collisions.py`` 的来龙去脉）。
        用完整 bbbs 是确定性的：同一篇每次跑都得到同一个名字，且不同版本必然不同名。
        """
        base = sanitize_filename(title)
        with self._lock:
            owner = self._used_names.get(base)
            if owner is None or owner == bbbs:
                self._used_names[base] = bbbs
                return base
            alt = f"{base}__{bbbs}"
            self._used_names[alt] = bbbs
            logger.warning("文件名冲突：%r 已被 %s 占用，改用 %r", base, owner, alt)
            return alt

    # ---------- 主流程 ----------
    def run(self) -> int:
        args = self.args
        self.load_state()
        self.client.warmup()
        if args.retry_failed:
            targets = [c for c in CATEGORIES if not args.only or c["name"] in args.only]
            failed = [bbbs for bbbs, r in self.state["docs"].items()
                      if r.get("status") == "fail"
                      and (not args.only or r.get("category") in args.only)]
            logger.info("--retry-failed：待重试 %d 篇", len(failed))
            rows = [self.state["docs"][b] for b in failed]
            for row in rows:
                row.setdefault("bbbs", row.get("bbbs"))
            pending = [(r, {"name": r.get("category") or "未分类"}) for r in rows]
            return self._consume(pending, total_hint=len(pending))

        pending: list[tuple[dict, dict]] = []
        for category in CATEGORIES:
            if args.only and category["name"] not in args.only:
                continue
            try:
                rows = self.list_category(category)
            except Exception:  # noqa: BLE001 - 一个分类列不出来不影响其它分类
                logger.exception("分类列表失败：%s", category["name"])
                continue
            fresh = []
            for row in rows:
                bbbs = str(row.get("bbbs") or "")
                if not bbbs:
                    logger.warning("列表行缺少 bbbs，跳过：%r", row.get("title"))
                    continue
                previous = self.state["docs"].get(bbbs)
                if previous and previous.get("status") == "ok" and not args.force:
                    continue
                fresh.append((row, category))
            pending.extend(fresh)
            logger.info("分类 %s：待抓 %d 篇（列表 %d 篇，其余已完成）",
                        category["name"], len(fresh), len(rows))
        if args.list_only:
            logger.info("--list-only：只列目录，结束")
            self.save_state(force=True)
            return 0
        return self._consume(pending, total_hint=len(pending))

    def _consume(self, pending: list[tuple[dict, dict]], *, total_hint: int) -> int:
        if not pending:
            logger.info("没有待抓篇目（全部已完成）。")
            return 0
        logger.info("开始抓取：%d 篇，并发 %d，最小间隔 %.2fs",
                    len(pending), self.args.workers, self.args.min_interval)
        ok = fail = skip = 0
        started = time.perf_counter()
        counter = 0
        with ThreadPoolExecutor(max_workers=max(1, self.args.workers)) as pool:
            futures = {pool.submit(self.fetch_one, row, category): row
                       for row, category in pending}
            try:
                for future in as_completed(futures):
                    counter += 1
                    row = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - 兜底，理论上 fetch_one 不抛
                        logger.exception("工作线程异常：%s", row.get("title"))
                        record = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
                    status = record.get("status")
                    ok += status == "ok"
                    skip += status == "skip"
                    fail += status == "fail"
                    elapsed = time.perf_counter() - started
                    rate = counter / elapsed if elapsed > 0 else 0.0
                    remaining = (total_hint - counter) / rate if rate > 0 else 0.0
                    print(f"[{counter}/{total_hint}] {status.upper():4} "
                          f"{record.get('category', '?')} :: {record.get('title', '?')}"
                          f"  ({record.get('chars', 0)} 字, {record.get('elapsed_ms', 0)}ms)"
                          f"  已用 {elapsed / 60:.1f}min 预计剩余 {remaining / 60:.1f}min",
                          flush=True)
            except KeyboardInterrupt:
                logger.warning("收到中断，正在保存进度（已完成 %d/%d）...", counter, total_hint)
                for future in futures:
                    future.cancel()
                self.save_state(force=True)
                raise
        self.save_state(force=True)
        logger.info("抓取结束：成功 %d / 跳过 %d / 失败 %d，总耗时 %.1f 分钟",
                    ok, skip, fail, (time.perf_counter() - started) / 60.0)
        print(f"\n抓取结束：成功 {ok} / 跳过 {skip} / 失败 {fail}", flush=True)
        return 0 if fail == 0 else 1


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取国家法律法规数据库（律师角色知识）")
    parser.add_argument("--only", nargs="*", default=None,
                        metavar="类别", help="只抓这些分类（宪法 法律 行政法规 监察法规 司法解释）")
    parser.add_argument("--limit", type=int, default=0, help="每个分类最多抓多少篇（0=不限，小样验证用）")
    parser.add_argument("--workers", type=int, default=2, help="并发线程数（默认 2，礼貌起见别开太大）")
    parser.add_argument("--min-interval", type=float, default=0.8, dest="min_interval",
                        help="全局最小请求间隔秒数（默认 0.8；该站有 WAF，调太快会被挑战）")
    parser.add_argument("--page-size", type=int, default=100, dest="page_size", help="列表每页条数")
    parser.add_argument("--timeout", type=float, default=40.0, help="单次请求超时秒数")
    parser.add_argument("--attempts", type=int, default=3, help="单次请求最大尝试次数")
    parser.add_argument("--state", default=str(STATE_PATH), help="进度文件路径")
    parser.add_argument("--save-every", type=int, default=10, dest="save_every",
                        help="每完成多少篇落一次进度（默认 10）")
    parser.add_argument("--list-only", action="store_true", dest="list_only",
                        help="只列目录并统计篇数，不下载")
    parser.add_argument("--force", action="store_true", help="忽略进度，重抓已成功的篇目")
    parser.add_argument("--rebuild-state", action="store_true", dest="rebuild_state",
                        help="丢弃既有进度文件，从头开始")
    parser.add_argument("--retry-failed", action="store_true", dest="retry_failed",
                        help="只重试上一轮失败的篇目")
    return parser


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据源     : {BASE}")
    print(f"分类       : {', '.join(c['name'] for c in CATEGORIES)}（地方法规明确排除）")
    print(f"原始件目录 : {RAW_DIR}")
    print(f"正文目录   : {TEXT_DIR}（入库请带 --role-id lawyer）")
    print(f"进度文件   : {args.state}")
    print(f"并发/间隔  : {args.workers} / {args.min_interval}s")
    print("-" * 60)
    crawler = LawCrawler(args)
    try:
        return crawler.run()
    except KeyboardInterrupt:
        print("\n已中断，进度已保存；再次运行即可续抓。", flush=True)
        return 130
    except Exception:  # noqa: BLE001 - 顶层兜底：把栈留在日志里
        logger.exception("抓取流程异常终止")
        return 1


if __name__ == "__main__":
    sys.exit(main())
